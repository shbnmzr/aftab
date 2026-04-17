import torch
import numpy
import time
from ..functions import flush, lambda_returns


class CategoricalTrainMixin:
    def __init__(self):
        super().__init__()

    def __get_q_values(self, float_observations, gradient: bool = False):
        if not gradient:
            with torch.no_grad():
                q_values = self._network.get_q(float_observations)
        else:
            q_values = self._network.get_q(float_observations)
        return q_values

    def __get_returns(
        self, float_observations, batch_q, batch_rewards, batch_terminations
    ):
        with (
            torch.no_grad(),
            torch.autocast(device_type=self.device.type, dtype=torch.float16),
        ):
            next_q = self._network.get_q(float_observations).max(dim=-1).values
            max_q_seq = batch_q.max(dim=-1).values
            q_seq_for_lambda = torch.cat([max_q_seq, next_q.unsqueeze(0)])
            returns = lambda_returns(
                batch_rewards,
                batch_terminations,
                q_seq_for_lambda[1:],
                self.gamma,
                self.lmbda,
            )
        return returns

    def __quantile_value_loss(self, observations, actions, targets):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            output = self._network(observations)
            quantiles = output["quantiles"]
            tau_hats = output["tau_hats"]
            action_idx = actions.unsqueeze(-1).unsqueeze(-1)
            predictions = torch.take_along_dim(
                quantiles,
                action_idx.expand(-1, quantiles.size(1), 1),
                dim=-1,
            ).squeeze(-1)
            return self._network.quantile_value_loss(predictions, targets, tau_hats)

    def __fraction_proposal_loss(self, observations, actions):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            output = self._network(observations)
            features = output["features"]
            quantiles = output["quantiles"].detach()
            taus = output["taus"]
            q_probs = output["q_probs"]
            entropy = output["entropy"]

            fraction_loss = self._network.fraction_proposal_loss(
                features, actions, quantiles, taus, q_probs
            )
            entropy_loss = -entropy.mean()
            total_fraction_loss = (
                self._network.fraction_proposal_coefficient * fraction_loss
                + self._network.entropy_coefficient * entropy_loss
            )
            return total_fraction_loss

    def categorical_train(self, environment: str, seed: int):
        frame_count = 0

        self.flush_results()
        self.set_precision()
        self.set_seed(seed)

        episode_returns = numpy.zeros(self.total_environments, dtype=numpy.float32)
        train_environment, test_environment, action_dimension, observation_shape = (
            self.make_environments(environment=environment, seed=seed)
        )
        self.prepare_network(action_dimension=action_dimension)
        optimizer = self.make_optimizer()

        observation_train, _ = train_environment.reset()
        observation_test, _ = test_environment.reset()
        observation = numpy.concatenate([observation_train, observation_test], axis=0)
        observation = torch.from_numpy(observation).to(torch.uint8).to(self.device)
        (
            batch_observations,
            batch_actions,
            batch_rewards,
            batch_terminations,
            batch_q,
        ) = self.make_batches(
            observation_shape=observation_shape, action_dimension=action_dimension
        )
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        training_start_time = time.time()

        for update in range(1, self.total_updates + 1):
            self._network.eval()

            for step in range(self.steps_per_update):
                float_observations = observation.float()
                epsilon_value = self._network.epsilon.get(
                    frame_count,
                    self.actual_frames,
                )
                q_values = self.__get_q_values(
                    float_observations=float_observations, gradient=False
                )
                actions = self.get_actions(
                    q_values=q_values, epsilon_value=epsilon_value
                )
                actions_train, actions_test = self.split_actions(actions)

                (
                    next_observation_train,
                    reward_train,
                    termination_train,
                    truncation_train,
                    info_train,
                ) = train_environment.step(actions_train)

                (
                    next_observation_test,
                    reward_test,
                    termination_test,
                    truncation_test,
                    info_test,
                ) = test_environment.step(actions_test)

                next_observation = numpy.concatenate(
                    [next_observation_train, next_observation_test], axis=0
                )
                rewards = numpy.concatenate([reward_train, reward_test], axis=0)
                terminations = numpy.concatenate(
                    [termination_train, termination_test], axis=0
                )
                truncations = numpy.concatenate(
                    [truncation_train, truncation_test], axis=0
                )
                terminations = numpy.logical_or(terminations, truncations)

                reward_train = info_train.get("reward", None)
                reward_test = info_test.get("reward", None)
                if reward_train is not None and reward_test is not None:
                    if isinstance(reward_train, numpy.ndarray) and isinstance(
                        reward_test, numpy.ndarray
                    ):
                        if reward_train.ndim == 0:
                            rewards = numpy.stack([reward_train, reward_test])
                        else:
                            rewards = numpy.concatenate(
                                [reward_train, reward_test], axis=0
                            )
                        episode_returns += rewards

                done_mask = terminations
                if numpy.any(done_mask):
                    idx = numpy.nonzero(done_mask)[0]
                    scores = episode_returns[done_mask]
                    split = idx < self.num_train_environments
                    self.results.rewards.train.extend(scores[split].tolist())
                    self.results.rewards.test.extend(scores[~split].tolist())
                    episode_returns[done_mask] = 0

                batch_observations[step] = observation
                batch_actions[step] = torch.from_numpy(actions).to(self.device)
                batch_rewards[step] = torch.from_numpy(rewards).to(self.device)
                batch_terminations[step] = torch.from_numpy(terminations).to(
                    self.device
                )
                batch_q[step] = q_values

                observation = (
                    torch.from_numpy(next_observation).to(torch.uint8).to(self.device)
                )
                frame_count += self.num_train_environments

            targets = self.__get_returns(
                float_observations=observation.float(),
                batch_q=batch_q,
                batch_rewards=batch_rewards,
                batch_terminations=batch_terminations,
            )

            train_slice = slice(0, self.num_train_environments)
            flattened_observations = batch_observations[
                :, : self.num_train_environments
            ].reshape((-1,) + observation_shape)
            flattened_actions = batch_actions[:, train_slice].reshape(-1)
            flattened_targets = targets[:, train_slice].reshape(-1)

            value_parameters = list(self._network.phi.parameters()) + list(
                self._network.quantile_value.parameters()
            )
            fraction_parameters = list(self._network.fraction_proposal.parameters())

            self._network.train()
            for _ in range(self.epochs):
                indices = torch.randperm(self.batch_size, device=self.device)

                for range_start in range(0, self.batch_size, self.minibatch_size):
                    range_end = range_start + self.minibatch_size
                    mini_batch_idx = indices[range_start:range_end]
                    mini_batch_observations = flattened_observations[mini_batch_idx]
                    mini_batch_actions = flattened_actions[mini_batch_idx]
                    mini_batch_targets = flattened_targets[mini_batch_idx]

                    optimizer.quantile_value.zero_grad(set_to_none=True)
                    value_loss = self.__quantile_value_loss(
                        observations=mini_batch_observations.float(),
                        actions=mini_batch_actions,
                        targets=mini_batch_targets,
                    )
                    scaler.scale(value_loss).backward()
                    scaler.unscale_(optimizer.quantile_value)
                    torch.nn.utils.clip_grad_norm_(
                        value_parameters,
                        self.gradient_norm,
                    )
                    scaler.step(optimizer.quantile_value)

                    optimizer.fraction_proposal.zero_grad(set_to_none=True)
                    fraction_loss = self.__fraction_proposal_loss(
                        observations=mini_batch_observations.float(),
                        actions=mini_batch_actions,
                    )
                    scaler.scale(fraction_loss).backward()
                    scaler.unscale_(optimizer.fraction_proposal)
                    torch.nn.utils.clip_grad_norm_(
                        fraction_parameters,
                        self.gradient_norm,
                    )
                    scaler.step(optimizer.fraction_proposal)
                    scaler.update()
                    self.results.loss.append((value_loss + fraction_loss).item())

            test_score = (
                0.0
                if len(self.results.rewards.test) < 10
                else numpy.mean(self.results.rewards.test[-10:])
            )

            if self.verbose and update % self.log_interval == 0:
                flush(f"Update {update} | Frames: {frame_count}")
                flush(
                    f"Test Score: {test_score:.4f}",
                )

        train_environment.close()
        test_environment.close()
        self.results.duration = time.time() - training_start_time

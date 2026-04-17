import torch
import numpy
import time
from ..functions import flush


class RegressionTrainMixin:
    def __init__(self):
        super().__init__()

    def regression_train(self, environment, seed: int):
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
                q_values = self.get_q_values(
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

            targets = self.get_returns(
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

            self._network.train()
            for _ in range(self.epochs):
                indices = torch.randperm(self.batch_size, device=self.device)

                for range_start in range(0, self.batch_size, self.minibatch_size):
                    range_end = range_start + self.minibatch_size
                    mini_batch_idx = indices[range_start:range_end]
                    mini_batch_observations = flattened_observations[mini_batch_idx]
                    mini_batch_actions = flattened_actions[mini_batch_idx]
                    mini_batch_targets = flattened_targets[mini_batch_idx]

                    optimizer.zero_grad(set_to_none=True)
                    loss = self.get_loss(
                        mini_batch_observations=mini_batch_observations,
                        mini_batch_targets=mini_batch_targets,
                        mini_batch_actions=mini_batch_actions,
                    )
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._network.parameters(), self.gradient_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    self.results.loss.append(loss.item())

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

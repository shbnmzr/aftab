import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import envpool
from typing import Any, NamedTuple


# ---------------------------------------------------------
# 1. Dataclasses & Replay Buffer Transition
# ---------------------------------------------------------
class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_obs: jnp.ndarray
    q_val: jnp.ndarray


class CustomTrainState(TrainState):
    batch_stats: Any


# ---------------------------------------------------------
# 2. Neural Networks (Flax -> equivalent to PyTorch nn.Module)
# ---------------------------------------------------------
class CNN(nn.Module):
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        def normalize(inputs):
            if self.norm_type == "layer_norm":
                return nn.LayerNorm()(inputs)
            elif self.norm_type == "batch_norm":
                return nn.BatchNorm(use_running_average=not train)(inputs)
            return inputs

        # PyTorch style forward pass
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)

        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)

        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


class QNetwork(nn.Module):
    action_dim: int
    norm_type: str = "layer_norm"
    norm_input: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        # NHWC format for JAX standard
        x = jnp.transpose(x, (0, 2, 3, 1))
        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)
            x = x / 255.0

        x = CNN(norm_type=self.norm_type)(x, train)
        x = nn.Dense(self.action_dim)(x)
        return x


# ---------------------------------------------------------
# 3. JIT-Compiled Training Components
# ---------------------------------------------------------


@jax.jit
def select_action(
    train_state: CustomTrainState,
    obs: jnp.ndarray,
    eps: jnp.ndarray,
    rng: jax.random.PRNGKey,
):
    """Epsilon-greedy action selection."""
    rng_a, rng_e = jax.random.split(rng)

    # Forward pass (eval mode)
    q_vals = train_state.apply_fn(
        {"params": train_state.params, "batch_stats": train_state.batch_stats},
        obs,
        train=False,
    )

    greedy_actions = jnp.argmax(q_vals, axis=-1)
    random_actions = jax.random.randint(
        rng_a, greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
    )

    # Where random condition is met, pick random_action, else greedy
    is_random = jax.random.uniform(rng_e, greedy_actions.shape) < eps
    actions = jnp.where(is_random, random_actions, greedy_actions)

    return actions, q_vals


@jax.jit
def compute_targets(last_q, q_vals, rewards, dones, gamma, lambda_):
    """Computes lambda returns (GAE-style) running backwards."""

    def _get_target(carry, transition):
        lambda_returns, next_q = carry
        reward, q, done = transition

        target_bootstrap = reward + gamma * (1 - done) * next_q
        delta = lambda_returns - next_q
        lambda_returns = target_bootstrap + gamma * lambda_ * delta
        lambda_returns = (1 - done) * lambda_returns + done * reward
        next_q = jnp.max(q, axis=-1)

        return (lambda_returns, next_q), lambda_returns

    lambda_returns = rewards[-1] + gamma * (1 - dones[-1]) * last_q
    last_q = jnp.max(q_vals[-1], axis=-1)

    # Scan backwards over the sequence (excluding the very last step initialized above)
    _, targets = jax.lax.scan(
        _get_target,
        (lambda_returns, last_q),
        (rewards[:-1], q_vals[:-1], dones[:-1]),
        reverse=True,
    )

    # Append the last step back to the targets
    targets = jnp.concatenate([targets, lambda_returns[None, ...]])
    return targets


@jax.jit
def train_step(
    train_state: CustomTrainState,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    targets: jnp.ndarray,
):
    """A single PyTorch-style minibatch gradient update."""

    def loss_fn(params):
        q_vals, updates = train_state.apply_fn(
            {"params": params, "batch_stats": train_state.batch_stats},
            obs,
            train=True,
            mutable=["batch_stats"],
        )

        # Gather Q-values for the actions we actually took: Q(s, a)
        chosen_action_qvals = jnp.take_along_axis(
            q_vals, jnp.expand_dims(actions, axis=-1), axis=-1
        ).squeeze(axis=-1)

        loss = 0.5 * jnp.square(chosen_action_qvals - targets).mean()
        return loss, updates

    # Compute gradients and updates
    (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        train_state.params
    )

    # Apply gradients (PyTorch optimizer.step())
    train_state = train_state.apply_gradients(grads=grads)
    train_state = train_state.replace(batch_stats=updates["batch_stats"])

    return train_state, loss


# ---------------------------------------------------------
# 4. Main Training Loop (PyTorch Style)
# ---------------------------------------------------------
def main():
    # --- Configuration ---
    config = {
        "ENV_NAME": "Pong-v5",
        "SEED": 42,
        "NUM_ENVS": 8,
        "TEST_ENVS": 2,
        "TEST_DURING_TRAINING": True,
        "NUM_STEPS": 128,  # Steps per rollout
        "NUM_MINIBATCHES": 4,  # Minibatches per epoch
        "NUM_EPOCHS": 3,  # Epochs per PPO/PQN update
        "TOTAL_TIMESTEPS": 10_000_000,
        "LR": 3e-4,
        "MAX_GRAD_NORM": 0.5,
        "GAMMA": 0.99,
        "LAMBDA": 0.95,
        "EPS_START": 1.0,
        "EPS_FINISH": 0.05,
        "EPS_DECAY": 0.5,
        "NORM_TYPE": "layer_norm",
    }

    # Derived sizes
    total_envs = (
        config["NUM_ENVS"] + config["TEST_ENVS"]
        if config["TEST_DURING_TRAINING"]
        else config["NUM_ENVS"]
    )
    num_updates = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
    minibatch_size = batch_size // config["NUM_MINIBATCHES"]

    # --- Setup Env ---
    env = envpool.make(
        config["ENV_NAME"], env_type="gym", num_envs=total_envs, seed=config["SEED"]
    )
    obs, env_state = env.reset()

    # --- Setup Agent & Optimizer ---
    rng = jax.random.PRNGKey(config["SEED"])
    rng, init_rng = jax.random.split(rng)

    network = QNetwork(action_dim=env.action_space.n, norm_type=config["NORM_TYPE"])
    init_obs = jnp.zeros((1, *env.observation_space.shape))
    variables = network.init(init_rng, init_obs, train=False)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.radam(learning_rate=config["LR"]),
    )

    train_state = CustomTrainState.create(
        apply_fn=network.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=optimizer,
    )

    # Epsilon scheduler setup
    eps_decay_steps = int(config["EPS_DECAY"] * num_updates)
    eps_scheduler = optax.linear_schedule(
        config["EPS_START"], config["EPS_FINISH"], eps_decay_steps
    )

    # --- PyTorch-style Training Loop ---
    for update in range(num_updates):
        transitions = []

        # ---------------------------------------------------
        # Phase 1: Environment Rollouts
        # ---------------------------------------------------
        for step in range(config["NUM_STEPS"]):
            rng, action_rng = jax.random.split(rng)

            # Assign epsilon: Exploration for train envs, 0.0 for test envs
            current_eps = eps_scheduler(update)
            eps_array = jnp.full(config["NUM_ENVS"], current_eps)
            if config["TEST_DURING_TRAINING"]:
                eps_array = jnp.concatenate((eps_array, jnp.zeros(config["TEST_ENVS"])))

            # Step agent
            actions, q_vals = select_action(train_state, obs, eps_array, action_rng)
            next_obs, next_env_state, reward, done, info = env.step(env_state, actions)

            # Store transition
            transitions.append(Transition(obs, actions, reward, done, next_obs, q_vals))

            obs = next_obs
            env_state = next_env_state

        # Stack list of transitions into unified JAX arrays (PyTorch `torch.stack` equivalent)
        batch = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *transitions)

        # ---------------------------------------------------
        # Phase 2: Compute Targets
        # ---------------------------------------------------
        # Discard test envs for gradient calculation targets
        if config["TEST_DURING_TRAINING"]:
            train_batch = jax.tree_util.tree_map(
                lambda x: x[:, : -config["TEST_ENVS"]], batch
            )
        else:
            train_batch = batch

        # Get Q-value of the final state for bootstrapping
        last_q_vals = train_state.apply_fn(
            {"params": train_state.params, "batch_stats": train_state.batch_stats},
            train_batch.next_obs[-1],
            train=False,
        )
        last_q = jnp.max(last_q_vals, axis=-1)

        targets = compute_targets(
            last_q,
            train_batch.q_val,
            train_batch.reward,
            train_batch.done,
            config["GAMMA"],
            config["LAMBDA"],
        )

        # ---------------------------------------------------
        # Phase 3: Optimize / Epoch Loop
        # ---------------------------------------------------
        # Flatten time and env dimensions: (Num_Steps, Num_Envs, ...) -> (Batch_Size, ...)
        flat_obs = train_batch.obs.reshape(-1, *train_batch.obs.shape[2:])
        flat_actions = train_batch.action.reshape(-1)
        flat_targets = targets.reshape(-1)

        dataset_size = flat_obs.shape[0]

        for epoch in range(config["NUM_EPOCHS"]):
            rng, shuffle_rng = jax.random.split(rng)

            # Shuffle indices (Standard PyTorch dataloader behavior)
            indices = jax.random.permutation(shuffle_rng, dataset_size)

            for start_idx in range(0, dataset_size, minibatch_size):
                batch_idx = indices[start_idx : start_idx + minibatch_size]

                # Sample minibatches
                mb_obs = flat_obs[batch_idx]
                mb_actions = flat_actions[batch_idx]
                mb_targets = flat_targets[batch_idx]

                # Gradient step
                train_state, loss = train_step(
                    train_state, mb_obs, mb_actions, mb_targets
                )

        # Basic Logging
        if update % 10 == 0:
            print(
                f"Update: {update}/{num_updates} | Loss: {loss:.4f} | Eps: {current_eps:.3f}"
            )


if __name__ == "__main__":
    main()

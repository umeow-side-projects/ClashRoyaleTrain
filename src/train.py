import os
import keras
import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.trajectories.trajectory import from_transition
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common
from time import time

from .game_envriment import GameEnvironment

def train_main():
    train_time = time()
    # === Step 1: Setup Environment === #
    # env_name = "CartPole-v1"
    train_env = tf_py_environment.TFPyEnvironment(GameEnvironment)

    # === Step 2: Define PPO Agent === #
    actor_net = ActorDistributionNetwork(
        input_tensor_spec=train_env.observation_spec(),
        output_tensor_spec=train_env.action_spec(),
        fc_layer_params=(16384, 8192, 4096, 4096)  # Hidden layers
    )

    value_net = ValueNetwork(
        input_tensor_spec=train_env.observation_spec(),
        fc_layer_params=(16384, 8192, 4096, 4096)
    )

    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = ppo_agent.PPOAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        actor_net=actor_net,
        value_net=value_net,
        optimizer=optimizer,
        normalize_observations=True,
        normalize_rewards=True,
        use_gae=True,
        num_epochs=10,
        train_step_counter=global_step
    )
    agent.initialize()

    # === Step 3: Define Replay Buffer and Policies === #
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=10000
    )

    collect_policy = agent.collect_policy
    eval_policy = agent.policy

    # === Step 4: Metrics and Logging === #
    train_metrics = [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric()
    ]

    # === Step 5: Driver for Data Collection === #
    collect_driver = DynamicStepDriver(
        train_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=200
    )

    # === Step 6: Training Loop === #
    num_iterations = 5000
    log_interval = 100
    checkpoint_dir = os.path.join('ppo_checkpoints', f'{train_time}')
    checkpoint = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=10)

    for iteration in range(num_iterations):
        # Collect Data
        time_step = train_env.reset()
        collect_driver.run()

        # Sample from the replay buffer
        experience = replay_buffer.gather_all()

        # Train the agent
        loss_info = agent.train(experience)

        # Clear the replay buffer for the next iteration
        replay_buffer.clear()

        # Logging and checkpointing
        if iteration % log_interval == 0:
            avg_return = train_metrics[0].result().numpy()
            avg_length = train_metrics[1].result().numpy()
            print(f"Iteration: {iteration}, Avg Return: {avg_return:.2f}, Avg Length: {avg_length:.2f}, Loss: {loss_info.loss.numpy():.2f}")
            try:
                checkpoint_manager.save(global_step)
            except:
                pass

    print("Training complete.")

    # # === Step 7: Evaluate the Policy === #
    # average_return = tf_metrics.AverageReturnMetric()
    # num_eval_episodes = 10

    # for _ in range(num_eval_episodes):
    #     time_step = valid_env.reset()
    #     episode_return = 0
    #     while not time_step.is_last():
    #         action_step = eval_policy.action(time_step)
    #         time_step = valid_env.step(action_step.action)
    #         episode_return += time_step.reward
    #     average_return(episode_return)

    # print(f"Average Return in Evaluation: {average_return.result().numpy():.2f}")
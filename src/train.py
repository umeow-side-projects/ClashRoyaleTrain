import os
import keras
import logging
import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks.actor_distribution_network import ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.environments import tf_py_environment
from tf_agents.environments import suite_gym
from tf_agents.policies import policy_saver
from tf_agents.trajectories.trajectory import from_transition
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common
from time import time, sleep

from .game_envriment import GameEnvironment
from .game_controller import GameController

def train_main():
    logging.info('train_main()')
    train_time = int(time())
    # === Step 1: Setup Environment === #
    # env_name = "CartPole-v1"
    train_env = tf_py_environment.TFPyEnvironment(GameEnvironment)

    # === Step 2: Define PPO Agent === #
    actor_net = ActorDistributionNetwork(
        input_tensor_spec=train_env.observation_spec(),
        output_tensor_spec=train_env.action_spec(),
        # fc_layer_params=(16384, 8192, 4096, 4096)  # Hidden layers
        fc_layer_params=(16384, 8192, 8192, 4096, 4096, 4096),
    )

    value_net = ValueNetwork(
        input_tensor_spec=train_env.observation_spec(),
        # fc_layer_params=(16384, 8192, 4096, 4096)
        fc_layer_params=(16384, 8192, 8192, 4096, 4096, 4096),
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
        max_length=100000
    )

    # === Step 4: Metrics and Logging === #
    train_metrics = [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric()
    ]

    # === Step 6: Training Loop === #
    num_iterations = 6000000
    log_interval = 3
    # checkpoint = tf.train.Checkpoint(agent=agent, optimizer=optimizer)
    
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    
    train_checkpointer = common.Checkpointer(
        ckpt_dir='ppo_checkpoints',
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=global_step
    )
    
    train_checkpointer.initialize_or_restore()
    collect_policy = agent.collect_policy
    eval_policy = agent.policy
    
    global_step = tf.compat.v1.train.get_global_step()
    
    # === Step 5: Driver for Data Collection === #
    collect_driver = DynamicEpisodeDriver(
        train_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=1  # 每次收集遊戲一場完整的 episode
    )
    
    # checkpoint_manager = tf.train.CheckpointManager(checkpoint, f'ppo_checkpoints', max_to_keep=5)

    for iteration in range(num_iterations):
        # Collect Data
        time_step = train_env.reset()
        
        logging.info('collect driver run')
        collect_driver.run()

        GameController.is_training = True
        logging.info('start training...')
        
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
            train_checkpointer.save(global_step)
            tf_policy_saver.save('policy_dir')
        
        GameController.is_training = False
        logging.info('end training...')

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
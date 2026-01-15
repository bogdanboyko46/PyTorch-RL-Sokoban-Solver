import tf_agents.environments.suite_gym as suite_gym
import tensorflow as tf
from tf_agents.networks import sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.optimizers import AdamOptimizer
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np

print("TensorFlow version:", tf.__version__)

# Define a simple Q-network
q_net = q_network.QNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params=(128, 128)
)

# initialize learning agent
optimizer = AdamOptimizer(learning_rate=1e-3)
agent = dqn_agent.DqnAgent(
    env.time_step_spec(),
    env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=tf.keras.losses.MeanSquaredError()
)
agent.initialize()

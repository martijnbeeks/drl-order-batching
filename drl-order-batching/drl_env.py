'''
Code for defining the simulation environment into a gym environment
'''
import gym
import pandas as pd
import numpy as np
import yaml

from simulation_model.WAREHOUSESimulation import WAREHOUSESimulation
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class WAREHOUSE(gym.Env):

    def __init__(self, params=(1, 1), heuristic=None):

        # Open file with parameters
        with open(r'config/scenario_2.yml') as file:
            config = yaml.full_load(file)

        self.config = config
        self.heuristic = heuristic
        self.params = params

        # Set Gym variables for action space and observation space
        self.action_space = gym.spaces.Discrete(self.config['environment']['action_space'])
        self.observation_space = gym.spaces.Box(
            low=0, high=1,
            shape=(self.config['environment']['observation_space'],), dtype=np.uint8)

        # Set simulation parameters and load order data
        self.n = self.config['environment']['throughput']
        self.t_start = self.config['environment']['t_start']
        self.order_data = pd.read_csv(r'data/dummy_order_data.csv')

        # Sample orders to simulate and initiate simulation instance
        data = self.sample_orders(self.order_data, self.n, self.config['environment']['time_window'])
        self.sim = WAREHOUSESimulation(self.config, data, self.t_start, params=self.params, heuristic=self.heuristic)

        # Set parameters to capture simulation performance
        self.steps = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_reward_copy = 0
        self.episode_reward_hist = -20000
        self.order_batch_ratio = 0
        self.infeasible_actions = []
        self.infeasible_ratio = 0
        self.tardy_orders = 0
        self.picking_time = 0
        self.action_to_action = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
                                 11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10}

    # Function for sampling orders based on n and time_window
    def sample_orders(self, data, n, time_window):
        data_filtered = data[((data['arrival_time'] >= time_window[0] * 3600) &
                              (data['arrival_time'] <= time_window[-1] * 3600))]
        sample_data = data_filtered.sample(n=n, replace=False)
        sample_data = sample_data.sort_values(by='cutoff_time', ascending=True)
        return sample_data

    # Reset function for agent, saves information and reinitiates simulation instance
    def reset(self):
        # Print and save information about the processed episode
        print('Episode {0} finished, steps per episode: {1}'.format(self.episode, self.episode_step))
        self.sim.episode_render()
        results = self.sim.episode_render_test()
        self.tardy_orders = results['tardy_orders']
        self.picking_time = results['picking_time']

        # Sample new set of orders for the next episode
        self.data = self.sample_orders(self.order_data, self.n, self.config['environment']['time_window'])

        # Initiate simulation environment and get initial state
        self.sim = WAREHOUSESimulation(self.config, self.data, self.t_start, self.params, self.heuristic)
        state = self.sim.get_state()
        self.episode += 1
        self.episode_step = 0
        self.episode_reward_hist = self.episode_reward_copy
        self.episode_reward_copy = 0

        return np.array(state)

    def step(self, action):
        # Check whether this is a feasible action
        state = self.sim.get_state()
        feasibility = self.sim.check_action(action, state)

        # Compute reward based on current state and action pair
        reward = self.sim.get_reward(action)

        # If the reward is feasible, simulate the action in the simulation model and observe new state
        # If the reward is not feasible, do nothing and provide negative reward
        if feasibility:
            state = self.sim.simulate(action)
            self.infeasible_actions.append(1)
        else:
            self.infeasible_actions.append(0)

        # Check whether an episode is done. An episode is done when all orders have been processed
        done = self.sim.check_termination()

        # If at some point, the agent takes too many step, the episode is terminated
        if self.episode_step > 400000:
            done = True
            print('Episode stopped, {0} steps taken'.format(self.episode_step))

        # If an episode is done, save information
        if done:
            self.sim.reward_distribution['final_reward'] += (1 - self.sim.state_representation[-3] / self.sim.nOrders)**2
            reward += ((1 - self.sim.state_representation[-3] / self.sim.nOrders) ** 2) * self.params[0]

        # Save information based on the step that was taken
        self.episode_reward_copy += reward
        self.steps += 1
        self.episode_step += 1
        self.order_batch_ratio = self.sim.order_batch_ratio_sim
        self.infeasible_ratio = self.infeasible_actions.count(0) / len(self.infeasible_actions)
        if len(self.infeasible_actions) > 1000:
            self.infeasible_actions = []

        return np.array(state), reward, done, {}

    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError()
        return np.array([0, 0, 0])


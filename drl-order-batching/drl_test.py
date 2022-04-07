'''
Code testing a trained DRL agent on the simulation environment
'''
from stable_baselines import PPO2
from drl_env import WAREHOUSE
import random
import numpy as np
import pandas as pd

# Set batching heuristic
heuristic = 'BOC'

# Initiate simulation environment and load trained agent
env = WAREHOUSE(params=(1,1), heuristic=heuristic)

path = "trained_models/scenario_2/V006_first_run.zip"
model = PPO2.load(path)

# Set lists for performance metrics
average_results = {'tardy_orders': 0, 'pick_by_batch': 0, 'finish_time': 0,
                   'episode_reward': 0, 'steps_per_episode': 0, 'infeasible_rate': 0,
                   'batch_size': 0, 'picking_time': 0, 'order_cutoff': 0, 'batch_size_pick_batch': 0}

average_results_tardy_orders = []
average_results_pick_batch = []
average_results_finish = []
average_results_batch_size = []
average_results_picking_time = []
valid_actions_list = []
order_list_first = []
action_list = []

# Reset environment to obtain state representation (obs)
obs = env.reset()

# Set number of episodes to test
total_episodes = 20

episode = 1
while episode < total_episodes:

    # Retrieve list with feasible actions
    valid_actions = env.sim.available_actions(obs)
    valid_actions = [idx for idx, x in enumerate(valid_actions) if x > 0]

    # Obtain policy prediction on current state representation
    action, _states = model.predict(obs)

    # Check whether this predicted action is feasible
    if action in valid_actions:
        valid_actions_list.append(1)
    else:
        action = random.sample(valid_actions, 1)[0]
        valid_actions_list.append(0)

    # With this action, make step within environment
    obs, rewards, done, info = env.step(action)
    action_list.append([action, env.sim.state_representation[-1]])

    # If an episode is completed, save results and display
    if done:
        average_results['steps_per_episode'] += env.episode_step
        average_results['episode_reward'] += env.sim.reward_episode

        for order in env.sim.finished_orders:
            if order.cutoff_time == 86400 and order.System_out < 82800:
                order_list_first.append(order)

        average_results['order_cutoff'] += len(order_list_first)/env.config['environment']['throughput']

        results = env.sim.episode_render_test()
        obs = env.reset()

        average_results_tardy_orders.append(results['tardy_orders'])
        average_results_pick_batch.append(results['pick_by_batch'])
        average_results_finish.append(results['finish_time'])
        average_results_batch_size.append(results['batch_size'])
        average_results_picking_time.append(results['picking_time'])

        episode += 1
        print('Episode number: ', episode)

print('Model: ', path)
print('Average results over {0} episodes'.format((env.episode - 1)))
print('Tardy orders (avg): ', round(np.mean(average_results_tardy_orders),4), round(np.std(average_results_tardy_orders),4))
print('Pick-by-batch (avg): ', round(np.mean(average_results_pick_batch),4), round(np.std(average_results_pick_batch),4))
print('Finish time (avg): ', round(np.mean(average_results_finish),4), round(np.std(average_results_finish),4))
print('Batch size (avg): ', round(np.mean(average_results_batch_size), 4), round(np.std(average_results_batch_size),4))
print('Picking time (avg): ', round(np.mean(average_results_picking_time), 4), round(np.std(average_results_picking_time),4))
print('Valid action rate: ', sum(valid_actions_list)/len(valid_actions_list))


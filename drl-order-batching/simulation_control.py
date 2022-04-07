import yaml
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from simulation_model.WAREHOUSESimulation import WAREHOUSESimulation

with open(r'config/scenario_day.yml') as file:
    config = yaml.full_load(file)

order_data = pd.read_csv('data/dummy_order_data.csv')


def sample_orders(data, n, time_window=[0, 23.5]):
    data_filtered = data[((data['arrival_time'] >= (time_window[0])*3600) &
                          (data['arrival_time'] <= time_window[-1]*3600))]
    sample_data = data_filtered.sample(n=n, replace=False)
    sample_data = sample_data.sort_values(by='cutoff_time', ascending=True)
    return sample_data


t_start = config['environment']['t_start']
n = config['environment']['throughput']
time_window = config['environment']['time_window']

# heuristic = 'LST'
# heuristic = 'GRASP_VND'
heuristic = 'BOC'
# heuristic = 'GVNS'
# heuristic = None

weights = {'scenario_1': [0.860933108285528, 0.9026343218274464],
           'scenario_2': [1.3356476376785569, 0.5199800258281753],
           'scenario_3': [0.8210920808200558, 0.5199952974746439]}

data = sample_orders(order_data, n=n, time_window=time_window)
sim = WAREHOUSESimulation(config, data, t_start, weights, heuristic)
average_results_tardy_orders = []
average_results_pick_batch = []
average_results_finish = []
average_results_batch_size = []
average_results_picking_time = []


episodes = 20
for i in tqdm(range(episodes)):
    data = sample_orders(order_data, n=n, time_window=time_window)
    sim = WAREHOUSESimulation(config, data, t_start, weights, heuristic)
    state_rep = sim.get_state()
    action_list = []

    t_1 = time.time()
    steps_episode = 0
    while not sim.check_termination():
        action = sim.edd_sequencing(state_rep)
        reward = sim.get_reward(action)
        state_rep = sim.simulate(action)
        steps_episode += 1
        action_list.append([action, sim.state_representation[-1]])

    results = sim.episode_render_test()
    average_results_tardy_orders.append(results['tardy_orders'])
    average_results_pick_batch.append(results['pick_by_batch'])
    average_results_finish.append(results['finish_time'])
    average_results_batch_size.append(results['batch_size'])
    average_results_picking_time.append(results['picking_time'])
    sim.res.save_action(action_list)

print('/n')
print('Heuristic: ', heuristic)
print('Average results over {0} episodes'.format(episodes))
print('Tardy orders (avg): ', round(np.mean(average_results_tardy_orders), 4),
      round(np.std(average_results_tardy_orders), 4))
print('Pick-by-batch (avg): ', round(np.mean(average_results_pick_batch), 4),
      round(np.std(average_results_pick_batch), 4))
print('Finish time (avg): ', round(np.mean(average_results_finish), 4),
      round(np.std(average_results_finish), 4))
print('Batch size (avg): ', round(np.mean(average_results_batch_size), 4),
      round(np.std(average_results_batch_size), 4))
print('Picking time (avg): ', round(np.mean(average_results_picking_time), 4),
      round(np.std(average_results_picking_time), 4))

data = {'tardy_orders': average_results_tardy_orders, 'picking_costs': average_results_picking_time}
df = pd.DataFrame(data=data)

# sim.res.plot_cutoff_moments(sim.finished_orders)
# sim.res.plot_avg_picking_times(sim.finished_orders)
# sim.res.print_resource_utilization()
# sim.res.plot_order_progress(sim.finished_orders, 'Order progress experiment D - BOC heuristic')
# sim.res.plot_order_progress(sim.finished_orders, 'LST heuristic')
# sim.res.plot_tardy_orders()

# sim.res.plot_QL(sim.finished_orders)
# sim.res.plot_resource_utilization()

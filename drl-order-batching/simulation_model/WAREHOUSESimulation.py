'''
Code for simulation model that simulates a Markov Decision Process in a warehousing environment
'''
from .Order import Order
from .Event import Event
from .FES import FES
from .Distribution import Distribution
from .SimResults import SimResults
from scipy import stats
import random
import numpy as np

# This simulation instance contains several functions.
# - simulate --> Inputs an action, processes this action and outputs a new state representation
# - get_state --> Retrieves state representation from simulation instance
# - rebuild_state_representation --> based on available data of simulate function, build new state representation
# - clip_state --> clips state representation in order for it to be normalized
# - action_to_orders --> used by simulate function to transform predicted action into list with orders to be processed
# - resource_rebalance --> used by simulate function to balance resources between workstations
# - change_shift --> used by simulate function to set resources per shift
# - custom_heuristic --> batching heuristic that always performs batch action
# - edd_sequencing --> batching heuristic that performs edd batching
# - LST_batching --> batching heuristic that performs LST batching
# - grasp_vnd --> batching heuristic that performs local search method
# - boc_batching --> batching heuristic that performs BOC batching
# - GVNS_batching --> batching heuristic that performs local search method


class WAREHOUSESimulation:

    def __init__(self, config, data, t, params, heuristic=None):
        
        self.fes = FES()
        self.res = SimResults(config)
        self.config = config['simulation']
        self.config_environment = config['environment']
        
        self.heuristic = heuristic

        self.weight_tardy = params[0]
        self.weight_picking = params[1]

        self.PtG_picker_available = config['simulation']['nPtG_pickers'][0]
        self.GtP_shuttle_available = config['simulation']['nGtP_shuttles'][0]
        self.DtO_operator_available = config['simulation']['nDtO_operators'][0]
        self.StO_operator_available = config['simulation']['nStO_operators'][0]
        
        self.rebalance_interval = 300

        self.qPtG = []
        self.qGtP = []
        self.qPack = []
        self.qDtO = []
        self.qStO = []
        self.finished_orders = []
        
        # Processing times
        self.PtG_picking_item = Distribution(stats.norm(loc=config['simulation']['PtG_picking_time'], scale=10))
        self.PtG_picking_constant = config['simulation']['PtG_picking_constant']
        self.GtP_picking_time = config['simulation']['GtP_picking_time']
        self.Pack_time = config['simulation']['Pack_time']  # [SIO, MIO]
        self.DtO_time = config['simulation']['DtO_time']
        self.StO_time = config['simulation']['StO_time']
        self.time_step_arrival = config['simulation']['time_step_arrival']
        
        # Travel times
        self.PtG_Out_time = config['simulation']['PtG_Out_time']
        self.PtG_Pack_time = config['simulation']['PtG_Pack_time']
        self.Pack_Out_time = config['simulation']['Pack_Out_time']
        self.GtP_DtO_time = config['simulation']['GtP_DtO_time']
        self.DtO_Out_time = config['simulation']['DtO_Out_time']
        self.GtP_StO_time = config['simulation']['GtP_StO_time']
        self.StO_Pack_time = config['simulation']['StO_Pack_time']
        self.StO_Out_time = config['simulation']['StO_Out_time']
        self.PtG_GtP_time = config['simulation']['PtG_GtP_time']

        self.virtual_q_ptg = config['simulation']['virtual_q_ptg']
        self.virtual_q_gtp = config['simulation']['virtual_q_gtp']
        self.virtual_q_dto = config['simulation']['virtual_q_dto']
        self.virtual_q_sto = config['simulation']['virtual_q_sto']

        self.action_route_mapping = {0: 1, 1: 2, 2: 5, 3: 5, 4:  1, 5: 3, 6: 5, 7: 4, 8: 3, 9: 6}

        self.route_action_mapping = {1: [0, 4], 2: [1], 3: [5, 8], 4: [7], 5: [2, 3, 6], 6: [9]}

        self.action_category_mapping = {0: [0, 1, 2], 1: [0, 1, 2], 2: [3, 4, 5], 3: [3, 4, 5], 4: [6, 7, 8],
                                        5: [6, 7, 8], 6: [9, 10, 11], 7: [9, 10, 11], 8: [12, 13, 14], 9: [12, 13, 14]}

        self.category_action_mapping = {0: [0, 1], 1: [0, 1], 2: [0, 1], 3: [2, 3], 4: [2, 3], 5: [2, 3], 6: [4, 4],
                                        7: [4, 5], 8: [4, 5], 9: [6, 7], 10: [6, 7], 11: [6, 7], 12: [8, 9],
                                        13: [8, 9], 14: [8, 9]}

        self.valid_actions = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}

        self.actions_tardy_orders = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.actions_picking_time = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.action_to_action = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
                                 11: 0, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10}

        self.actions_pick_by_batch = [1, 3, 5, 7, 8]
        self.max_batchsize_ptg = config['simulation']['max_batchsize_ptg']
        self.max_batchsize_ptg_items = config['simulation']['max_batchsize_ptg_items']
        self.max_batchsize_gtp = config['simulation']['max_batchsize_gtp']
        self.max_batchsize_ptg_gtp = config['simulation']['max_batchsize_ptg_gtp']
        
        self.order_data = data
        self.state_representation, self.order_categories = self.build_state_representation(self.order_data, t)
        self.refresh_state_representation = 30
        self.old_state = self.state_representation[:]
        self.initial_state = self.state_representation[:]

        self.reward_structure = {'infeasible_action': -0.5, 'tardy_order': -1.5, 'feasible_action': 0,
                                 'batch_action': 0.1}
        self.reward_action = 0
        self.reward_episode = 0
        self.nOrders = len(data)

        self.order_batch_ratio_sim = 0
        self.action_list_render = []
        self.picking_strategy = []

        self.nActions = config['environment']['action_space']

        self.infeasible_action_rate = 0
        
        self.DtO_resource_transfer = 0
        self.nOrders_hist = 1
        self.nItems_ptg_hist = 0
        self.nItems_gtp_hist = 0

        self.avg_batch_size = []
        self.avg_size_pick_batch = []
        self.picking_time = []
        self.picking_time_total = []
        self.reward_distribution = {'infeasible_action': 0, 'tardy_order': 0, 'batch_action': 0, 'batch_composition': 0,
                                    'final_reward': 0}

        self.tardy_order_hist = 0

        self.tardy_order_list_test = []
        
        self.shift_change = False
        
        self.batch_list = []

    def simulate(self, action):
        self.fes.events.sort()
        current_t = self.state_representation[-1]

        # Register performance of system
        self.res.register_QL(len(self.qGtP), len(self.qPtG), len(self.qPack), len(self.qDtO), len(self.qStO))
        self.res.register_resources(self.PtG_picker_available, max(self.GtP_shuttle_available, 0),
                                    max(self.DtO_operator_available, 0), max(self.StO_operator_available, 0), current_t)
        self.res.register_time(current_t)
        
        picking_items = None
        if len(self.action_list_render) > 100:
            self.order_batch_ratio_sim = len([x for x in self.action_list_render[-100 - 1:-1] if x not in self.actions_pick_by_batch]) / len(
                        self.action_list_render[-100 - 1:-1])
        
        # There are two types of actions:
        # 1. Processing some kind of order --> action 0 - 9
        # 2. Do nothing and wait for state change --> action 10
        
        # 1. Processing some kind of order
        if action < 10: 
            picking_order, picking_items, order_category, all_picking_items = self.action_to_orders(action)
            
            if self.heuristic == 'LST':
                picking_items = self.LST_batching(action, picking_items, current_t)
                
            elif self.heuristic == 'GRASP_VND':
                picking_items = self.grasp_vnd(action, picking_items, all_picking_items, current_t)
                
            elif self.heuristic == 'BOC':
                picking_items = self.boc_batching(action, picking_items, all_picking_items)
                
            elif self.heuristic == 'GVNS':
                picking_items = self.GVNS_batching(action, picking_items, all_picking_items)
                
            picking_order, picking_items = self.remove_orders(action, picking_items, order_category)
                
            cutoff_time, nOrders, nItems_ptg, nItems_gtp, route = picking_order
            order = Order(current_t, cutoff_time, nOrders, nItems_ptg, nItems_gtp, route,
                          order_category, action)

            self.nOrders_hist = nOrders
            self.nItems_ptg_hist = nItems_ptg
            self.nItems_gtp_hist = nItems_gtp
            self.action_list_render.append(action)
            if action in self.actions_pick_by_batch:
                self.picking_strategy.append(1)
            else:
                self.picking_strategy.append(0)

            # Route 1, 2, 3
            if order.route in [1, 2, 3, 6]:
                self.qPtG.append(order)
                arr = Event(Event.ARRIVAL, 'PtG', current_t, order)
                self.fes.add(arr)

                # If a picker is available, assign picker and adjust picker availability
                if self.PtG_picker_available > 0:
                    dummy_event = Event(Event.ARRIVAL, 'DUMMY', 0, None)
                    arr_event = next((x for x in self.fes.events if x.type == Event.ARRIVAL and x.station == 'PtG'),
                                     dummy_event)  # Arrival at PtG station
                    self.fes.events.remove(arr_event)  # arrival is handled, can be removed

                    dep = Event(Event.DEPARTURE, 'PtG',
                                current_t + (self.PtG_picking_item.rvs() * arr_event.order.nItems_ptg) +
                                self.PtG_picking_constant, arr_event.order)

                    self.fes.add(dep)  # schedule his departure
                    self.PtG_picker_available -= 1
                    arr_event.order.PtG_in = current_t

                # new simulation time
                new_t = current_t + self.time_step_arrival

            # Route 4, 5
            elif order.route in [4, 5]:
                self.qGtP.append(order)
                arr = Event(Event.ARRIVAL, 'GtP', current_t, order)
                self.fes.add(arr)

                # If a shuttle is available, assign shuttle and adjust shuttle availability
                if self.GtP_shuttle_available > 0:
                    dummy_event = Event(Event.ARRIVAL, 'DUMMY', 0, None)
                    arr_event = next((x for x in self.fes.events if x.type == Event.ARRIVAL and x.station == 'GtP'),
                                     dummy_event)  # Arrival at GtP station
                    self.fes.events.remove(arr_event)  # arrival is handled, can be removed

                    dep = Event(Event.DEPARTURE, 'GtP', current_t + self.GtP_picking_time * arr_event.order.nItems_gtp,
                                arr_event.order)
                    self.fes.add(dep)  # schedule his departure
                    self.GtP_shuttle_available -= 1
                    arr_event.order.GtP_in = current_t

                # new simulation time
                new_t = current_t + self.time_step_arrival

        # 2. Do nothing and wait for state change
        elif action == 10:
            order_category = False
            # Sometime, there are no events in the future event set anymore.
            # Resources need to wait until new orders arrive. 
            
            if len(self.fes.events) == 0:
                new_t = current_t + self.time_step_arrival
            
            else:    
                # Select first event
                event = self.fes.events[0]
                if event.type == Event.DEPARTURE:
                    # handle departure for route 1
                    if event.station == 'PtG' and event.order.route == 1:
                        t = event.time
                        self.qPtG.remove(event.order)
                        self.fes.events.remove(event)
                        event.order.System_out = t + self.PtG_Out_time
                        self.finished_orders.append(event.order)
                        self.res.report_tardiness(event.order, t + self.PtG_Out_time)
                        self.PtG_picker_available += 1
                        event.order.PtG_out = t
    
                    # handle departure for route 2
                    elif event.order.route == 2:
                        if event.station == 'PtG':
                            t = event.time
                            self.qPtG.remove(event.order)
                            self.fes.events.remove(event)
                            event.order.System_out = t + self.PtG_Out_time
                            self.finished_orders.append(event.order)
                            self.res.report_tardiness(event.order, t + self.PtG_Out_time)
                            self.PtG_picker_available += 1
                            event.order.PtG_out = t
    
                    # handle departure for route 3
                    elif event.order.route == 3:
                        if event.station == 'PtG':
                            t = event.time
                            self.qPtG.remove(event.order)
                            self.fes.events.remove(event)
                            self.qGtP.append(event.order)
                            self.PtG_picker_available += 1
                            event.order.PtG_out = t
    
                            arr = Event(Event.ARRIVAL, 'GtP', t + self.PtG_GtP_time, event.order)
                            self.fes.add(arr)
    
                        elif event.station == 'GtP':
                            t = event.time
                            self.qGtP.remove(event.order)
                            self.fes.events.remove(event)
                            self.qStO.append(event.order)
                            self.GtP_shuttle_available += 1
                            event.order.GtP_out = t
    
                            arr = Event(Event.ARRIVAL, 'StO', t + self.GtP_StO_time, event.order)
                            self.fes.add(arr)
    
                        elif event.station == 'StO':
                            t = event.time
                            self.qStO.remove(event.order)
                            self.fes.events.remove(event)
                            event.order.System_out = t + self.StO_Out_time
                            self.finished_orders.append(event.order)
                            self.res.report_tardiness(event.order, t + self.StO_Out_time)
                            self.StO_operator_available += 1
                            event.order.StO_out = t
    
                    # handle departure for route 4
                    elif event.order.route == 4:
                        if event.station == 'GtP':
                            t = event.time
                            self.qGtP.remove(event.order)
                            self.fes.events.remove(event)
                            self.qStO.append(event.order)
                            self.GtP_shuttle_available += 1
                            event.order.GtP_out = t
    
                            arr = Event(Event.ARRIVAL, 'StO', t + self.GtP_StO_time, event.order)
                            self.fes.add(arr)
    
                        elif event.station == 'StO':
                            t = event.time
                            self.qStO.remove(event.order)
                            self.fes.events.remove(event)
                            event.order.System_out = t + self.StO_Out_time
                            self.finished_orders.append(event.order)
                            self.res.report_tardiness(event.order, t + self.StO_Out_time)
                            self.StO_operator_available += 1
                            event.order.StO_out = t
    
                    # handle event for route 5
                    elif event.order.route == 5:
                        if event.station == 'GtP':
                            t = event.time
                            self.qGtP.remove(event.order)
                            self.fes.events.remove(event)
                            self.qDtO.append(event.order)
                            self.GtP_shuttle_available += 1
                            event.order.GtP_out = t
    
                            arr = Event(Event.ARRIVAL, 'DtO', t + self.GtP_DtO_time, event.order)
                            self.fes.add(arr)
    
                        elif event.station == 'DtO':
                            t = event.time
                            self.qDtO.remove(event.order)
                            self.fes.events.remove(event)
                            event.order.System_out = t + self.DtO_Out_time
                            self.finished_orders.append(event.order)
                            self.res.report_tardiness(event.order, t + self.DtO_Out_time)
                            self.DtO_operator_available += 1
                            event.order.DtO_out = t
    
                    # handle departure for route 6
                    elif event.order.route == 6:
                        if event.station == 'PtG':
                            t = event.time
                            self.qPtG.remove(event.order)
                            self.fes.events.remove(event)
                            self.qGtP.append(event.order)
                            self.PtG_picker_available += 1
                            event.order.PtG_out = t
    
                            arr = Event(Event.ARRIVAL, 'GtP', t + self.PtG_GtP_time, event.order)
                            self.fes.add(arr)
    
                        elif event.station == 'GtP':
                            t = event.time
                            self.qGtP.remove(event.order)
                            self.fes.events.remove(event)
                            self.qDtO.append(event.order)
                            self.GtP_shuttle_available += 1
                            event.order.GtP_out = t
    
                            arr = Event(Event.ARRIVAL, 'DtO', t + self.GtP_DtO_time, event.order)
                            self.fes.add(arr)
    
                        elif event.station == 'DtO':
                            t = event.time
                            self.qDtO.remove(event.order)
                            self.fes.events.remove(event)
                            event.order.System_out = t + self.DtO_Out_time
                            self.finished_orders.append(event.order)
                            self.res.report_tardiness(event.order, t + self.DtO_Out_time)
                            self.DtO_operator_available += 1
                            event.order.DtO_out = t
    
                else:
                    state_change = False
                    dummy_event = Event(Event.ARRIVAL, 'DUMMY', 0, None)
                    # Arrival at PtG station
                    arr_event = next((x for x in self.fes.events if x.type == Event.ARRIVAL and x.station == 'PtG'),
                                     dummy_event)
                    if arr_event.station == 'PtG' and self.PtG_picker_available > 0 and state_change == False:
                        t = current_t + self.time_step_arrival
                        self.fes.events.remove(arr_event)  # arrival is handled, can be removed
                        dep = Event(Event.DEPARTURE, 'PtG', t + self.PtG_picking_item.rvs() * arr_event.order.nItems_ptg +
                                    self.PtG_picking_constant, arr_event.order)
                        self.fes.add(dep)  # schedule his departure
                        self.PtG_picker_available -= 1
                        state_change = True
                        arr_event.order.PtG_in = t
    
                    # Arrival at GtP station
                    arr_event = next((x for x in self.fes.events if x.type == Event.ARRIVAL and x.station == 'GtP'),
                                     dummy_event)
                    if arr_event.station == 'GtP' and self.GtP_shuttle_available > 0 and state_change == False:
                        t = current_t + self.time_step_arrival
                        self.fes.events.remove(arr_event)  # arrival is handled, can be removed
                        dep = Event(Event.DEPARTURE, 'GtP', t + self.GtP_picking_time * arr_event.order.nItems_gtp,
                                    arr_event.order)
                        self.fes.add(dep)  # schedule his departure
                        self.GtP_shuttle_available -= 1
                        state_change = True
                        arr_event.order.GtP_in = t
    
                    # Arrival at DtO station
                    arr_event = next((x for x in self.fes.events if x.type == Event.ARRIVAL and x.station == 'DtO'),
                                     dummy_event)
                    if arr_event.station == 'DtO' and self.DtO_operator_available > 0 and state_change == False:
                        t = current_t + self.time_step_arrival
                        self.fes.events.remove(arr_event)  # arrival is handled, can be removed
                        dep = Event(Event.DEPARTURE, 'DtO', t + self.DtO_time, arr_event.order)
                        self.fes.add(dep)  # schedule his departure
                        self.DtO_operator_available -= 1
                        state_change = True
                        arr_event.order.DtO_in = t
    
                    # Arrival at StO station
                    arr_event = next((x for x in self.fes.events if x.type == Event.ARRIVAL and x.station == 'StO'),
                                     dummy_event)
                    if arr_event.station == 'StO' and self.StO_operator_available > 0 and state_change == False:
                        t = current_t + self.time_step_arrival
                        self.fes.events.remove(arr_event)  # arrival is handled, can be removed
                        dep = Event(Event.DEPARTURE, 'StO', t + self.StO_time, arr_event.order)
                        self.fes.add(dep)  # schedule his departure
                        self.StO_operator_available -= 1
                        state_change = True
                        arr_event.order.StO_in = t
    
                    # If there is no picking capacity available, select next departure.
                    # When next departure of some order occurs, picking capacity becomes
                    # available and new orders can be processed.
                    if not state_change:
    
                        dep_event = next(x for x in self.fes.events if x.type == Event.DEPARTURE)
                        t = dep_event.time
    
                        # Select earliest departure event at PtG for route 1
                        if dep_event.station == 'PtG' and dep_event.order.route == 1:
                            self.qPtG.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            dep_event.order.System_out = t + self.PtG_Out_time
                            self.finished_orders.append(dep_event.order)
                            self.res.report_tardiness(dep_event.order, t + self.PtG_Out_time)
                            self.PtG_picker_available += 1
                            dep_event.order.PtG_out = t
    
                        # Select earliest departure event at PtG for route 2
                        elif dep_event.station == 'PtG' and dep_event.order.route == 2:
                            self.qPtG.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            dep_event.order.System_out = t + self.PtG_Out_time
                            self.finished_orders.append(dep_event.order)
                            self.res.report_tardiness(dep_event.order, t + self.PtG_Out_time)
                            self.PtG_picker_available += 1
                            dep_event.order.PtG_out = t
    
                        # Select earliest departure event at GtP for route 3
                        elif dep_event.station == 'PtG' and dep_event.order.route == 3:
                            self.qPtG.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            self.qGtP.append(dep_event.order)
                            self.PtG_picker_available += 1
                            dep_event.order.PtG_out = t
    
                            arr = Event(Event.ARRIVAL, 'GtP', t + self.PtG_GtP_time, dep_event.order)
                            self.fes.add(arr)
    
                        elif dep_event.station == 'GtP' and dep_event.order.route == 3:
                            self.qGtP.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            self.qStO.append(dep_event.order)
                            self.GtP_shuttle_available += 1
                            dep_event.order.GtP_out = t
    
                            arr = Event(Event.ARRIVAL, 'StO', t + self.GtP_StO_time, dep_event.order)
                            self.fes.add(arr)
    
                        elif dep_event.station == 'StO' and dep_event.order.route == 3:
                            self.qStO.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            dep_event.order.System_out = t + self.StO_Out_time
                            self.finished_orders.append(dep_event.order)
                            self.res.report_tardiness(dep_event.order, t + self.StO_Out_time)
                            self.StO_operator_available += 1
                            dep_event.order.StO_out = t
    
                        # Select earliest departure event at GtP for route 4
                        elif dep_event.station == 'GtP' and dep_event.order.route == 4:
                            self.qGtP.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            self.qStO.append(dep_event.order)
                            self.GtP_shuttle_available += 1
                            dep_event.order.GtP_out = t
    
                            arr = Event(Event.ARRIVAL, 'StO', t + self.GtP_StO_time, dep_event.order)
                            self.fes.add(arr)
    
                        elif dep_event.station == 'StO' and dep_event.order.route == 4:
                            self.qStO.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            dep_event.order.System_out = t + self.StO_Out_time
                            self.finished_orders.append(dep_event.order)
                            self.res.report_tardiness(dep_event.order, t + self.StO_Out_time)
                            self.StO_operator_available += 1
                            dep_event.order.StO_out = t
    
                        # Select earliest departure event at GtP for route 5
                        elif dep_event.station == 'GtP' and dep_event.order.route == 5:
                            self.qGtP.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            self.qDtO.append(dep_event.order)
                            self.GtP_shuttle_available += 1
                            dep_event.order.GtP_out = t
    
                            arr = Event(Event.ARRIVAL, 'DtO', t + self.GtP_DtO_time, dep_event.order)
                            self.fes.add(arr)
    
                        elif dep_event.station == 'DtO' and dep_event.order.route == 5:
                            self.qDtO.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            dep_event.order.System_out = t + self.DtO_Out_time
                            self.finished_orders.append(dep_event.order)
                            self.res.report_tardiness(dep_event.order, t + self.DtO_Out_time)
                            self.DtO_operator_available += 1
                            dep_event.order.DtO_out = t
    
                        # Select earliest departure event at GtP for route 6
                        elif dep_event.station == 'PtG' and dep_event.order.route == 6:
                            self.qPtG.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            self.qGtP.append(dep_event.order)
                            self.PtG_picker_available += 1
                            dep_event.order.PtG_out = t
    
                            arr = Event(Event.ARRIVAL, 'GtP', t + self.PtG_GtP_time, dep_event.order)
                            self.fes.add(arr)
    
                        elif dep_event.station == 'GtP' and dep_event.order.route == 6:
                            self.qGtP.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            self.qDtO.append(dep_event.order)
                            self.GtP_shuttle_available += 1
                            dep_event.order.GtP_out = t
    
                            arr = Event(Event.ARRIVAL, 'DtO', t + self.GtP_DtO_time, dep_event.order)
                            self.fes.add(arr)
    
                        elif dep_event.station == 'DtO' and dep_event.order.route == 6:
                            self.qDtO.remove(dep_event.order)
                            self.fes.events.remove(dep_event)
                            # self.qPack.append(dep_event.order)
                            dep_event.order.System_out = t + self.DtO_Out_time
                            self.finished_orders.append(dep_event.order)
                            self.res.report_tardiness(dep_event.order, t + self.DtO_Out_time)
                            self.DtO_operator_available += 1
                            dep_event.order.DtO_out = t
    
                # Simulation time
                new_t = t
        
        # Check if shift 1 is done and shift 2 starts
        # self.change_shift(new_t)
        
        # Update state representation
        self.old_state = self.state_representation[:]
        norm_state_rep = self.rebuild_state_representation(action, new_t, picking_items, order_category)
        
        return norm_state_rep

    def get_state(self):
        state = self.clip_state(self.state_representation[:])
        return state

    def rebuild_state_representation(self, action, new_t, picking_items, order_category):
        
        if self.refresh_state_representation == 0:
            self.state_representation, self.order_categories = self.build_state_representation(self.order_data, new_t)
            self.refresh_state_representation = 20
            
        else:
            PtG_available = 0 if (self.virtual_q_ptg - len(self.qPtG)) < 1 else 1
            GtP_available = 0 if (self.virtual_q_gtp - len(self.qGtP)) < 1 else 1
            
            self.state_representation[15] = PtG_available
            self.state_representation[16] = GtP_available
            self.state_representation[17] = self.res.finished_orders(self.finished_orders)
            self.state_representation[18] = self.res.tardy_orders
            self.state_representation[19] = new_t
            
            if action < 10:
                self.state_representation[order_category] -= len(picking_items)
            self.refresh_state_representation -= 1

        if self.state_representation[16] < 0:
            self.state_representation[16] = 0
        
        norm_state_rep = self.clip_state(self.state_representation[:])
        return norm_state_rep

    def build_state_representation(self, sample_data, current_time):
        # adjust sample data for orders that not have been arrived yet
        sample_data = sample_data[sample_data['arrival_time'] <= current_time * 1.05] 
        
        sample_data_sio = sample_data[sample_data['comp'] == 'SIO']  # 1. Order composition: SIO
        sample_data_sio_ptg = sample_data_sio[((sample_data_sio['nItems_ptg'] > 0) &
                                               (sample_data_sio['nItems_gtp'] == 0))]
        sample_data_sio_ptg_e1 = sample_data_sio_ptg[((sample_data_sio_ptg['cutoff_time'] - current_time) / 60) <= 15]
        sample_data_sio_ptg_e2 = sample_data_sio_ptg[(((sample_data_sio_ptg['cutoff_time'] - current_time) / 60) > 15) & (((sample_data_sio_ptg['cutoff_time'] - current_time) / 60) < 40)]
        sample_data_sio_ptg_e3 = sample_data_sio_ptg[((sample_data_sio_ptg['cutoff_time'] - current_time) / 60) >= 40]
        
        sample_data_sio_gtp = sample_data_sio[(sample_data_sio['nItems_ptg'] == 0) & (sample_data_sio['nItems_gtp'] > 0)]
        sample_data_sio_gtp_e1 = sample_data_sio_gtp[((sample_data_sio_gtp['cutoff_time'] - current_time) / 60) <= 15]
        sample_data_sio_gtp_e2 = sample_data_sio_gtp[(((sample_data_sio_gtp['cutoff_time'] - current_time) / 60) > 15) & (((sample_data_sio_gtp['cutoff_time'] - current_time) / 60) < 40)]
        sample_data_sio_gtp_e3 = sample_data_sio_gtp[((sample_data_sio_gtp['cutoff_time'] - current_time) / 60) >= 40]

        sample_data_mio = sample_data[sample_data['comp'] == 'MIO']  # 2. Order composition: MIO
        sample_data_mio_ptg = sample_data_mio[(sample_data_mio['nItems_ptg'] > 0) & (sample_data_mio['nItems_gtp'] == 0)]
        sample_data_mio_ptg_e1 = sample_data_mio_ptg[((sample_data_mio_ptg['cutoff_time'] - current_time) / 60) <= 15]
        sample_data_mio_ptg_e2 = sample_data_mio_ptg[(((sample_data_mio_ptg['cutoff_time'] - current_time) / 60) > 15) & (((sample_data_mio_ptg['cutoff_time'] - current_time) / 60) < 40)]
        sample_data_mio_ptg_e3 = sample_data_mio_ptg[((sample_data_mio_ptg['cutoff_time'] - current_time) / 60) >= 40]
        
        sample_data_mio_gtp = sample_data_mio[(sample_data_mio['nItems_ptg'] == 0) & (sample_data_mio['nItems_gtp'] > 0)]
        sample_data_mio_gtp_e1 = sample_data_mio_gtp[((sample_data_mio_gtp['cutoff_time'] - current_time) / 60) <= 15]
        sample_data_mio_gtp_e2 = sample_data_mio_gtp[(((sample_data_mio_gtp['cutoff_time'] - current_time) / 60) > 15) & (((sample_data_mio_gtp['cutoff_time'] - current_time) / 60) < 40)]
        sample_data_mio_gtp_e3 = sample_data_mio_gtp[((sample_data_mio_gtp['cutoff_time'] - current_time) / 60) >= 40]
        
        sample_data_mio_ptg_gtp = sample_data_mio[(sample_data_mio['nItems_ptg'] > 0) & (sample_data_mio['nItems_gtp'] > 0)]
        sample_data_mio_ptg_gtp_e1 = sample_data_mio_ptg_gtp[((sample_data_mio_ptg_gtp['cutoff_time'] - current_time) / 60) <= 15]
        sample_data_mio_ptg_gtp_e2 = sample_data_mio_ptg_gtp[(((sample_data_mio_ptg_gtp['cutoff_time'] - current_time) / 60) > 15) & (((sample_data_mio_ptg_gtp['cutoff_time'] - current_time) / 60) < 40)]
        sample_data_mio_ptg_gtp_e3 = sample_data_mio_ptg_gtp[((sample_data_mio_ptg_gtp['cutoff_time'] - current_time) / 60) >= 40]
        
        order_categories = [sample_data_sio_ptg_e1, sample_data_sio_ptg_e2, sample_data_sio_ptg_e3, sample_data_sio_gtp_e1, sample_data_sio_gtp_e2, sample_data_sio_gtp_e3,
                            sample_data_mio_ptg_e1, sample_data_mio_ptg_e2, sample_data_mio_ptg_e3, sample_data_mio_gtp_e1, sample_data_mio_gtp_e2, sample_data_mio_gtp_e3,
                            sample_data_mio_ptg_gtp_e1, sample_data_mio_ptg_gtp_e2, sample_data_mio_ptg_gtp_e3]

        PtG_available = 0 if (self.virtual_q_ptg - len(self.qPtG)) < 1 else 1
        GtP_available = 0 if (self.virtual_q_gtp - len(self.qGtP)) < 1 else 1
        
        state_representation = [len(sample_data_sio_ptg_e1), len(sample_data_sio_ptg_e2), len(sample_data_sio_ptg_e3), 
                                len(sample_data_sio_gtp_e1), len(sample_data_sio_gtp_e2), len(sample_data_sio_gtp_e3),
                                len(sample_data_mio_ptg_e1), len(sample_data_mio_ptg_e2), len(sample_data_mio_ptg_e3),
                                len(sample_data_mio_gtp_e1), len(sample_data_mio_gtp_e2), len(sample_data_mio_gtp_e3), 
                                len(sample_data_mio_ptg_gtp_e1), len(sample_data_mio_ptg_gtp_e2), len(sample_data_mio_ptg_gtp_e3),
                                PtG_available,
                                GtP_available,
                                self.res.finished_orders(self.finished_orders),
                                self.res.tardy_orders,
                                current_time]
            
        return state_representation, order_categories

    def clip_state(self, state):
        # Clipping and normalization operation
        # x_norm = (x - min(x)) / (max(x) - min(x))
        state_norm = []
        for index, state in enumerate(state):
            if index < 15:
                clipped_value = min(state, self.config['state_clipping'])
                state_norm.append(clipped_value / (self.config['state_clipping']))
                
            elif index == 15 or index == 16:
                state_norm.append(state)

            elif index == 17:
                clipped_value = min(state, self.config_environment['throughput'])
                state_norm.append(clipped_value / (self.config_environment['throughput']))

            elif index == 18:
                clipped_value = min(state, self.config_environment['throughput'] / 2)
                state_norm.append(clipped_value / (self.config_environment['throughput'] / 2))
                
            elif index == 19:
                if state > 24 * 3600:
                    state -= 24 * 3600
                time = (state / 3600) / 24
                state_norm.append(time)
        return state_norm

    def action_to_orders(self, action):
        categories = self.action_category_mapping[action]
        for i in categories:
            if len(self.order_categories[i]) > 0:
                order_category = i
                break
            
        picking_items = self.order_categories[order_category]
        all_picking_items = self.order_categories[order_category]
        
        if action not in self.actions_pick_by_batch:  # pick-by-order decision
            
            # cutoff_time, nOrders, nItems_ptg, nItems_gtp, route
            picking_items = picking_items.iloc[0:1]
            picking_order = [picking_items['cutoff_time'].iloc[0], 1,
                             sum(picking_items['nItems_ptg']), sum(picking_items['nItems_gtp']),
                             self.action_route_mapping[action]]

        elif action in self.actions_pick_by_batch:  # pick-by-batch decision
            # PtG Batching
            if action in [1, 5]:
                
                if action == 1:  # SIO order
                    picking_items_batch = picking_items.iloc[0:1]
                    sku = picking_items['skuIDlist'].iloc[0][0]
                    
                    for item in range(len(picking_items)):
                        if picking_items['skuIDlist'].iloc[item][0] == sku and item > 0:
                            picking_items_batch = picking_items_batch.append(picking_items.iloc[item])
                    
                    batch_size = min(self.max_batchsize_ptg, len(picking_items_batch))
                    item_list = list(set(x for item in picking_items_batch['skuIDlist'].to_list() for x in item))   
                    
                    # Check if item constraint of batch is not violated
                    if len(item_list) > self.max_batchsize_ptg_items:
                        for order in range(len(picking_items_batch)):
                            picking_items_batch = picking_items_batch[:-1]
                            item_list = list(set(x for item in picking_items_batch['skuIDlist'].to_list() for x in item))   
                            batch_size = min(self.max_batchsize_ptg, len(picking_items_batch))
                            
                            if len(item_list) == 0:
                                picking_items_batch = picking_items.iloc[0:1]
                                break
                            
                            if len(item_list) <= self.max_batchsize_ptg_items:
                                break
        
                    if batch_size > 1:
                        picking_items = picking_items_batch.iloc[0:batch_size]
                        picking_order = [picking_items['cutoff_time'].iloc[0], batch_size, 0,  
                                         1, self.action_route_mapping[action]]
                    else:
                        picking_items = picking_items.iloc[0:batch_size]
                        picking_order = [picking_items['cutoff_time'].iloc[0], batch_size,
                                         sum(picking_items['nItems_ptg']),
                                         sum(picking_items['nItems_gtp']), 
                                         self.action_route_mapping[action]]
                        
                elif action == 5:  # MIO order
                    picking_items_batch = picking_items.iloc[0:1]
                    sku_list = picking_items_batch['skuIDlist'].iloc[0]
                    for sku in sku_list:
                        for item in range(len(picking_items)):
                            test_item = picking_items['skuIDlist'].iloc[item]
                            if sku in test_item and item > 0:
                                picking_items_batch = picking_items_batch.append(picking_items.iloc[item])
                                
                    if len(picking_items_batch) > 1:
                        item_list = list(set(x for item in picking_items_batch['skuIDlist'].tolist() for x in item))
                        
                        # Check if items constraint of batch is not violated
                        if len(item_list) > self.max_batchsize_ptg_items:
                            for order in range(len(picking_items_batch)):
                                picking_items_batch = picking_items_batch[:-1]
                                item_list = list(set(x for item in picking_items_batch['skuIDlist'].to_list() for x in item))   
                                
                                if len(item_list) == 0:
                                    picking_items_batch = picking_items.iloc[0:1]
                                    break
                                
                                if len(item_list) <= self.max_batchsize_ptg_items:
                                    break
                        
                        pick_movements = list(set(x for item in picking_items_batch['skuIDlist'].tolist() for x in item))
                        picking_items = picking_items_batch
                        picking_order = [picking_items['cutoff_time'].iloc[0], len(picking_items), 0,  
                                         len(pick_movements), self.action_route_mapping[action]]
                    else:
                        batch_size_new = min(self.max_batchsize_ptg, len(picking_items))
                        picking_items = picking_items.iloc[0:batch_size_new]
                        picking_order = [picking_items['cutoff_time'].iloc[0], batch_size_new,
                                         sum(picking_items['nItems_ptg']),
                                         sum(picking_items['nItems_gtp']), 
                                         self.action_route_mapping[action]]
                        
            # GtP Batching  
            elif action in [3, 7, 8]:
                if action == 3:  # SIO order
                    picking_items_batch = picking_items.iloc[0:1]
                    sku = picking_items['skuIDlist'].iloc[0][0]
                    
                    for item in range(len(picking_items)):
                        if picking_items['skuIDlist'].iloc[item][0] == sku and item > 0:
                            picking_items_batch = picking_items_batch.append(picking_items.iloc[item])
                    
                    batch_size = min(self.max_batchsize_gtp, len(picking_items_batch))
                    picking_items = picking_items_batch.iloc[0:batch_size]
                    picking_order = [picking_items['cutoff_time'].iloc[0], batch_size, 0,  
                                     1, self.action_route_mapping[action]]
                    
                else:  # MIO order
                    picking_items_batch = picking_items.iloc[0:1]
                    sku_list = picking_items_batch['skuIDlist'].iloc[0]
                    for sku in sku_list:
                        for item in range(len(picking_items)):
                            test_item = picking_items['skuIDlist'].iloc[item]
                            if sku in test_item and item > 0:
                                picking_items_batch = picking_items_batch.append(picking_items.iloc[item])
                    
                    if action == 7:  # mio_gtp
                        batch_size = min(self.max_batchsize_gtp, len(picking_items_batch))
                        pick_movements = list(set(x for item in picking_items_batch['skuIDlist'].tolist() for x in item))
                        picking_items = picking_items_batch.iloc[0:batch_size]
                        picking_order = [picking_items['cutoff_time'].iloc[0], len(picking_items), 0,  
                                         len(pick_movements), self.action_route_mapping[action]]
                    else:  # mio_ptg_gtp
                        batch_size = min(self.max_batchsize_ptg_gtp, len(picking_items_batch))
                        picking_items = picking_items_batch.iloc[0:batch_size]
                        pick_movements = list(set(x for item in picking_items['skuIDlist'].tolist() for x in item))
                        picking_order = [picking_items['cutoff_time'].iloc[0], len(picking_items),
                                         len(pick_movements[:len(pick_movements)//2]),
                                         len(pick_movements[len(pick_movements)//2:]),
                                         self.action_route_mapping[action]]
                        
        return picking_order, picking_items, order_category, all_picking_items
    
    def resource_rebalance(self, t):
        # If PtG operator is idle for 15 minutes, allocate to DtO work station 1 operator at a time
        if self.DtO_resource_transfer == 0:
            if self.PtG_picker_available > 0:
                self.DtO_resource_transfer = t
                
        elif self.DtO_resource_transfer != 0 and t - self.DtO_resource_transfer > self.rebalance_interval:
            self.PtG_picker_available -= 1  
            self.DtO_operator_available += 1
            self.DtO_resource_transfer = 0
            
        elif self.DtO_resource_transfer != 0:
            if self.PtG_picker_available == 0:
                self.DtO_resource_transfer = t
                
    def change_shift(self, time):
        if not self.shift_change:
            if time > self.config['shift_start'][1]*3600:
                self.PtG_picker_available += (self.config['nPtG_pickers'][1] - self.config['nPtG_pickers'][0])
                self.GtP_shuttle_available += (self.config['nGtP_shuttles'][1] - self.config['nGtP_shuttles'][0])
                self.DtO_operator_available += (self.config['nDtO_operators'][1] - self.config['nDtO_operators'][0])
                self.StO_operator_available += (self.config['nStO_operators'][1] - self.config['nStO_operators'][0])
                self.shift_change = True
                
    def custom_heuristic(self, state):
        resource_availability = {1: state[15], 2: state[15],
                                 3: state[15], 4: state[16],
                                 5: state[16], 6: state[15]}
        routes_available = [x for x in resource_availability.keys() if resource_availability[x] > 0]
        
        actions_available_resource = [self.route_action_mapping[x] for x in routes_available]
        actions_available_resource = list(set(x for action in actions_available_resource for x in action))

        # Compute action availability based on order categories
        order_categories_available = [idx for idx, x in enumerate(state) if state[idx] > 0 and idx <= 14]

        actions_available_orders = [self.category_action_mapping[x] for x in order_categories_available]
        actions_available_orders = list(set(x for action in actions_available_orders for x in action))
        
        # Concatenate both lists
        actions_available = []
        for i in actions_available_orders:
            if i in actions_available_resource:
                actions_available.append(i)
                
        # Prefer pick-by-batch decision over pick-by-order decision
        orders_earliness_e1 = [0, 3, 6, 9, 12]
        orders_earliness_e2 = [1, 4, 7, 10, 13]
        orders_earliness_e3 = [3, 5, 8, 11, 14]

        random.shuffle(actions_available)
        chosen_action = False
        
        for action in actions_available:
            if self.action_category_mapping[action][0] in orders_earliness_e1:
                if action in self.actions_pick_by_batch:
                    chosen_action = action
                    break
        
        if not chosen_action:
            for action in actions_available:
                if self.action_category_mapping[action][1] in orders_earliness_e2:
                    if action in self.actions_pick_by_batch:
                        chosen_action = action
                        break
                
        if not chosen_action:
            for action in actions_available:
                if self.action_category_mapping[action][2] in orders_earliness_e3:
                    if action in self.actions_pick_by_batch:
                        chosen_action = action
                        break 

        # if the action is still false, resources and orders are not available, wait action is selected
        if not chosen_action:
            chosen_action = 10

        return chosen_action

    def edd_sequencing(self, state):
        # Compute action availability based on resources
        resource_availability = {1: state[15], 2: state[15],
                                 3: state[15], 4: state[16],
                                 5: state[16], 6: state[15]}
        routes_available = [x for x in resource_availability.keys() if resource_availability[x] > 0]

        actions_available_resource = [self.route_action_mapping[x] for x in routes_available]
        actions_available_resource = list(set(x for action in actions_available_resource for x in action))

        # Compute action availability based on order categories
        order_categories_available = [idx for idx, x in enumerate(state) if state[idx] > 0 and idx <= 14]

        actions_available_orders = [self.category_action_mapping[x] for x in order_categories_available]
        actions_available_orders = list(set(x for action in actions_available_orders for x in action))

        # Concatenate both lists
        actions_available = []
        for i in actions_available_orders:
            if i in actions_available_resource:
                actions_available.append(i)

        # Prefer pick-by-batch decision over pick-by-order decision
        orders_earliness_e1 = [0, 3, 6, 9, 12]
        orders_earliness_e2 = [1, 4, 7, 10, 13]
        orders_earliness_e3 = [3, 5, 8, 11, 14]

        random.shuffle(actions_available)
        chosen_action = False
        
        for action in actions_available:
            if self.action_category_mapping[action][0] in orders_earliness_e1 and action in self.actions_pick_by_batch:
                chosen_action = action
                break
            if self.action_category_mapping[action][0] in orders_earliness_e1:
                chosen_action = action
                break
        
        if not chosen_action:
            for action in actions_available:
                if self.action_category_mapping[action][1] in orders_earliness_e2 and action in self.actions_pick_by_batch:
                    chosen_action = action
                    break
                if self.action_category_mapping[action][1] in orders_earliness_e2:
                    chosen_action = action
                    break
                
        if not chosen_action:
            for action in actions_available:
                if self.action_category_mapping[action][2] in orders_earliness_e3 and action in self.actions_pick_by_batch:
                    chosen_action = action
                    break  
                if self.action_category_mapping[action][2] in orders_earliness_e3:
                    chosen_action = action
                    break  

        # if the action is still false, resources and orders are not available, wait action is selected
        if not chosen_action:
            chosen_action = 10

        return chosen_action
    
    def LST_batching(self, action, picking_items, t):
        if action in [1,5]:
            if action in self.actions_pick_by_batch:
                new_batch = picking_items.iloc[0:self.max_batchsize_ptg]
                slack_batch = picking_items.iloc[0][7] - t
                
                slack_batch -= (self.config['PtG_picking_constant'] + self.PtG_Out_time)
                
                if action == 5:
                    slack_batch -= (self.PtG_GtP_time + self.StO_time + self.GtP_StO_time)
                
                for order in range(len(new_batch)):
                    if action == 1: # only for SIO PtG batching
                        processing_time = picking_items.iloc[order][4] * self.config['PtG_picking_time']
                    elif action == 5: # only for MIO PtG batching
                        processing_time = picking_items.iloc[order][4] * self.config['PtG_picking_time'] + self.GtP_picking_time * new_batch.iloc[order][5]
                    
                    slack_batch -= processing_time
                    
                for order in range(len(new_batch)):
                    if slack_batch < 0:
                        new_batch = new_batch[:-1]
                        
                        if action == 1: # only for SIO PtG batching
                            processing_time = picking_items.iloc[order][4] * self.config['PtG_picking_time']
                        elif action == 5: # only for MIO PtG batching
                            processing_time = picking_items.iloc[order][4] * self.config['PtG_picking_time'] + self.GtP_picking_time * picking_items.iloc[order][5]
                    
                        slack_batch += processing_time
                        
                        if len(new_batch) == 0:
                            new_batch = picking_items.iloc[0:1]
                            break
                        if slack_batch >= 0:
                            break
                        
                # Check if items constraint of batch is not violated
                item_list = list(set(x for item in new_batch['skuIDlist'].tolist() for x in item))
                if len(item_list) > self.max_batchsize_ptg_items:
                    for order in range(len(new_batch)):
                        new_batch = new_batch[:-1]
                        item_list = list(set(x for item in new_batch['skuIDlist'].to_list() for x in item))   
                        
                        if len(item_list) == 0:
                            new_batch = picking_items.iloc[0:1]
                            break
                        
                        if len(item_list) <= self.max_batchsize_ptg_items:
                            break
                        
            else:
                new_batch = picking_items
        else:
            new_batch = picking_items

        self.batch_list.append(len(new_batch))
        return new_batch
    
    def grasp_vnd(self, action, picking_items, all_picking_items, t):
        if action in [1, 5] and len(picking_items) > 1:
            # GRASP batching method --> constructive method
            alpha = random.random()
            batch_list = all_picking_items.iloc[0:1]
            all_picking_items = all_picking_items[~all_picking_items['orderID'].isin(batch_list['orderID'])]
            
            for i in range(len(all_picking_items)):
                threshold = int(round(all_picking_items['cutoff_time'].argmin() - alpha * (all_picking_items['cutoff_time'].argmin() - all_picking_items['cutoff_time'].argmax())))
                threshold = min(threshold, len(all_picking_items))
                threshold = max(1, threshold)
                restricted_order_list = all_picking_items.iloc[0:threshold]
                selected_order = restricted_order_list.sample(n=1)
    
                batch_list = batch_list.append(selected_order)
            
            # batch_list contains sorted orders that need to be arranged in batches of max 10
            # These are initially batched using the LST batching algorithm
            new_batch_list = []
            while len(all_picking_items) > 0:
                new_batch = self.LST_batching(action, all_picking_items, t)
                new_batch_list.append(new_batch)
                all_picking_items = all_picking_items[~all_picking_items['orderID'].isin(new_batch['orderID'])]
                
            # new_batch_list contains a list with batches compiled by the constructor and the LST heuristic
            # Now Variable Neighborhood Descent will be applied to perform insert and swap moves
            k = 1
            k_max = 7
            
            best_solution = new_batch_list
            if len(new_batch_list) == 0:
                print('Batch list is empty !')
                print(picking_items)
            best_solution_result = self.evaluate_solution(new_batch_list)
            
            while k != k_max:
                if k in [1, 4, 7, 10]:
                    solution = self.local_search_grasp(best_solution, 1)
                elif k in [2, 5, 8, 11]:
                    solution = self.local_search_grasp(best_solution, 2)
                elif k in [3, 6, 9, 12]:
                    solution = self.local_search_grasp(best_solution, 3)
                
                solution_result = self.evaluate_solution(solution)
                if solution_result < best_solution_result:
                    best_solution = solution
                    best_solution_result = solution_result
                    k = 1
                else:
                    k += 1
            
            # return the first batch of best solution
            new_solution = best_solution[0]
                    
        else:
            new_solution = picking_items
        
        return new_solution
    
    def evaluate_solution(self, batch_list):
        # all batches in batch list are either MIO or SIO PtG orders
        # for each batch, a completion time is computed.
        # the max completion time defines the solution quality
        pick_times = []
        for batch in batch_list:
            item_list = batch['skuIDlist'].tolist()
            
            # remove all duplicates and compute picking time
            item_list = list(set(x for item in item_list for x in item))    
            pick_time = len(item_list) * self.PtG_picking_item.rvs() + self.PtG_picking_constant
            pick_times.append(pick_time)
            
        max_picking_time = max(pick_times)
        return max_picking_time
    
    def local_search_grasp(self, batch_list, move):
        if move == 1:
            # Insert an order within another batch
            # This cannot violate the maximum batch size constraint
            no_batches = len(batch_list)

            # Collect batches where an order can be inserted
            batches_available_in = [idx for idx, batch in enumerate(batch_list) if len(batch) < self.max_batchsize_ptg]
            if len(batches_available_in) > 0:
                # sample batch and order
                batch_out = random.sample(range(0, no_batches), k=1)[0]
                batch_in = random.sample(batches_available_in, k = 1)[0]
                if batch_out == batch_in:
                    batch_out = random.sample(range(0, no_batches), k=1)[0]
                
                sample_batch_out = batch_list[batch_out]
                
                # insert order
                order_out = random.sample(range(0,len(sample_batch_out)), k=1)[0]
                sample_order_out = sample_batch_out.iloc[order_out-1:order_out]
                batch_list[batch_in] = batch_list[batch_in].append(sample_order_out)
                
                # remove order from old batch
                batch_list[batch_out] = batch_list[batch_out][~batch_list[batch_out]['orderID'].isin(sample_order_out['orderID'])]

        elif move == 2:
            # Swap move 1: randomly swap a single order between two batches
            no_batches = len(batch_list)
            
            batch_out = random.sample(range(0, no_batches), k=1)[0]
            batch_in = random.sample (range(0, no_batches), k = 1)[0]
            if batch_out == batch_in:
                batch_out = random.sample(range(0, no_batches), k=1)[0]
                
            sample_batch_out = batch_list[batch_out]
            sample_batch_in = batch_list[batch_in]
            
            # select order
            order_out = random.sample(range(0,len(sample_batch_out)), k=1)[0]
            order_in = random.sample(range(0,len(sample_batch_in)), k=1)[0]
            
            # swap orders and remove order from old batch
            sample_order_out = sample_batch_out.iloc[order_out-1:order_out]
            batch_list[batch_in] = batch_list[batch_in].append(sample_order_out)
            batch_list[batch_out] = batch_list[batch_out][~batch_list[batch_out]['orderID'].isin(sample_order_out['orderID'])]
            
            sample_order_in = sample_batch_in.iloc[order_in-1:order_in]
            batch_list[batch_out] = batch_list[batch_out].append(sample_order_in)
            batch_list[batch_in] = batch_list[batch_in][~batch_list[batch_in]['orderID'].isin(sample_order_in['orderID'])]
            
        elif move == 3:
            # Swap move 2: randomly swap a set of two orders with another single order
            no_batches = len(batch_list)
            
            # Collect batches where a set of 2 orders can be inserted
            batches_available_in = [idx for idx, batch in enumerate(batch_list) if len(batch) <= 9]
            if len(batches_available_in) > 0:
                # sample batch and order
                batch_out = random.sample(range(0, no_batches), k=1)[0]
                batch_in = random.sample (batches_available_in, k = 1)[0]
                if batch_out == batch_in:
                    batch_out = random.sample(range(0, no_batches), k=1)[0]
                
                sample_batch_out = batch_list[batch_out]
                sample_batch_in = batch_list[batch_in]
                
                if len(sample_batch_out) >= 2:
                    # select order
                    orders_out = random.sample(range(0,len(sample_batch_out)), k=2)
                    order_1_out = sample_batch_out.iloc[orders_out[0]-1:orders_out[0]]
                    order_2_out = sample_batch_out.iloc[orders_out[1]-1:orders_out[1]]
                    
                    order_in = random.sample(range(0,len(sample_batch_in)), k=1)[0]
                    
                    # swap orders and remove order from old batch
                    batch_list[batch_in] = batch_list[batch_in].append(order_1_out)
                    batch_list[batch_out] = batch_list[batch_out][~batch_list[batch_out]['orderID'].isin(order_1_out['orderID'])]
        
                    batch_list[batch_in] = batch_list[batch_in].append(order_2_out)
                    batch_list[batch_out] = batch_list[batch_out][~batch_list[batch_out]['orderID'].isin(order_2_out['orderID'])]
        
                    sample_order_in = sample_batch_in.iloc[order_in-1:order_in]
                    batch_list[batch_out] = batch_list[batch_out].append(sample_order_in)
                    batch_list[batch_in] = batch_list[batch_in][~batch_list[batch_in]['orderID'].isin(sample_order_in['orderID'])]
                    
        return batch_list
    
    def boc_batching(self, action, picking_items, all_picking_items):
        # BOC batching method
        if action in [1, 5] and len(picking_items) > 1 and len(all_picking_items) > 1:
            
            if action == 1:  # SIO order
                # 1. select seed order with the most items
                sku_item = all_picking_items["skuIDlist"].value_counts().index[0]
                for order in range(len(all_picking_items)):
                    if all_picking_items["skuIDlist"].iloc[order] == sku_item:
                        picking_items_batch = all_picking_items.iloc[order:order+1]
                        break
                    
                # 2. compute similarity coefficients for all remaining orders
                for i in range(self.max_batchsize_ptg - 1):
                    if len(all_picking_items) > 1:
                        for order in range(len(picking_items_batch)):
                            all_picking_items = all_picking_items[all_picking_items['orderID'] != picking_items_batch['orderID'].iloc[order]]
                            
                        similarity_coefficient = []
                        sku_list = picking_items_batch['skuIDlist'].to_list()
                        sku_list = [item for sublist in sku_list for item in sublist]
                        
                        for order in range(len(all_picking_items)):
                            similarity_counter = 0
                            items_order = all_picking_items['skuIDlist'].iloc[order]
                            for sku in sku_list:
                                for item in items_order:
                                    if sku == item:
                                        similarity_counter += 1 
                            similarity_coefficient.append(similarity_counter / len(items_order))
                            
                        # 3. Combine seed order with selected order
                        index_order = np.argmax(similarity_coefficient)
                        
                        # if selected order is not similar to seed orders, choose order with most imminent cutoff time
                        if max(similarity_coefficient) == 0:
                            index_order = all_picking_items["cutoff_time"].argmin()
                            
                        picking_items_batch = picking_items_batch.append(all_picking_items.iloc[index_order:index_order+1])
                    
                # Remove last order from picking_items
                all_picking_items = all_picking_items[all_picking_items['orderID'] != picking_items_batch['orderID'].iloc[-1]]
                    
            elif action == 5:  # MIO order
                # 1. select seed order with the most items
                max_length = 0
                for order in range(len(all_picking_items)):
                    if len(all_picking_items["skuIDlist"].iloc[order]) > max_length:
                        max_length = len(all_picking_items["skuIDlist"].iloc[order])
                        picking_items_batch = all_picking_items.iloc[order:order+1]
                
                # 2. compute similarity coefficients for all remaining orders
                for i in range(self.max_batchsize_ptg - 1):
                    for order in range(len(picking_items_batch)):
                        all_picking_items = all_picking_items[all_picking_items['orderID'] != picking_items_batch['orderID'].iloc[order]]
                        
                    similarity_coefficient = []
                    sku_list = picking_items_batch['skuIDlist'].to_list()
                    sku_list = [item for sublist in sku_list for item in sublist]
                    
                    for order in range(len(all_picking_items)):
                        similarity_counter = 0
                        items_order = all_picking_items['skuIDlist'].iloc[order]
                        for sku in sku_list:
                            for item in items_order:
                                if sku == item:
                                    similarity_counter += 1 
                        similarity_coefficient.append(similarity_counter / len(items_order))
                        
                    # 3. Combine seed order with selected order
                    if len(all_picking_items) > 0:
                        if len(similarity_coefficient) == 0:
                            index_order = all_picking_items["cutoff_time"].argmin()
                        else:
                            index_order = np.argmax(similarity_coefficient)

                        # if selected order is not similar to seed orders, choose order with most imminent cutoff time
                        if max(similarity_coefficient) == 0:
                            index_order = all_picking_items["cutoff_time"].argmin()

                        picking_items_batch = picking_items_batch.append(all_picking_items.iloc[index_order:index_order+1])

                # Remove last order from picking_items
                all_picking_items = all_picking_items[all_picking_items['orderID'] != picking_items_batch['orderID'].iloc[-1]]                    
                        
        else:
            picking_items_batch = picking_items
        
        return picking_items_batch

    def GVNS_batching(self, action, picking_items, all_picking_items):
        # 1. Compute constructive solution with EDD batching heuristic
        all_picking_items = all_picking_items.sort_values(by='cutoff_time', ascending=True)
        batches = []
        if action in [1, 5] and len(picking_items) > 1:
            for order in range(len(all_picking_items)):
                if len(all_picking_items) > 0:
                    # Compile batch
                    batch = all_picking_items.iloc[0:self.max_batchsize_ptg-1]
                    batches.append(batch)
                    
                    # Remove batched orders from all_picking_items
                    all_picking_items = all_picking_items[~all_picking_items['orderID'].isin(batch['orderID'])]  

            # 2. Execute neighborhood search
            max_steps = 5
            k_max = 2
            best_solution = batches
            best_solution_result = self.evaluate_solution(batches)
            
            for step in range(max_steps):
                for k in range(1, k_max):
                    solution_1 = self.local_search_gnvs(best_solution, 1, k) # SHAKE
                    solution_2 = self.vnd_gnvs(best_solution, k_max) # VND
                    
                    solution_1_result = self.evaluate_solution(solution_1)
                    solution_2_result = self.evaluate_solution(solution_2)
                    
                    if best_solution_result > solution_1_result and solution_2_result > solution_1_result:
                        best_solution = solution_1
                        best_solution_result = solution_1_result
                        
                    elif best_solution_result > solution_2_result and solution_1_result > solution_2_result:
                        best_solution = solution_2
                        best_solution_result = solution_2_result
                        
            picking_items_batch = best_solution[0]
                
        else:
            picking_items_batch = picking_items
        
        return picking_items_batch
    
    def local_search_gnvs(self, batch_list, move, k=1):
        if move == 1:
            for i in range(k):
                no_batches = len(batch_list)
                # Collect batches where an order can be inserted
                batches_available_in = [idx for idx, batch in enumerate(batch_list) if len(batch) < self.max_batchsize_ptg]
                if len(batches_available_in) > 0:
                    # sample batch and order
                    batch_out = random.sample(range(0, no_batches), k=1)[0]
                    batch_in = random.sample(batches_available_in, k = 1)[0]
                    if batch_out == batch_in:
                        batch_out = random.sample(range(0, no_batches), k=1)[0]
                    
                    sample_batch_out = batch_list[batch_out]
                    
                    # insert order
                    order_out = random.sample(range(0,len(sample_batch_out)), k=1)[0]
                    sample_order_out = sample_batch_out.iloc[order_out-1:order_out]
                    
                    # append order to new batch
                    batch_list[batch_in] = batch_list[batch_in].append(sample_order_out)
                    
                    # remove order from old batch
                    batch_list[batch_out] = batch_list[batch_out][~batch_list[batch_out]['orderID'].isin(sample_order_out['orderID'])]                                          

        elif move == 2:
            # Swap move 1: randomly swap a single order between two batches
            no_batches = len(batch_list)
            
            batch_out = random.sample(range(0, no_batches), k=1)[0]
            batch_in = random.sample (range(0, no_batches), k = 1)[0]
            if batch_out == batch_in:
                batch_out = random.sample(range(0, no_batches), k=1)[0]
                
            sample_batch_out = batch_list[batch_out]
            sample_batch_in = batch_list[batch_in]
            
            # select order
            order_out = random.sample(range(0,len(sample_batch_out)), k=1)[0]
            order_in = random.sample(range(0,len(sample_batch_in)), k=1)[0]
            
            # swap orders and remove order from old batch
            sample_order_out = sample_batch_out.iloc[order_out-1:order_out]
            batch_list[batch_in] = batch_list[batch_in].append(sample_order_out)
            batch_list[batch_out] = batch_list[batch_out][~batch_list[batch_out]['orderID'].isin(sample_order_out['orderID'])]
            
            sample_order_in = sample_batch_in.iloc[order_in-1:order_in]
            batch_list[batch_out] = batch_list[batch_out].append(sample_order_in)
            batch_list[batch_in] = batch_list[batch_in][~batch_list[batch_in]['orderID'].isin(sample_order_in['orderID'])]
            
        return batch_list
    
    def vnd_gnvs(self, batches, k_max):
        best_solution = batches
        best_solution_result = self.evaluate_solution(batches)
        for x in range(k_max):
            solution_1 = self.local_search_gnvs(batches, 1, k=k_max)
            solution_2 = self.local_search_gnvs(batches, 2, k=k_max)
            
            solution_1_result = self.evaluate_solution(solution_1)
            solution_2_result = self.evaluate_solution(solution_2)
            
            if best_solution_result > solution_1_result and solution_2_result > solution_1_result:
                best_solution = solution_1
                best_solution_result = solution_1_result
                
            elif best_solution_result > solution_2_result and solution_1_result > solution_2_result:
                best_solution = solution_2
                best_solution_result = solution_2_result
                
        return best_solution
                
            

    def random_policy(self, state):
        # Compute action availability based on resources
        resource_availability = {1: state[15], 2: state[15],
                                 3: state[15], 4: state[16],
                                 5: state[16], 6: state[15]}
        routes_available = [x for x in resource_availability.keys() if resource_availability[x] > 0]

        actions_available_resource = [self.route_action_mapping[x] for x in routes_available]
        actions_available_resource = list(set(x for action in actions_available_resource for x in action))

        # Compute action availability based on order categories
        order_categories_available = [idx for idx, x in enumerate(state) if state[idx] > 0 and idx <= 14]

        actions_available_orders = [self.category_action_mapping[x] for x in order_categories_available]
        actions_available_orders = list(set(x for action in actions_available_orders for x in action))

        # Concatenate both lists
        actions_available = []
        for i in actions_available_orders:
            if i in actions_available_resource:
                actions_available.append(i)
                
        action = random.choice(actions_available)
        
        return action
    
    # @profile
    def remove_orders(self, action, picking_items, order_category):
        if action in self.actions_pick_by_batch:
            self.order_data = self.order_data[~self.order_data['orderID'].isin(picking_items['orderID'])]
            batch_size = len(picking_items)

            if len(picking_items) == 1:
                picking_order = [picking_items['cutoff_time'].iloc[0], len(picking_items['nItems_ptg']),
                                 sum(picking_items['nItems_ptg']),  sum(picking_items['nItems_gtp']),
                                 self.action_route_mapping[action]]
                
            else:
                if action not in [1, 5]:
                    picking_order = [picking_items['cutoff_time'].iloc[0], batch_size,
                                     sum(picking_items['nItems_ptg'].iloc[0:batch_size]),
                                     sum(picking_items['nItems_gtp'].iloc[0:batch_size]), self.action_route_mapping[action]]
                else:
                    pick_movements_ptg = list(set(x for item in picking_items['skuIDlist'].tolist() for x in item))
                    picking_order = [picking_items['cutoff_time'].iloc[0], batch_size,
                                     len(pick_movements_ptg),
                                     0, self.action_route_mapping[action]]
                
        else:
            picking_items = picking_items.iloc[0:1]
            self.order_data = self.order_data[~self.order_data['orderID'].isin(picking_items['orderID'])]
            picking_order = [picking_items['cutoff_time'].iloc[0], len(picking_items['nItems_ptg']),
                             sum(picking_items['nItems_ptg']), sum(picking_items['nItems_gtp']),
                             self.action_route_mapping[action]]
            
        # Remove orders from order_categories
        all_picking_items = self.order_categories[order_category]
        filtered_picking_items = all_picking_items[~all_picking_items['orderID'].isin(picking_items['orderID'])]
        self.order_categories[order_category] = filtered_picking_items

        # change route if batch contains only 1 order and is considered as pick-by-order
        if action in self.actions_pick_by_batch:
            nOrders = picking_order[1]
            route = picking_order[4]
            if nOrders == 1:
                if action == 1:
                    route = 1
                elif action == 5:
                    route = 1
                elif action == 7:
                    route = 5
                elif action == 8:
                    route = 6

            picking_order[4] = route

        return picking_order, picking_items
    
    def check_action(self, action, state):
        # Compute action availability based on resources
        resource_availability = {1: state[15], 2: state[15],
                                 3: state[15], 4: state[16],
                                 5: state[16], 6: state[15]}
        routes_available = [x for x in resource_availability.keys() if resource_availability[x] > 0]

        actions_available_resource = [self.route_action_mapping[x] for x in routes_available]
        actions_available_resource = list(set(x for action in actions_available_resource for x in action))

        # Compute action availability based on order categories
        order_categories_available = [idx for idx, x in enumerate(state) if state[idx] > 0 and idx <= 14]

        actions_available_orders = [self.category_action_mapping[x] for x in order_categories_available]
        actions_available_orders = list(set(x for action in actions_available_orders for x in action))

        # Concatenate both lists
        actions_available = []
        for i in actions_available_orders:
            if i in actions_available_resource:
                actions_available.append(i)
                
        if action != 10:
            if action in actions_available:
                feasibility = True
            else:
                feasibility = False
        
        if action == 10:
            if len(actions_available) == 0:
                feasibility = True
            else:
                feasibility = False
                
        return feasibility

    def available_actions(self, state):
        resource_availability = {1: state[15], 2: state[15],
                                 3: state[15], 4: state[16],
                                 5: state[16], 6: state[15]}

        order_categories_available = [idx for idx, x in enumerate(state)
                                      if x > 0 and idx <= 14]

        actions_available_orders = [self.category_action_mapping[x] for x in order_categories_available]
        actions_available_orders = list(set(x for j in actions_available_orders for x in j))

        # 2. Resource availability
        actions_available_resources = [x for x in self.action_route_mapping.keys() if
                                       resource_availability[self.action_route_mapping[x]] > 0]
        actions_available = []
        for i in actions_available_orders:
            if i in actions_available_resources:
                actions_available.append(i)

        valid_actions = []
        if len(actions_available) > 0:
            for i in range(self.nActions):
                if i in actions_available:
                    valid_actions.append(1)
                else:
                    valid_actions.append(0)
        else:
            for i in range(self.nActions):
                if i != 10:
                    valid_actions.append(0)
                else:
                    valid_actions.append(1)
        return valid_actions

    def set_weight_settings(self, weights, t):
        t_hours = t / 3600

        if 15 <= t_hours < 17:
            self.weight_tardy = weights['scenario_1'][0]
            self.weight_picking = weights['scenario_1'][1]
        elif 17 <= t_hours < 22:
            self.weight_tardy = weights['scenario_2'][0]
            self.weight_picking = weights['scenario_2'][1]
        elif t_hours >= 22:
            self.weight_tardy = weights['scenario_3'][0]
            self.weight_picking = weights['scenario_3'][1]
        else:
            self.weight_tardy = 1
            self.weight_picking = 1

    def get_weight_settings(self, weights, t):
        t_hours = t / 3600

        if 15 <= t_hours < 17:
            weight_tardy = weights['scenario_1'][0]
        elif 20 <= t_hours < 22:
            weight_tardy = weights['scenario_2'][0]
        elif t_hours >= 22:
            weight_tardy = weights['scenario_3'][0]
        else:
            weight_tardy = 1

        return weight_tardy

    def get_reward(self, action):
        # set correct weight settings
        # self.set_weight_settings(self.weights, self.state_representation[:][-1])

        self.reward_action = 0
        feasibility = self.check_action(action, self.state_representation[:])
        tardy_orders = self.state_representation[:][-2] - self.tardy_order_hist
        self.tardy_order_hist = self.state_representation[:][-2]

        if not feasibility:
            self.reward_action += self.reward_structure['infeasible_action']
            self.reward_distribution['infeasible_action'] += self.reward_structure['infeasible_action']

        if tardy_orders > 0:
            self.reward_action += (self.reward_structure['tardy_order'] * tardy_orders) * self.weight_tardy
            self.reward_distribution['tardy_order'] += self.reward_structure['tardy_order'] * tardy_orders

        if action in self.actions_pick_by_batch:
            # self.reward_action += self.reward_structure['batch_action'] * self.weight_picking
            self.reward_distribution['batch_action'] += self.reward_structure['batch_action']

        if action < 10:
            if action in [0, 1, 4, 5]:  # PtG orders
                # self.reward_action -= ((1 - self.nOrders_hist / self.max_batchsize_ptg)/5) * self.weight_picking
                self.avg_batch_size.append(self.nOrders_hist / self.max_batchsize_ptg)
                self.picking_time.append((self.nItems_ptg_hist * self.PtG_picking_item.rvs() +
                                          self.PtG_picking_constant) / self.nOrders_hist)
                self.reward_distribution['batch_composition'] -= (1 - self.nOrders_hist / self.max_batchsize_ptg)/5
                if action in self.actions_pick_by_batch:
                    self.avg_size_pick_batch.append(self.nOrders_hist / self.max_batchsize_ptg)
            elif action in [2, 3, 6, 7]:  # GtP orders
                # self.reward_action -= ((1 - self.nOrders_hist / self.max_batchsize_gtp)/5) * self.weight_picking
                self.avg_batch_size.append(self.nOrders_hist / self.max_batchsize_gtp)
                self.picking_time.append((self.nItems_gtp_hist * self.GtP_picking_time) / self.nOrders_hist)
                self.reward_distribution['batch_composition'] -= (1 - self.nOrders_hist / self.max_batchsize_gtp)/5
                if action in self.actions_pick_by_batch:
                    self.avg_size_pick_batch.append(self.nOrders_hist / self.max_batchsize_gtp)
            elif action in [8, 9]:  # PtG GtP orders
                # self.reward_action -= ((1 - self.nOrders_hist / self.max_batchsize_ptg_gtp)/5) * self.weight_picking
                self.avg_batch_size.append(self.nOrders_hist / self.max_batchsize_ptg_gtp)
                self.picking_time.append((self.nItems_ptg_hist * self.PtG_picking_item.rvs() + self.PtG_picking_constant
                                         + self.nItems_gtp_hist * self.GtP_picking_time) / self.nOrders_hist)
                self.reward_distribution['batch_composition'] -= (1 - self.nOrders_hist / self.max_batchsize_ptg_gtp)/5
                if action in self.actions_pick_by_batch:
                    self.avg_size_pick_batch.append(self.nOrders_hist / self.max_batchsize_ptg_gtp)

        self.reward_episode += self.reward_action
        return self.reward_action

    def get_reward_mo(self, action):
        # Compared to the traditional reward function, this needs to output a vector of results

        self.reward_action = [0, 0]
        feasibility = self.check_action(action, self.state_representation[:])
        tardy_orders = self.state_representation[:][-2] - self.tardy_order_hist
        self.tardy_order_hist = self.state_representation[:][-2]

        converted_action = self.action_to_action[action]

        # Action feasibility, this is for all actions
        if not feasibility:
            self.reward_action[0] += self.reward_structure['infeasible_action']
            self.reward_action[1] += self.reward_structure['infeasible_action']
            self.reward_distribution['infeasible_action'] += self.reward_structure['infeasible_action']

        # Reward for objective: minimizing the number of tardy orders
        if tardy_orders > 0:
            self.reward_action[0] += self.reward_structure['tardy_order'] * tardy_orders
            self.reward_distribution['tardy_order'] += self.reward_structure['tardy_order'] * tardy_orders

        # Reward for objective: minimizing order picking time
        if converted_action in self.actions_pick_by_batch:
            self.reward_action[1] += self.reward_structure['batch_action']
            self.reward_distribution['batch_action'] += self.reward_structure['batch_action']
        if converted_action < 10:
            if converted_action in [0, 1, 4, 5]:  # PtG orders
                self.reward_action[1] -= (1 - self.nOrders_hist / self.max_batchsize_ptg)/100
            elif converted_action in [2, 3, 6, 7]:  # GtP orders
                self.reward_action[1] -= (1 - self.nOrders_hist / self.max_batchsize_gtp)/100
            elif converted_action in [8, 9]:  # PtG GtP orders
                self.reward_action[1] -= (1 - self.nOrders_hist / self.max_batchsize_ptg_gtp)/100

        # Save results on batch size etc.
        # The reward distribution dict contains all reward, however agent is only trained on reward based on the
        # selected objective
        if converted_action < 10:
            if converted_action in [0, 1, 4, 5]:  # PtG orders
                self.avg_batch_size.append(self.nOrders_hist / self.max_batchsize_ptg)
                self.picking_time.append((self.nItems_ptg_hist * self.PtG_picking_item.rvs() +
                                          self.PtG_picking_constant) / self.nOrders_hist)
                self.reward_distribution['batch_composition'] -= (1 - self.nOrders_hist / self.max_batchsize_ptg)/100
                if converted_action in self.actions_pick_by_batch:
                    self.avg_size_pick_batch.append(self.nOrders_hist / self.max_batchsize_ptg)
            elif converted_action in [2, 3, 6, 7]:  # GtP orders
                self.avg_batch_size.append(self.nOrders_hist / self.max_batchsize_gtp)
                self.picking_time.append((self.nItems_gtp_hist * self.GtP_picking_time) / self.nOrders_hist)
                self.reward_distribution['batch_composition'] -= (1 - self.nOrders_hist / self.max_batchsize_gtp)/100
                if converted_action in self.actions_pick_by_batch:
                    self.avg_size_pick_batch.append(self.nOrders_hist / self.max_batchsize_gtp)
            elif converted_action in [8, 9]:  # PtG GtP orders
                self.avg_batch_size.append(self.nOrders_hist / self.max_batchsize_ptg_gtp)
                self.picking_time.append((self.nItems_ptg_hist * self.PtG_picking_item.rvs() + self.PtG_picking_constant
                                         + self.nItems_gtp_hist * self.GtP_picking_time) / self.nOrders_hist)
                self.reward_distribution['batch_composition'] -= (1 - self.nOrders_hist / self.max_batchsize_ptg_gtp)/100
                if converted_action in self.actions_pick_by_batch:
                    self.avg_size_pick_batch.append(self.nOrders_hist / self.max_batchsize_ptg_gtp)

        return self.reward_action

    def check_termination(self):
        if self.res.finished_orders(self.finished_orders) >= self.nOrders:
            return True
        else:
            return False

    def episode_render(self):
        if len(self.picking_strategy) > 100:
            tardy_orders = self.state_representation[-2] / self.state_representation[-3]
            batch_decision = self.picking_strategy.count(1) / len(self.picking_strategy)
            batch_size = round(sum(self.avg_batch_size) / len(self.avg_batch_size), 3)
            picking_time = round(sum(self.picking_time) / len(self.picking_time), 3)
            batch_size_pick_batch = round(sum(self.avg_size_pick_batch) / len(self.avg_size_pick_batch), 3)
        else:
            tardy_orders = 0
            batch_decision = 0
            batch_size = 0
            picking_time = 0
            batch_size_pick_batch = 0

        print('Tardy orders:', round(tardy_orders, 4), 'Pick-by-batch: ', round(batch_decision, 3))
        print('Finish time: ', round(self.state_representation[-1] / 3600, 3))
        print('Episode reward: ', self.reward_episode)
        print('Batch size: ', batch_size)
        print('Batch size pick by batch: ', batch_size_pick_batch)
        print('Picking time: ', picking_time)
        self.reward_episode = 0

    def episode_render_test(self):
        if len(self.picking_strategy) > 100:
            tardy_orders = self.state_representation[-2] / self.state_representation[-3]
            batch_decision = self.picking_strategy.count(1) / len(self.picking_strategy)
            batch_size = round(sum(self.avg_batch_size) / len(self.avg_batch_size), 3)
            pick_time = round(sum(self.picking_time) / len(self.picking_time), 3)
            batch_size_batch = round(sum(self.avg_size_pick_batch) / len(self.avg_size_pick_batch), 3)
        else:
            tardy_orders = 0
            batch_decision = 0
            batch_size = 0
            pick_time = 0
            batch_size_batch = 0

        results = {'tardy_orders': round(tardy_orders, 4), 'pick_by_batch': round(batch_decision, 3),
                   'finish_time': round(self.state_representation[-1] / 3600, 3), 'episode_reward': self.reward_episode,
                   'batch_size': batch_size,
                   'picking_time': pick_time,
                   'batch_size_pick_batch': batch_size_batch}
        self.reward_episode = 0

        return results

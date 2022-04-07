import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class SimResults:
    def __init__(self, config):
      self.tardy_orders = 0
      self.tardy_orders_list = []
      self.tardy_orders_list_all = []
      self.time_register = []
      self.order_progress = []
      
      self.hist_ql_gtp = []
      self.hist_ql_ptg = []
      self.hist_ql_pack = []
      self.hist_ql_dto = []
      self.hist_ql_sto = []
      
      self.availability_ptg = []
      self.availability_gtp = []
      self.availability_pack = []
      self.availability_dto = []
      self.availability_sto = []
      self.availability_time = []
      
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
      
      # Available resources
      self.PtG_picker_available_init = config['simulation']['nPtG_pickers']
      self.GtP_shuttle_available_init = config['simulation']['nGtP_shuttles']
      self.DtO_operator_available_init = config['simulation']['nDtO_operators']
      self.StO_operator_available_init = config['simulation']['nStO_operators']
      
      # Start hours shifts
      self.start_shifts = config['simulation']['shift_start']
      
      self.avg_batch_size_sio = []
      self.avg_batch_size_mio = []
      self.time_batch_size_sio = []
      self.time_batch_size_mio = []
      
      self.pick_by_order_ptg = []
      self.order_progress_individual = []
      
      self.actions_pick_by_batch = config['simulation']['actions_pick_by_batch']
      
    def report_tardiness(self, order, current_t):
        if order.cutoff_time < current_t:
            tardiness = True
            self.tardy_orders_list.append(order)
        else: 
            tardiness = False
        self.tardy_orders += tardiness * order.nOrders     
        
    def finished_orders(self, finished_orders):
        counter = 0
        for i in finished_orders:
            counter += i.nOrders
        return counter
      
    def register_QL(self, ptg, gtp, pack, dto, sto):
        self.hist_ql_gtp.append(ptg)
        self.hist_ql_ptg.append(gtp)
        self.hist_ql_pack.append(pack)
        self.hist_ql_dto.append(dto)
        self.hist_ql_sto.append(sto)
        
    def register_resources(self, ptg, gtp, dto, sto, time):
        self.availability_ptg.append(ptg)
        self.availability_gtp.append(gtp)
        self.availability_dto.append(dto)
        self.availability_sto.append(sto)
        self.availability_time.append(time)
        
    def register_time(self, time):
        self.time_register.append(time)
        
    def compute_picking_time_total(self, finished_orders):
        total_picking_time = []
        for order in finished_orders:
            total_time = order.System_out - order.arr_time
            total_picking_time.append(total_time / order.nOrders)
        return total_picking_time

    def order_progress_list(self, finished_orders):
        progress = [[0,0,0,0,0]]
        
        time = []

        for idx, i in enumerate(finished_orders):
            temp = []
            if i.category in [0,1,2]:
                temp = 0
            elif i.category in [3,4,5]:
                temp = 1
            elif i.category in [6,7,8]:
                temp = 2
            elif i.category in [9,10,11]:
                temp = 3
            elif i.category in [12,13,14]:
                temp = 4
        
            temp_list = progress[-1][:]
            temp_list[temp] += i.nOrders
            progress.append(temp_list)
            
            time.append(i.System_out)
            
        progress.remove([0,0,0,0,0])
        return np.array(progress), time
       
        
    def register_order_progress(self, order_categories):
        order_progress_temp = [sum([len(order_categories[0]), len(order_categories[1]),len(order_categories[2])]),
            sum([len(order_categories[3]), len(order_categories[4]),len(order_categories[5])]),
            sum([len(order_categories[6]), len(order_categories[7]),len(order_categories[8])]),
            sum([len(order_categories[9]), len(order_categories[10]),len(order_categories[11])]),
            sum([len(order_categories[12]), len(order_categories[13]),len(order_categories[14])])]
        self.order_progress.append(order_progress_temp)
        
        order_progress_temp = [len(order_categories[0]), len(order_categories[1]),len(order_categories[2]),
                               len(order_categories[12]), len(order_categories[13]),len(order_categories[14])]
        self.order_progress_individual.append(order_progress_temp)
        
    def plot_ind_order(self):
        self.order_progress_individual = np.array(self.order_progress_individual)
        
        fig, axs = plt.subplots(2, 1, figsize=(14,6))
        fig.suptitle('Orders in order categories', fontsize=16)
        axs[0].plot(self.time_register, self.order_progress_individual[:,0], label = 'sio_ptg_e1')
        axs[0].plot(self.time_register, self.order_progress_individual[:,1], label = 'sio_ptg_e2')
        axs[0].plot(self.time_register, self.order_progress_individual[:,2], label = 'sio_ptg_e3')
        axs[0].legend()
        axs[0].set_xlabel('time in seconds')
        axs[0].set_ylabel('Number of orders in category')
        
        axs[1].plot(self.time_register, self.order_progress_individual[:,3], label = 'mio_ptg_gtp_e1')
        axs[1].plot(self.time_register, self.order_progress_individual[:,4], label = 'mio_ptg_gtp_e2')
        axs[1].plot(self.time_register, self.order_progress_individual[:,5], label = 'mio_ptg_gtp_e3')
        axs[1].legend()
        axs[1].set_xlabel('time in seconds')
        axs[1].set_ylabel('Number of orders in category')
        plt.show()
        

    def register_batch_size(self, action, picking_items, new_t):
        if action <= 11:
            self.avg_batch_size_sio.append(len(picking_items))
            self.time_batch_size_sio.append(new_t)
        elif action > 11:
            self.avg_batch_size_mio.append(len(picking_items))
            self.time_batch_size_mio.append(new_t)
    
    def plot_avg_batch_size(self):
        fig, axs = plt.subplots(1, 2, figsize=(14,6))
        fig.suptitle('Batch size for pick-by-batch ptg orders', fontsize=16)
        axs[0].bar(self.time_batch_size_sio, self.avg_batch_size_sio, width=10, label='SIO')
        axs[0].legend()
        axs[0].set_xlabel('time steps')
        axs[0].set_ylabel('Batch size')
        
        axs[1].bar(self.time_batch_size_mio, self.avg_batch_size_mio, width=10, label='MIO')
        axs[1].legend()
        axs[1].set_xlabel('time steps')
        axs[1].set_ylabel('Batch size')
        plt.show()
        
    def plot_waiting_time(self, finished_orders):
        wait_ptg = []
        wait_gtp = []
        wait_pack = []
        wait_dto = []
        wait_sto = []
        
        for i in finished_orders:
            if i.route == 1:
                wait_ptg.append(i.PtG_in - i.arr_time)
            elif i.route == 2:
                wait_ptg.append(i.PtG_in - i.arr_time)
                wait_pack.append(i.Pack_in - i.PtG_out)
            elif i.route == 3:
                wait_ptg.append(i.PtG_in - i.arr_time)
                wait_gtp.append(i.GtP_in - i.PtG_out)
                wait_sto.append(i.StO_in - i.GtP_out)
                wait_pack.append(i.Pack_in - i.StO_out)
            elif i.route == 4:
                wait_gtp.append(i.GtP_in - i.arr_time)
                wait_sto.append(i.StO_in - i.GtP_out)
                wait_pack.append(i.Pack_in - i.StO_out)
            elif i.route == 5:
                wait_gtp.append(i.GtP_in - i.arr_time)
                wait_dto.append(i.DtO_in - i.GtP_out)
                
        fig, axs = plt.subplots(5, 1, figsize=(12,12))
        fig.suptitle('Waiting times in front of workstations', fontsize=16)
        axs[0].plot(wait_ptg, label='PtG')
        axs[0].legend()
        axs[0].set_xlabel('Order number')
        axs[0].set_ylabel('Waiting time (sec)')
        axs[0].grid(True)
        
        axs[1].plot(wait_gtp, label='GtP')
        axs[1].legend()
        axs[1].set_xlabel('Order number')
        axs[1].set_ylabel('Waiting time (sec)')
        axs[1].grid(True)
        
        axs[2].plot(wait_pack, label='Pack')
        axs[2].legend()
        axs[2].set_xlabel('Order number')
        axs[2].set_ylabel('Waiting time (sec)')
        axs[2].grid(True)
        
        axs[3].plot(wait_dto, label='DtO')
        axs[3].legend()
        axs[3].set_xlabel('Order number')
        axs[3].set_ylabel('Waiting time (sec)')
        axs[3].grid(True)
        
        axs[4].plot(wait_sto, label='StO')
        axs[4].legend()
        axs[4].set_xlabel('Order number')
        axs[4].set_ylabel('Waiting time (sec)')
        axs[4].grid(True)
        plt.show()
        
    def print_resource_utilization(self):
        u_ptg = []
        u_gtp = []
        u_dto = []
        u_sto = []
        for i in range(len(self.availability_ptg)):
            u_ptg.append(self.availability_ptg[i] / self.PtG_picker_available_init[0])
            u_gtp.append(self.availability_gtp[i] / self.GtP_shuttle_available_init[0])
            u_dto.append(self.availability_dto[i] / self.DtO_operator_available_init[0])
            u_sto.append(self.availability_sto[i] / self.StO_operator_available_init[0])
        
        print('Utilization PtG :', np.mean(u_ptg))
        print('Utilization GtP :', np.mean(u_gtp))
        print('Utilization DtO :', np.mean(u_dto))
        print('Utilization StO :', np.mean(u_sto))
             
        
    def plot_resource_utilization(self):
        # resource utilization averaged per minute
        u_ptg = []
        u_gtp = []
        u_dto = []
        u_sto = []
        
        aggregation = 100
        agg_counter = 0
        u_temp = {'PtG':0, 'GtP':0, 'DtO':0, 'StO':0}
        for i in range(len(self.availability_ptg)):
            # if self.availability_time[i] < self.start_shifts[1] * 3600:
            #     u_temp['PtG'] += (1- (self.availability_ptg[i] / self.PtG_picker_available_init[0]))
            #     u_temp['GtP'] += (1- (self.availability_gtp[i] / self.GtP_shuttle_available_init[0]))
    
            #     u_temp['DtO'] += (1- (self.availability_dto[i] / self.DtO_operator_available_init[0]))
            #     u_temp['StO'] += (1- (self.availability_sto[i] / self.StO_operator_available_init[0]))
            #     agg_counter += 1
                
            # elif self.availability_time[i] >= self.start_shifts[1] * 3600:
            #     u_temp['PtG'] += (1- (self.availability_ptg[i] / self.PtG_picker_available_init[1]))
            #     u_temp['GtP'] += (1- (self.availability_gtp[i] / self.GtP_shuttle_available_init[1]))
    
            #     u_temp['DtO'] += (1- (self.availability_dto[i] / self.DtO_operator_available_init[1]))
            #     u_temp['StO'] += (1- (self.availability_sto[i] / self.StO_operator_available_init[1]))
            #     agg_counter += 1
            
            u_temp['PtG'] += (1- (self.availability_ptg[i] / self.PtG_picker_available_init[0]))
            u_temp['GtP'] += (1- (self.availability_gtp[i] / self.GtP_shuttle_available_init[0]))

            u_temp['DtO'] += (1- (self.availability_dto[i] / self.DtO_operator_available_init[0]))
            u_temp['StO'] += (1- (self.availability_sto[i] / self.StO_operator_available_init[0]))
            agg_counter += 1
                
            if agg_counter == aggregation:
                u_ptg.append(u_temp['PtG'] / aggregation) 
                u_gtp.append(u_temp['GtP'] / aggregation) 
                u_dto.append(u_temp['DtO'] / aggregation) 
                u_sto.append(u_temp['StO'] / aggregation) 
                agg_counter = 0
                u_temp = {'PtG':0, 'GtP':0, 'DtO':0, 'StO':0}

        fig, axs = plt.subplots(4, 1, figsize=(12,12))
        
        axs[0].plot(u_ptg, label='PtG')
        axs[0].legend()
        axs[0].set_ylabel('Utilization %')
        axs[0].grid(True)
        axs[0].set_ylim([0,1])
        fig.suptitle('Resource utilization (smoothed)', fontsize=16)
        
        axs[1].plot(u_gtp, label='GtP')
        axs[1].legend()
        axs[1].set_ylabel('Utilization %')
        axs[1].grid(True)
        axs[1].set_ylim([0,1])
        
        axs[2].plot(u_dto, label='DtO')
        axs[2].legend()
        axs[2].set_ylabel('Utilization %')
        axs[2].grid(True)
        axs[2].set_ylim([0,1])
        
        axs[3].plot(u_sto, label='StO')
        axs[3].legend()
        axs[3].set_xlabel('time steps')
        axs[3].set_ylabel('Utilization %')
        axs[3].grid(True)
        axs[3].set_ylim([0,1])
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
        
                
    def plot_resources(self):
        fig, axs = plt.subplots(5, 1, figsize=(12,12))
        fig.suptitle('Availabel resources over time', fontsize=16)
        axs[0].plot(self.time_register, self.availability_ptg, label='PtG')
        axs[0].legend()
        axs[0].set_xlabel('time in seconds')
        axs[0].set_ylabel('Available PtG pickers')
        axs[0].grid(True)
        
        axs[1].plot(self.time_register, self.availability_gtp, label='GtP')
        axs[1].legend()
        axs[1].set_xlabel('time in seconds')
        axs[1].set_ylabel('Available GtP pickers')
        axs[1].grid(True)
        
        axs[2].plot(self.time_register, self.availability_pack, label='Pack')
        axs[2].legend()
        axs[2].set_xlabel('time in seconds')
        axs[2].set_ylabel('Available Pack operators')
        axs[2].grid(True)
        
        axs[3].plot(self.time_register, self.availability_dto, label='DtO')
        axs[3].legend()
        axs[3].set_xlabel('time in seconds')
        axs[3].set_ylabel('Available DtO operators')
        axs[3].grid(True)
        
        axs[4].plot(self.time_register, self.availability_sto, label='StO')
        axs[4].legend()
        axs[4].set_xlabel('time in seconds')
        axs[4].set_ylabel('Available StO operators')
        axs[4].grid(True)
        
    def plot_QL(self, finished_orders):
        fig, axs = plt.subplots(5, 1, figsize=(12,12))
        fig.suptitle('Queue length', fontsize=16)
        axs[0].plot(self.time_register, self.hist_ql_ptg, label='PtG')
        axs[0].legend()
        axs[0].set_xlabel('time in seconds')
        axs[0].set_ylabel('Queue length')
        axs[0].grid(True)
        
        axs[1].plot(self.time_register, self.hist_ql_gtp, label='GtP')
        axs[1].legend()
        axs[1].set_xlabel('time in seconds')
        axs[1].set_ylabel('Queue length')
        axs[1].grid(True)
        
        axs[2].plot(self.time_register, self.hist_ql_dto, label='DtO')
        axs[2].legend()
        axs[2].set_xlabel('time in seconds')
        axs[2].set_ylabel('Queue length')
        axs[2].grid(True)
        
        axs[3].plot(self.time_register, self.hist_ql_sto, label='StO')
        axs[3].legend()
        axs[3].set_xlabel('time in seconds')
        axs[3].set_ylabel('Queue length')
        axs[3].grid(True)
        
        progress, time = self.order_progress_list(finished_orders)
        axs[4].plot(time, progress[:,0], label = 'sio_ptg')
        axs[4].plot(time, progress[:,1], label = 'sio_gtp')
        axs[4].plot(time, progress[:,2], label = 'mio_ptg')
        axs[4].plot(time, progress[:,3], label = 'mio_gtp')
        axs[4].plot(time, progress[:,4], label = 'mio_ptg_gtp')
        axs[4].legend()
        axs[4].set_xlabel('time in seconds')
        axs[4].set_ylabel('Number of orders finished')
        axs[4].grid(True)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
        
    def plot_order_progress(self, finished_orders, name):
        fig, axs = plt.subplots(1, 1, figsize=(12,12))
        fig.suptitle('Order progress per category. Model: '+name, fontsize=16)
        progress, time = self.order_progress_list(finished_orders)
        time_transformed = []
        for i in range(len(time)):
            time_transformed.append(time[i]/3600)
        axs.plot(time_transformed, progress[:,0], label = 'sio_ptg')
        axs.plot(time_transformed, progress[:,1], label = 'sio_gtp')
        axs.plot(time_transformed, progress[:,2], label = 'mio_ptg')
        axs.plot(time_transformed, progress[:,3], label = 'mio_gtp')
        axs.plot(time_transformed, progress[:,4], label = 'mio_ptg_gtp')
        axs.legend()
        axs.set_xlabel('Time in hours')
        axs.set_ylabel('Number of orders finished')
        axs.grid(True)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig(r"C:\Users\nlmbeeks\Desktop\order_progress"+name+".png")


        df = pd.DataFrame(progress)
        df_time = pd.DataFrame(time_transformed)
        df.to_csv(r"C:\Users\nlmbeeks\Desktop\order_progress"+name+".csv")
        df_time.to_csv(r"C:\Users\nlmbeeks\Desktop\order_progress_time" + name + ".csv")
        
    def plot_tardy_orders(self):
        category_name = {0: 'sio_ptg', 1: 'sio_ptg', 2:'sio_ptg', 
                         3: 'sio_gtp', 4: 'sio_gtp', 5:'sio_gtp',
                         6: 'mio_ptg', 7: 'mio_ptg', 8:'mio_ptg',
                         9: 'mio_gtp', 10: 'mio_gtp', 11:'mio_gtp',
                         12: 'mio_ptg_gtp', 13: 'mio_ptg_gtp', 14:'mio_ptg_gtp'}
        tardy_order_categories = []
        for order in self.tardy_orders_list:
            for i in range(order.nOrders):
                tardy_order_categories.append(category_name[order.category])
        
        plt.hist(tardy_order_categories)
        plt.show()

    def plot_cutoff_moments(self, finished_orders):
        tardy_orders = []
        cutoff_times = []
        for order in finished_orders:
            if order.cutoff_time < order.System_out:
                tardy_orders.append(order)
                if order.cutoff_time not in cutoff_times:
                    cutoff_times.append(order.cutoff_time)
                
        graph_data_sio_ptg = []
        graph_data_sio_gtp = []
        graph_data_mio_ptg = []
        graph_data_mio_gtp = []
        graph_data_mio_ptg_gtp = []
        
        for t_order in tardy_orders:
            if t_order.category in [0,1,2]:
                graph_data_sio_ptg.append([t_order.cutoff_time, t_order.System_out])
            elif t_order.category in [3,4,5]:
                graph_data_sio_gtp.append([t_order.cutoff_time, t_order.System_out])
            elif t_order.category in [6,7,8]:
                graph_data_mio_ptg.append([t_order.cutoff_time, t_order.System_out])
            elif t_order.category in [9,10,11]:
                graph_data_mio_gtp.append([t_order.cutoff_time, t_order.System_out])
            elif t_order.category in [12,13,14]:
                graph_data_mio_ptg_gtp.append([t_order.cutoff_time, t_order.System_out])
           
        graph_data_sio_ptg_arr = np.array(graph_data_sio_ptg)
        graph_data_sio_gtp_arr = np.array(graph_data_sio_gtp)
        graph_data_mio_ptg_arr = np.array(graph_data_mio_ptg)
        graph_data_mio_gtp_arr = np.array(graph_data_mio_gtp)
        graph_data_mio_ptg_gtp_arr = np.array(graph_data_mio_ptg_gtp)
        
        
        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        fig.suptitle("Tardy orders per cutoff time", fontsize=16)
        
        if len(graph_data_sio_ptg_arr) > 0:
            axs.scatter(graph_data_sio_ptg_arr[:,1], graph_data_sio_ptg_arr[:,0], label = 'sio ptg')
        if len(graph_data_sio_gtp_arr) > 0:
            axs.scatter(graph_data_sio_gtp_arr[:,1], graph_data_sio_gtp_arr[:,0], label = 'sio gtp')
        if len(graph_data_mio_ptg_arr) > 0:
            axs.scatter(graph_data_mio_ptg_arr[:,1], graph_data_mio_ptg_arr[:,0], label = 'mio ptg')
        if len(graph_data_mio_gtp_arr) > 0:
            axs.scatter(graph_data_mio_gtp_arr[:,1], graph_data_mio_gtp_arr[:,0], label = 'mio gtp')
        if len(graph_data_mio_ptg_gtp_arr) > 0:
            axs.scatter(graph_data_mio_ptg_gtp_arr[:,1], graph_data_mio_ptg_gtp_arr[:,0], label = 'mio ptg gtp')
        
        for line in cutoff_times:
            axs.vlines(line, ymin=line-500, ymax=line+500, color='r')
        axs.legend()
        axs.set_xlabel('Finish time of order')
        axs.set_ylabel('Cutoff time')
        axs.grid(True)
        plt.show()

        df_sio_ptg = pd.DataFrame({'sio_ptg': graph_data_sio_ptg})
        df_sio_gtp = pd.DataFrame({'sio_gtp': graph_data_sio_gtp})
        df_mio_ptg = pd.DataFrame({'mio_ptg': graph_data_mio_ptg})
        df_mio_gtp = pd.DataFrame({'mio_gtp': graph_data_mio_gtp})
        df_mio_ptg_gtp = pd.DataFrame({'mio_ptg_gtp': graph_data_mio_ptg_gtp})

        df_sio_ptg.to_csv(r'C:\Users\nlmbeeks\Desktop\exD_tardy_orders_sio_ptg.csv')
        df_sio_gtp.to_csv(r'C:\Users\nlmbeeks\Desktop\exD_tardy_orders_sio_gtp.csv')
        df_mio_ptg.to_csv(r'C:\Users\nlmbeeks\Desktop\exD_tardy_orders_mio_ptg.csv')
        df_mio_gtp.to_csv(r'C:\Users\nlmbeeks\Desktop\exD_tardy_orders_mio_gtp.csv')
        df_mio_ptg_gtp.to_csv(r'C:\Users\nlmbeeks\Desktop\exD_tardy_orders_mio_ptg_gtp.csv')
                    
    def plot_avg_picking_times(self, finished_orders):
        
        order_list = []
        time_list = []
        for order in finished_orders:
            order_list.append((order.System_out - order.arr_time) / order.nOrders)
            time_list.append(order.System_out)
            
        fig, axs = plt.subplots(1, 1, figsize=(6,6))
        fig.suptitle("Relative picking times per order", fontsize=16)
        axs.plot(time_list, order_list)
        axs.set_xlabel('Time in seconds')
        axs.set_ylabel('Picking time in seconds')
        axs.grid(True)
        plt.show()
        
        df = pd.DataFrame({'picking_time':order_list, 'time': time_list})
        df.to_csv(r'C:\Users\nlmbeeks\Desktop\picking_times.csv')

    def save_picking_times_storage_area(self, finished_orders):
        picking_time_list_ptg = []
        time_list_ptg = []
        picking_time_list_gtp = []
        time_list_gtp = []

        for order in finished_orders:
            if order.route in [1, 2, 3, 6]:
                picking_time_list_ptg.append((order.PtG_out - order.arr_time ) / order.nOrders)
                time_list_ptg.append(order.System_out)
            else:
                picking_time_list_gtp.append((order.GtP_out - order.arr_time) / order.nOrders)
                time_list_gtp.append(order.System_out)

        df_ptg = pd.DataFrame({'picking_time': picking_time_list_ptg, 'time': time_list_ptg})
        df_ptg.to_csv(r'C:\Users\nlmbeeks\Desktop\exD_picking_times_ptg.csv')

        df_gtp = pd.DataFrame({'picking_time': picking_time_list_gtp, 'time': time_list_gtp})
        df_gtp.to_csv(r'C:\Users\nlmbeeks\Desktop\exD_picking_times_gtp.csv')

    def save_action(self, action_list):
        action_list_ptg = []
        time_list_ptg = []
        action_list_gtp = []
        time_list_gtp = []

        for action, time in action_list:
            if action in [0, 4, 9]:
                action_list_ptg.append(0)
                time_list_ptg.append(time)
            elif action in [1,5,8]:
                action_list_ptg.append(1)
                time_list_ptg.append(time)

            if action in [2, 6]:
                action_list_gtp.append(0)
                time_list_gtp.append(time)
            elif action in [3, 7]:
                action_list_gtp.append(1)
                time_list_gtp.append(time)

        df_ptg = pd.DataFrame({'action_list': action_list_ptg, 'time': time_list_ptg})
        df_ptg.to_csv(r'C:\Users\nlmbeeks\Desktop\exC_actions_ptg_BOC.csv')

        df_gtp = pd.DataFrame({'action_list': action_list_gtp, 'time': time_list_gtp})
        df_gtp.to_csv(r'C:\Users\nlmbeeks\Desktop\exC_actions_gtp_BOC.csv')


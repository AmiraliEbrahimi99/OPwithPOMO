##########################################################################################
# Path Config

import os
import sys
from itertools import product

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


import torch

import os
from logging import getLogger

from OPHSEnv import OPHSEnv as Env
from OPHSModel import OPHSModel as Model

from utils.utils import *

##########################################################################################
# Machine Environment Config

DEBUG_MODE = False 
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0




##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from OPHSTester import OPHSTester as Tester


##########################################################################################
# parameters

env_params = {
    'problem_size': 30,
    'pomo_size': 30,
    'hotel_size': 4,
    'day_number': 3
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}


tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/train_ophs_n30_with_instNorm_100_epoch_static_order',  # directory path of pre-trained model and log files saved.
        'epoch': 100,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10*1000,
    'test_batch_size': 1000,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 400,
    'test_data_load': {
        'enable': True,
        'filename': './T1-65-2-3.pt',
        'hotel_swap': True
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_ophs30_static',
        'filename': 'log.txt'
    }
}


class OPTester:
    def __init__(self,
                 env_params: dict,
                 model_params: dict,
                 tester_params: dict):
        self.step_count = 0 
        self.path = {}
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)
        
        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device, hotel_swap= self.tester_params['test_data_load']['hotel_swap'], order= hotel_order)
        
        
        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
    
    def run(self, batch_size : int = 1):
        self.reward = None
        done = False
        self.path = {}
        self.step_count = 0
        
        
        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1 
        
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            self.reset_state, _, _ = self.env.reset()
            self.model.pre_forward(self.reset_state)
        
        ###############################################
        state, self.reward, done = self.env.pre_step()

        while not done:
            self.step_count += 1 
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, self.reward, self.prize_per_day, done = self.env.step(selected)     
            self.path[self.step_count] = selected

  
        # print(f'\nwhole reawrds are {self.reward}\n')
       
        pomo = torch.argmax(self.reward)
        # print(f'the best pomo is :{pomo}')

        batch = 0
        self.plot_path=[] 
        for i in self.path:
            self.plot_path.append(int(self.path[i][batch][pomo]))
        
        # print(f'this is the path for batch: {batch}, pomo:{pomo} : {self.plot_path} with reward: {self.reward[batch,pomo]}\n')
        plot_paths[hotel_order] = self.plot_path
        best_rewards[hotel_order] = self.reward[batch, pomo]

        # Display the prize amount accumulated between each checkpoint
        # for i, difference in enumerate(self.prize_per_day):
        #     print(f"Collected prize between hotel {i} and hotel {i+1}:\n{difference}")

        # best_pomo = [prize[0, pomo] for prize in self.prize_per_day]
        # # Display the collected values for best pomo at each checkpoint
        # for i, value in enumerate(best_pomo):
        #     print(f"Collected prize between hotel {i} and {i+1} for pomo={pomo}: {value}")

    def plot(self,batch : int = 0 , pomo : int = 0 , best_result : bool = False) :
        # print(f'\nwhole reawrds are {self.reward}\n')
        # if best_result: 
        #     pomo = torch.argmax(self.reward)
        #     print(f'the best pomo is :{pomo}')

        # self.plot_path=[] 
        # for i in self.path:
        #     self.plot_path.append(int(self.path[i][batch][pomo]))
        # print(f'this is the path for batch: {batch}, pomo:{pomo} : {self.plot_path} with reward: {self.reward[batch,pomo]}\n')

        self.plot_depot1 = self.reset_state.depot_xy[batch][0].tolist()
        self.plot_depot2 = self.reset_state.depot_xy[batch][1].tolist()  # second depot coordinates
        self.plot_depot3 = self.reset_state.depot_xy[batch][2].tolist()  # third depot coordinates
        self.plot_depot4 = self.reset_state.depot_xy[batch][3].tolist()  # fourth depot coordinates
        self.plot_nodes = self.reset_state.node_xy[batch].tolist()
        self.plot_prize = self.reset_state.node_prize[batch].tolist()
        self.plot_size = [i * 100 for i in self.plot_prize]
        self.plot_nodes = np.array(self.plot_nodes)

        # Plot the nodes
        plt.scatter(self.plot_nodes[:, 0], self.plot_nodes[:, 1], color='Gray', s = self.plot_size )
        for i, node in enumerate(self.plot_nodes):
            plt.text(node[0], node[1], str(i + 4), fontsize=12, ha='center', va='center')

        # Plot the two depots with different colors
        plt.scatter(self.plot_depot1[0], self.plot_depot1[1], color='green')
        plt.text(self.plot_depot1[0], self.plot_depot1[1], 'Hotel 1', fontsize=12, ha='center', va='bottom')
        plt.scatter(self.plot_depot2[0], self.plot_depot2[1], color='green')  # second depot
        plt.text(self.plot_depot2[0], self.plot_depot2[1], 'Hotel 2', fontsize=12, ha='center', va='bottom')
        plt.scatter(self.plot_depot3[0], self.plot_depot3[1], color='green')  # third depot
        plt.text(self.plot_depot3[0], self.plot_depot3[1], 'Hotel 3', fontsize=12, ha='center', va='bottom')
        plt.scatter(self.plot_depot4[0], self.plot_depot4[1], color='green')  # fourth depot
        plt.text(self.plot_depot4[0], self.plot_depot4[1], 'Hotel 4', fontsize=12, ha='center', va='bottom')


        # plot arrows
        self.plot_nodes = self.reset_state.node_xy[batch].tolist()
        self.plot_depot = self.reset_state.depot_xy[batch].tolist()
        self.plot_nodes_depot = self.plot_depot + self.plot_nodes
        
        # for i in range(len(self.plot_path) - 1):
        plot_list = plot_paths[best_hotel_order]
        for i in range(len(plot_list) - 1):

            start = self.plot_nodes_depot[plot_list[i] ]
            end = self.plot_nodes_depot[plot_list[i + 1] ]
            if start == end :
                continue
            
            plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=0.05, head_length=0.02,
                        fc='white', ec='red')

        plt.title('Orienteering Problem')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)
        plt.axis('equal')
        # plt.show()       

        plt.savefig('my_plot.png', dpi=300)




if __name__ == '__main__' : 
    
    day_numb = env_params['day_number'] # Number of days
    best_rewards = {}
    plot_paths = {}

    # hotel_order = 12
    # self = OPTester(env_params=env_params, model_params=model_params, tester_params=tester_params)
    # self.run(batch_size=1)

    for hotel_order in range((4*4)):
        self = OPTester(env_params=env_params, model_params=model_params, tester_params=tester_params)
        self.run(batch_size=1)
        
    best_hotel_order = max(best_rewards, key=best_rewards.get)
    best_reward = best_rewards[best_hotel_order]

    points = [0, 1, 2, 3]  # points 0, 1, 2, 3
    all_states_list = [[0, i, j, 1] for i in points for j in points]
    best_state = all_states_list[best_hotel_order]

    print("\nBest rewards for each hotel order:")
    for hotel_order, reward in best_rewards.items():
        print(f"Hotel order {hotel_order}: {all_states_list[hotel_order]} reward: {reward}")
    
    # Print the corresponding plot_path
    print(f"\nBest hotel order is: {best_hotel_order} with hotel {best_state} visited and reward = {best_reward}")
    print(f"Plot path for best hotel order: {plot_paths[best_hotel_order]}")


    self.plot(batch=0,best_result= True) 
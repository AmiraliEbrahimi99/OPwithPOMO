##########################################################################################
# Path Config

import os
import sys

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
    'problem_size': 15,
    'pomo_size': 15,
    'hotel_size': 7,
    'day_number': 3,
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
        'path': './result/ophs_H7D3_510',  # directory path of pre-trained model and log files saved.
        'epoch': 510,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10*1000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 400,
    'test_data_load': {
        'enable': True,
        'filename': './T1-65-5-3.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_op20',
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
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)
        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
    
    def run(self, batch_size : int = 1):
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
            state, self.reward, done = self.env.step(selected)     
            self.path[self.step_count] = selected

    def plot(self,batch : int = 0 , pomo : int = 0 , best_result : bool = False) :
        print(f'\nwhole reawrds are {self.reward}\n')
        if best_result: 
            pomo = torch.argmax(self.reward)
            print(f'the best pomo is :{pomo}')

        self.plot_path=[] 
        for i in self.path:
            self.plot_path.append(int(self.path[i][batch][pomo]))
        
        print(f'this is the path for batch: {batch}, pomo:{pomo} : {self.plot_path} with reward: {self.reward[batch,pomo]}\n')
        self.plot_depot1 = self.reset_state.depot_xy[batch][0].tolist()
        self.plot_depot2 = self.reset_state.depot_xy[batch][1].tolist()  # second depot coordinates
        self.plot_depot3 = self.reset_state.depot_xy[batch][2].tolist()  # third depot coordinates
        self.plot_depot4 = self.reset_state.depot_xy[batch][3].tolist()  # fourth depot coordinates
        self.plot_depot5 = self.reset_state.depot_xy[batch][4].tolist()  # fifth depot coordinates
        self.plot_depot6 = self.reset_state.depot_xy[batch][5].tolist()  # sixth depot coordinates
        self.plot_depot7 = self.reset_state.depot_xy[batch][6].tolist()  # seventh depot coordinates
        self.plot_nodes = self.reset_state.node_xy[batch].tolist()
        self.plot_prize = self.reset_state.node_prize[batch].tolist()
        self.plot_size = [i * 100 for i in self.plot_prize]
        self.plot_nodes = np.array(self.plot_nodes)
        # print(f'prizes : {self.plot_prize}')
        # Plot the nodes
        plt.scatter(self.plot_nodes[:, 0], self.plot_nodes[:, 1], color='Gray', s = self.plot_size )
        for i, node in enumerate(self.plot_nodes):
            plt.text(node[0], node[1], str(i + 7), fontsize=12, ha='center', va='center')

        # Plot the two depots with different colors
        plt.scatter(self.plot_depot1[0], self.plot_depot1[1], color='green')
        plt.text(self.plot_depot1[0], self.plot_depot1[1], 'Hotel 1', fontsize=12, ha='center', va='bottom')
        plt.scatter(self.plot_depot2[0], self.plot_depot2[1], color='green')  # second depot
        plt.text(self.plot_depot2[0], self.plot_depot2[1], 'Hotel 2', fontsize=12, ha='center', va='bottom')
        plt.scatter(self.plot_depot3[0], self.plot_depot3[1], color='green')  # third depot
        plt.text(self.plot_depot3[0], self.plot_depot3[1], 'Hotel 3', fontsize=12, ha='center', va='bottom')
        plt.scatter(self.plot_depot4[0], self.plot_depot4[1], color='green')  # fourth depot
        plt.text(self.plot_depot4[0], self.plot_depot4[1], 'Hotel 4', fontsize=12, ha='center', va='bottom')
        plt.scatter(self.plot_depot5[0], self.plot_depot5[1], color='green')  # fifth depot
        plt.text(self.plot_depot5[0], self.plot_depot5[1], 'Hotel 5', fontsize=12, ha='center', va='bottom')
        plt.scatter(self.plot_depot6[0], self.plot_depot6[1], color='green')  # sixth depot
        plt.text(self.plot_depot6[0], self.plot_depot6[1], 'Hotel 6', fontsize=12, ha='center', va='bottom')
        plt.scatter(self.plot_depot7[0], self.plot_depot7[1], color='green')  # seventh depot
        plt.text(self.plot_depot7[0], self.plot_depot7[1], 'Hotel 7', fontsize=12, ha='center', va='bottom')


        # plot arrows
        self.plot_nodes = self.reset_state.node_xy[batch].tolist()
        self.plot_depot = self.reset_state.depot_xy[batch].tolist()
        self.plot_nodes_depot = self.plot_depot + self.plot_nodes
        for i in range(len(self.plot_path) - 1):

            start = self.plot_nodes_depot[self.plot_path[i] ]
            end = self.plot_nodes_depot[self.plot_path[i + 1] ]
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
    self = OPTester(env_params=env_params, model_params=model_params, tester_params=tester_params)
    self.run(batch_size=1)
    self.plot(batch=0,best_result= True)    

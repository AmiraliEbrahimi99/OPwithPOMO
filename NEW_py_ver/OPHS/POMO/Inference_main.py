##########################################################################################
import math, os, sys, time, torch, random, warnings, copy
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.backends.cuda.sdp_kernel.*")

# Path Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

from logging import getLogger
from OPHSEnv import OPHSEnv as Env
from OPHSModel import OPHSModel as Model
from utils.utils import *

##########################################################################################
# Machine Environment Config

DEBUG_MODE = True 
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# parameters
stochastic_prize = True

env_params = {
    'problem_size': 64,
    'pomo_size': 64,
    'hotel_size': 12,
    'day_number': 5,
    'stochastic_prize': stochastic_prize
}

model_params = {
    'stochastic_prize': stochastic_prize,
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
        'path': './result/ophssp_do_64',  # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch number of pre-trained model to laod.
    },
    'test_episodes': 10*1000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    # 'aug_factor': 16,
    'test_data_load': {
        'enable': True,
        'filename': '../../../Instances/OPHSSP_pt/66-125-10-5.pt',
    },
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
               
        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
    
    def run(self, batch_size: int = 1):
        self.reward = None
        done = False
        self.path = {}
        self.step_count = 0

        # Augmentation setup
        aug_factor = self.tester_params['aug_factor'] if self.tester_params['augmentation_enable'] else 1

        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, aug_factor)
            self.reset_state, _, _ = self.env.reset()
            self.model.pre_forward(self.reset_state)
        
        state, self.reward, done = self.env.pre_step()

        while not done:
            self.step_count += 1 
            selected, _ = self.model(state)
            state, self.reward, done = self.env.step(selected)
            self.path[self.step_count] = selected

#####################################################   FUNCTIONS   ###################################################################################
if __name__ == '__main__': 

    def parse_instance(instance_path, stochastic_prize: bool = False):
        
        def is_valid_line(line):
            parts = line.strip().split()
            expected_length = 4 if stochastic_prize else 3  
            if (len(parts) != expected_length):
                return False
            try:
                list(map(float, parts))  # Ensure all values are valid floats (handles negatives too)
                return True
            except ValueError:
                return False
            
        with open(instance_path, 'r') as file:
            n, h, day_number = map(int, file.readline().split())  # Read n, h, d
            file.readline()  # Skip unused line
            file.readline()  # Skip t_max line
            data = [list(map(float, line.split())) for line in file if is_valid_line(line)]

        if not stochastic_prize:
            x_coords, y_coords, scores = zip(*data) 
            scores = torch.tensor(scores, dtype=torch.float64)  
        else: 
            x_coords, y_coords, mean, variance = zip(*data)
            scores = torch.stack((torch.tensor(mean), torch.tensor(variance)))

        nodes_number = n + h  # Total nodes including hotels
        hotels_number = h + 2
        
        return scores 
    
    def RL_inference(node_scores, confidence_level: float = 0.95):

        self = OPTester(env_params=env_params, model_params=model_params, tester_params=tester_params)
        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(
                self.tester_params['test_data_load']['filename'], 
                self.device, 
            )

        aug_factor = self.tester_params['aug_factor'] if self.tester_params['augmentation_enable'] else 1

        self.run(batch_size=1)

        pomo = torch.argmax(self.reward, dim=1)

        best_score = -float('inf')  # Track the best collected score
        best_order = None  # Track the best order sequence

        for batch_idx in range(aug_factor):
            batch_order = [int(self.path[i][batch_idx][pomo[batch_idx]]) for i in self.path]

            complete_order = torch.tensor(batch_order, dtype=torch.int64)

            if stochastic_prize:
                mean_and_variance = node_scores[:, complete_order].sum(dim=1)  
                mean = mean_and_variance[0].item()
                std_dev = torch.sqrt(mean_and_variance[1]).item()
                collected_score = mean + norm.ppf(confidence_level) * std_dev                
            else:
                collected_score = node_scores[complete_order].sum().item()

            if collected_score > best_score:
                best_score = collected_score
                best_order = complete_order
            
        return best_order, best_score

#####################################################   Main loop   ###################################################################################
    
    def run_repeats_and_save(repeats, node_scores, output_file):
        results = []

        for factor in augmentation_factors:
            print(f"\nRunning for aug = {factor}\n")
            tester_params['aug_factor'] = factor

            for repeat in range(repeats):

                start_time = time.time()
                best_solution, best_score = RL_inference(node_scores)
                end_time = time.time()
                runtime = end_time - start_time

                # Save results for this repeat
                results.append({
                    "aug_factor": factor,
                    "Repeat": repeat + 1,
                    "Final_Score": best_score,
                    "Runtime": runtime,
                    "Best_solution": best_solution.tolist(),
                })

        df = pd.DataFrame(results)

        summary = df.groupby("aug_factor").agg(
            Mean_Final_Score=("Final_Score", "mean"),
            Max_Final_Score=("Final_Score", "max"),
            Min_Final_Score=("Final_Score", "min"),
            Mean_Runtime=("Runtime", "mean"),
            Max_Runtime=("Runtime", "max"),
            Min_Runtime=("Runtime", "min"),
        ).reset_index()

        best_solution_df = df.loc[df["Final_Score"].idxmax(), ["aug_factor", "Best_solution"]]
        summary["Final_Sequence"] = None
        summary.loc[0, "Final_Sequence"] = str(best_solution_df)  # Assign only to the first row

        # with pd.ExcelWriter(output_file) as writer:
        #     df.to_excel(writer, index=False, sheet_name="Raw Results")
        #     summary.to_excel(writer, index=False, sheet_name="Summary Statistics")

        # print(f"\nResults saved to {output_file}\n")
        print(df)
        print
        print(summary)

    ####################################### testing #############################################################################################

    #inputs
    instance_path = r"../../../Instances/raw_OPHSSP_instances/66-125-10-5.ophs"
    max_no_improve = 20
    repeats = 20
    augmentation_factors = [1, 8, 16]
    output_file = "output_results/66-125-10-5_no_rescaled_score.xlsx"

    scores = parse_instance(instance_path, stochastic_prize)
    final_hps = run_repeats_and_save(repeats, scores, output_file)

    # best_solution, best_score = RL_inference(scores)
    # print(type(best_solution))





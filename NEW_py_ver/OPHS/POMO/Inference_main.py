##########################################################################################
import os, sys, time, torch, warnings, glob
from pathlib import Path
import pandas as pd
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
stochastic_prize = False

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
        'path': './result/ophs_do_100',  # directory path of pre-trained model and log files saved.
        'epoch': 210,  # epoch number of pre-trained model to laod.
    },
    'test_episodes': 10*1000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    # 'aug_factor': 16,
    'test_data_load': {'enable': True, 'filename': '' },
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

def run_repeats(instance_name, repeats, node_scores):
    results = []

    for repeat in range(repeats):
        start_time = time.time()
        best_solution, best_score = RL_inference(node_scores)
        end_time = time.time()
        runtime = end_time - start_time

        results.append({
            "Instance": instance_name,
            "Repeat": repeat + 1,
            "Final_Score": best_score,
            "Runtime": runtime,
            "Best_solution": best_solution.tolist(),
        })

    return pd.DataFrame(results)

#####################################################   Main loop   ###################################################################################

#inputs
base_pt_path = "../../../Instances/OPHS_pt/*.pt"
base_ophs_path = "../../../Instances/raw_OPHS_instances/*.ophs"
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "OPHS_RL_traned_on_100_Aug16x.xlsx")

augmentation_factor = 16
repeats = 1

pt_instances = glob.glob(base_pt_path)
ophs_instances = glob.glob(base_ophs_path)
pt_instances = [os.path.normpath(p) for p in pt_instances]
ophs_instances = [os.path.normpath(p) for p in ophs_instances]

all_results = []

for index, pt_path in enumerate(pt_instances):
    instance_name = os.path.basename(pt_path).replace(".pt", "")
    problem_size, _, hotel_size, day_number = map(int, instance_name.split('-'))
    problem_size -= 2  # Adjust as per naming convention
    hotel_size += 2  # Adjust as per naming convention
    
    env_params = {
        'problem_size': problem_size,
        'pomo_size': problem_size,
        'hotel_size': hotel_size,
        'day_number': day_number,
        'stochastic_prize': stochastic_prize,
    }

    tester_params['aug_factor'] = augmentation_factor
    tester_params['test_data_load']['filename'] = pt_path

    # Get corresponding .ophs file
    ophs_path = os.path.normpath(pt_path.replace("OPHS_pt", "raw_OPHS_instances").replace(".pt", ".ophs"))      # replace names here too
    if ophs_path not in ophs_instances:
        print(f"Warning: No corresponding .ophs file found for {instance_name}")
        continue
    
    # Parse instance and run testing
    print(f"\nRunning for instance {index+1}/ {len(pt_instances)}, {instance_name}\n")
    node_scores = parse_instance(ophs_path, stochastic_prize)
    df_results = run_repeats(instance_name, repeats, node_scores)
    all_results.append(df_results)

# Combine results from all instances
final_results = pd.concat(all_results, ignore_index=True)

summary = final_results.groupby("Instance").agg(
    Mean_Final_Score=("Final_Score", "mean"),
    Max_Final_Score=("Final_Score", "max"),
    Min_Final_Score=("Final_Score", "min"),
    Mean_Runtime=("Runtime", "mean"),
    Max_Runtime=("Runtime", "max"),
    Min_Runtime=("Runtime", "min"),
).reset_index()

best_solution_df = final_results.loc[final_results.groupby("Instance")["Final_Score"].idxmax(), ["Instance", "Best_solution"]]
summary = summary.merge(best_solution_df, on="Instance", how="left")

with pd.ExcelWriter(output_file) as writer:
    final_results.to_excel(writer, index=False, sheet_name="Raw Results")
    summary.to_excel(writer, index=False, sheet_name="Summary Statistics")

print(f"\nResults saved to {output_file}\n")






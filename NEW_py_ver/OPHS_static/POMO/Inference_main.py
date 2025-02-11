##########################################################################################
import math, os, sys, time, torch, random, warnings, copy, glob
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
        'path': './result/ophs_so_32',  # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch number of pre-trained model to laod.
    },
    'test_episodes': 10*1000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    # 'aug_factor': 16,
    'test_data_load': {
        'enable': True,
        'filename': '',
        'hotel_swap': True
        # 'order': None  # Add 'order' to hold the hotel order
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

###################################################################################

def plot_trip(sequence, hotel_number, hotel_order, coordinates, scores):

    trip_number = len(hotel_order)  # Number of hotels
    sequence_mapped = [
        hotel_order[idx] if idx <= trip_number else idx for idx in sequence
    ]

    # Separate coordinates into hotels and nodes
    hotels_coords = coordinates[:hotel_number]  # First entries are hotels
    nodes_coords = coordinates[hotel_number:]  # Remaining entries are nodes

    # Extract node sizes based on scores (only for nodes)
    node_scores = scores[hotel_number:]  # Exclude hotel scores (first 7 entries)
    node_sizes = [score * 20 for score in node_scores]  # Scale for visualization

    # Plot nodes
    nodes_coords = np.array(nodes_coords)
    plt.scatter(nodes_coords[:, 0], nodes_coords[:, 1], color='gray', s=node_sizes)
    for idx, (x, y) in enumerate(nodes_coords):
        plt.text(x, y, str(trip_number + idx), fontsize=8, ha='center', va='center')  # Adjust index to match nodes

    # Plot hotels
    for idx, (x, y) in enumerate(hotels_coords):
        plt.scatter(x, y, color='green', marker='s', s=50)
        plt.text(x, y, f'Hotel {idx}', fontsize=8, ha='center', va='center')

    # Plot paths with arrows
    for i in range(len(sequence_mapped) - 1):
        start = coordinates[sequence_mapped[i]]
        end = coordinates[sequence_mapped[i + 1]]

        if not np.array_equal(start, end):  # Avoid drawing self-loops
            plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
                    head_width=0.4, head_length=0.4, fc='red', ec='red')

    # Plot settings
    plt.title('Orienteering Problem Solution')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('test_plot.png', dpi=300)
    # plt.show()

#####################################################   FUNCTIONS   ###################################################################################

def parse_instance(instance_path, stochastic_prize: bool = False, confidence_level: float = 0.95):
    
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
        t_max = float(file.readline().split()[0])  # max time
        file.readline()  # Skip t_max line
        data = [list(map(float, line.split())) for line in file if is_valid_line(line)]

    if not stochastic_prize:
        x_coords, y_coords, scores = zip(*data) 
        scores = torch.tensor(scores, dtype=torch.float64)  
        scores_for_hps = copy.deepcopy(scores)
    else: 
        x_coords, y_coords, mean, variance = zip(*data)
        scores = torch.stack((torch.tensor(mean), torch.tensor(variance)))
        # scores_for_hps = scores[0] + norm.ppf(confidence_level) * torch.sqrt((scores[1]))             # method confidence level
        scores_for_hps = copy.deepcopy(mean)                                                            # method mean 

    nodes_number = n + h  # Total nodes including hotels
    hotels_number = h + 2
    all_nodes_index = list(range(nodes_number))
    distance_matrix = squareform(pdist(np.column_stack((x_coords, y_coords))))
    hotel_nodes_index = all_nodes_index[:hotels_number]
    hps = np.zeros((hotels_number, hotels_number))  # hps = hotel_potential_score
    
    pair_list = []
    for i in hotel_nodes_index:
        for j in hotel_nodes_index:
            if j >= i:
                pair_list.append([i,j])

    for i in pair_list:
        a = i[0]
        b = i[1]
        for node_i in all_nodes_index:
            total_pair_distance = distance_matrix[node_i][a] + distance_matrix[node_i][b]
            if total_pair_distance <= t_max:
                hps[a,b] += scores_for_hps[node_i]
                if a != b: 
                    hps[b,a] += scores_for_hps[node_i]

    hps = torch.tensor(hps)  # Convert to tensor

    return hps, scores, hotels_number, day_number 

def order_to_sequence(order, h, n_day):
    sequence = [0]  # Start with the fixed start point
    current_order = order
    for _ in range(n_day - 1):
        point = current_order % h
        sequence.append(point)
        current_order //= h
    sequence.append(1)  # Add the fixed end point
    return sequence

def sequence_to_order(sequence, h):

    K = len(sequence) - 2  # Exclude start and end points
    order = 0
    for i in range(K):
        order += sequence[i + 1] * (h ** i)  # Use i + 1 to skip the start point (0)
    return order

def RL_inference(input_order, node_scores, n_days, confidence_level: float = 0.95):

    self = OPTester(env_params=env_params, model_params=model_params, tester_params=tester_params)

    if self.tester_params['test_data_load']['enable']:
        self.env.use_saved_problems(
            self.tester_params['test_data_load']['filename'], 
            self.device, 
            hotel_swap=self.tester_params['test_data_load']['hotel_swap'], 
            order=input_order  # Ensure order is passed here
        )

    aug_factor = self.tester_params['aug_factor'] if self.tester_params['augmentation_enable'] else 1

    self.run(batch_size=1)

    pomo = torch.argmax(self.reward, dim=1)

    best_score = -float('inf')  # Track the best collected score
    best_order = None  # Track the best order sequence

    for batch_idx in range(aug_factor):
        batch_order = [int(self.path[i][batch_idx][pomo[batch_idx]]) for i in self.path]

        cleaned_solution = []
        for value in batch_order:
            if value not in cleaned_solution:
                cleaned_solution.append(value)

        complete_order = torch.tensor(cleaned_solution, dtype=torch.int64)

        if stochastic_prize:
            mean_and_variance = node_scores[:, complete_order].sum(dim=1)  
            mean = mean_and_variance[0].item()
            std_dev = torch.sqrt(mean_and_variance[1]).item()
            collected_score = mean + norm.pdf(confidence_level) * std_dev                
        else:
            collected_score = node_scores[complete_order].sum().item()

        if collected_score > best_score:
            best_score = collected_score
            best_order = complete_order
        
    num_hotels = (node_scores[0] == 0 if stochastic_prize else node_scores == 0).nonzero().squeeze()[-1] + 1  # Number of hotels
    hotel_visits = (best_order < num_hotels).nonzero().squeeze()
    
    if hotel_visits.all() != 0:
        if not stochastic_prize:
            score_per_day = torch.tensor([
                node_scores[best_order[hotel_visits[i] + 1 : hotel_visits[i + 1]]].sum()
                for i in range(len(hotel_visits) - 1)
            ], dtype=torch.float32)
        else:
            score_per_day = torch.stack([
                node_scores[:, best_order[hotel_visits[i] + 1 : hotel_visits[i + 1]]].sum(dim=1) 
                for i in range(len(hotel_visits) - 1)
            ])
            mean_per_day = score_per_day[:, 0]
            std_dev_per_day = torch.sqrt(score_per_day[:, 1])
            score_per_day = mean_per_day + norm.pdf(confidence_level) * std_dev_per_day
    else:
        score_per_day = torch.zeros(n_days, dtype=torch.float32)
        score_per_day[n_days-1] = -1

    return best_order, best_score, score_per_day

def simulated_annealing(hps, n_days, initial_solution=None, T0=10000, T_min=0.01, alpha=0.99):

    def evaluate(solution):
        score = 0
        for i in range(len(solution) - 1):
            score += hps[solution[i], solution[i + 1]].item()
        return score

    # If no initial solution is provided, generate one randomly
    if initial_solution is None:
        initial_solution = [0] + [random.choice(range(hps.shape[0])) for _ in range(n_days - 1)] + [1]

    current_solution = initial_solution
    best_solution = current_solution[:]
    best_score = evaluate(current_solution)

    T = T0
    while T > T_min:
        # Generate a neighbor
        new_solution = current_solution[:]
        i, j = random.sample(range(1, n_days), 2)  # Pick two hotels to swap
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        new_score = evaluate(new_solution)

        # Accept the new solution or a worse one with a certain probability
        if new_score > best_score or random.random() < math.exp((new_score - best_score) / T):
            current_solution = new_solution
            if new_score > best_score:
                best_solution = new_solution
                best_score = new_score

        # Cool down the temperature
        T *= alpha

    return best_solution

def greedy_trip_with_exploration(hps, n_days, exploration_prob=0.2):
    current_hotel = 0
    trip = [current_hotel]
    total_score = 0

    for _ in range(n_days - 1):
        if np.random.rand() < exploration_prob:
            # Randomly explore among top candidates
            candidates = np.argsort(hps[current_hotel])[-3:]  # Top 3 candidates
            next_hotel = np.random.choice(candidates)
        else:
            # Greedy selection
            next_hotel = torch.argmax(hps[current_hotel]).item()
        
        total_score += hps[current_hotel][next_hotel]
        trip.append(next_hotel)
        current_hotel = next_hotel

    total_score += hps[current_hotel][1]
    trip.append(1)

    return trip

def greedy_trip(hps, n_days):
    current_hotel = 0
    trip = [current_hotel]
    total_score = 0

    for _ in range(n_days - 1):
        next_hotel = torch.argmax(hps[current_hotel]).item()  # Use .item() to get the Python int
        total_score += hps[current_hotel, next_hotel].item()  # Access tensor element and convert to Python float
        trip.append(next_hotel)
        current_hotel = next_hotel

    total_score += hps[current_hotel, 1].item()  # Add the score for returning to hotel 1
    trip.append(1)

    return trip

def greedy_trip_with_penalty(hps, n_days, penalty_factor=0.9):
    current_hotel = 0
    trip = [current_hotel]
    total_score = 0

    for _ in range(n_days - 1):
        scores = hps[current_hotel]
        next_hotel = torch.argmax(scores).item()

        total_score += hps[current_hotel, next_hotel].item()
        trip.append(next_hotel)
        current_hotel = next_hotel

        # Apply penalty to discourage reusing the same path
        hps[current_hotel] *= penalty_factor

    total_score += hps[current_hotel, 1].item()
    trip.append(1)

    return trip

def update_hps(hps, hotel_order, prize_per_day, updated_mask):
    zero_flag =  torch.all(prize_per_day[:-1] == 0) and prize_per_day[-1] == -1

    for day in range(len(prize_per_day)):

        from_hotel, to_hotel, prize = hotel_order[day], hotel_order[day + 1], prize_per_day[day]
        
        if zero_flag and prize == -1: 
            hps[from_hotel, to_hotel] = prize
            updated_mask[from_hotel, to_hotel] = True
            continue

        if not updated_mask[from_hotel, to_hotel]:  # First-time update
            hps[from_hotel, to_hotel] = prize
            hps[to_hotel, from_hotel] = prize
            updated_mask[from_hotel, to_hotel] = True
            updated_mask[to_hotel, from_hotel] = True
        else:  # Subsequent updates only if the prize is greater
            hps[from_hotel, to_hotel] = max(hps[from_hotel, to_hotel], prize)
            hps[to_hotel, from_hotel] = max(hps[to_hotel, from_hotel], prize)

    return hps, updated_mask

def run_repeats(instance_name, hps, hotel_size, n_days, repeats, scores):
    results = []

    # for factor in augmentation_factors:
    #     print(f"\nRunning for aug = {factor}\n")
    #     tester_params['aug_factor'] = factor

    for repeat in range(repeats):

        start_time = time.time()
        best_order, best_score, best_solution, final_hps, each_iter, iter_to_converge = optimize_trip(hps.clone(), hotel_size, n_days, scores)
        end_time = time.time()
        runtime = end_time - start_time

        sequence = order_to_sequence(best_order, hotel_size, n_days)
        mapping = {i: sequence[i] for i in range(len(sequence))}
        final_sequence = [mapping[value] if value in mapping else value for value in best_solution.tolist()]

        # Save results for this repeat
        results.append({
            # "aug_factor": factor,
            "Instance": instance_name,
            "Repeat": repeat + 1,
            "Final_Score": best_score,
            "Runtime": runtime,
            "Improvement_each_iter": each_iter,
            "iteration_to_converge":iter_to_converge,
            "Best_solution": final_sequence,
        })

    return pd.DataFrame(results)

def optimize_trip(hps, hotel_size, n_days, scores, max_no_improve=20):

    hotel_order = greedy_trip_with_exploration(hps, n_days)  
    best_order = sequence_to_order(hotel_order, hotel_size)  
    best_complete_solution, best_score, prize_per_day = RL_inference(best_order, scores, n_days)

    no_improve_count = 0
    updated_mask = torch.zeros_like(hps, dtype=torch.bool)  
    scores_history = [best_score] 
    improvement_each_iter = []

    while no_improve_count < max_no_improve:

        previous_hps = hps.clone()
        hps, updated_mask = update_hps(hps, hotel_order, prize_per_day, updated_mask)

        hotel_order = greedy_trip_with_exploration(hps.clone(), n_days)  
        new_order_number = sequence_to_order(hotel_order, hotel_size)  
        complete_solution, new_score, prize_per_day  = RL_inference(new_order_number, scores, n_days)
        # print(hotel_order, new_score)

        scores_history.append(new_score)

        if new_score > best_score:
            best_order, best_score, best_complete_solution = new_order_number, new_score, complete_solution
            no_improve_count = 0  
        else:
            no_improve_count += 1 

        improvement_each_iter.append((previous_hps != hps).sum().item())

    # best_complete_solution, best_score =  best_solution_augmentation(best_order, scores, augmentation_factor = 16)    #Augment the final results
    iterations_to_convergence = len(scores_history) - max_no_improve
    return best_order, best_score, best_complete_solution, hps, improvement_each_iter, iterations_to_convergence

#####################################################   Main loop   ###################################################################################

#inputs
base_pt_path = "../../../Instances/OPHS_pt/*.pt"
base_ophs_path = "../../../Instances/raw_OPHS_instances/*.ophs"
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "OPHS_RL_trained_on_32_Aug16x_3_repeat.xlsx")
    
repeats = 3
augmentation_factor = 1

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
    hps, scores, hotels_number, day_number = parse_instance(ophs_path, stochastic_prize)
    df_results = run_repeats(instance_name, hps, hotels_number, day_number, repeats, scores)
    all_results.append(df_results)
 
final_results = pd.concat(all_results, ignore_index=True)

summary = final_results.groupby("Instance").agg(
            Mean_Final_Score=("Final_Score", "mean"),
            Max_Final_Score=("Final_Score", "max"),
            Min_Final_Score=("Final_Score", "min"),
            Mean_Runtime=("Runtime", "mean"),
            Max_Runtime=("Runtime", "max"),
            Min_Runtime=("Runtime", "min"),
            Mean_iter_to_converge = ("iteration_to_converge", "mean"),
        ).reset_index()

best_solution_df = final_results.loc[final_results.groupby("Instance")["Final_Score"].idxmax(), ["Instance", "Best_solution"]]
summary = summary.merge(best_solution_df, on="Instance", how="left")

with pd.ExcelWriter(output_file) as writer:
    final_results.to_excel(writer, index=False, sheet_name="Raw Results")
    summary.to_excel(writer, index=False, sheet_name="Summary Statistics")

print(f"\nResults saved to {output_file}\n")



# same_indices = (final_hps == hps).sum().item()  # Count matching elements
# total_indices = hps.numel()                   # Total number of elements
# metric = same_indices / total_indices
# print(hps, final_hps, metric)


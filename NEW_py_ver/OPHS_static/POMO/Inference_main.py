##########################################################################################
import warnings
warnings.filterwarnings("ignore", message="UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, please use torch.set_default_dtype() and torch.set_default_device() as alternatives")  
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# Path Config

import math
import os
import sys
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

import time
import torch
from scipy.spatial.distance import pdist, squareform
import numpy as np

import random

import os
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
# import

import logging
from utils.utils import create_logger, copy_all_src

##########################################################################################
# parameters
stochastic_prize = False
env_params = {
    'problem_size': 30,
    'pomo_size': 30,
    'hotel_size': 7,
    'day_number': 3,
    'test_stage': True,
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
        # 'path': './result/ophs_H7D3_510_fixed_order',  # directory path of pre-trained model and log files saved.
        'path': './result/ophs_so_32',  # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch number of pre-trained model to laod.
    },
    'test_episodes': 10*1000,
    'test_batch_size': 1000,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 400,
    'test_data_load': {
        'enable': True,
        # 'filename': './T1-65-5-3.pt',
        'filename': '../../../100-160-15-8.pt',
        'hotel_swap': True
        # 'order': None  # Add 'order' to hold the hotel order

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
        
        # if self.tester_params['test_data_load']['enable']:
        #     self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device, hotel_swap= self.tester_params['test_data_load']['hotel_swap'], order= self.tester_params['test_data_load']['order'])
        
        
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
            state, self.reward, self.prize_per_day, done = self.env.step(selected)
            self.path[self.step_count] = selected

###################################################################################
if __name__ == '__main__': 

    def plot_trip(sequence, hotel_number, hotel_order, coordinates, scores):
        """
        Plots the solution path for the trip.

        Args:
            sequence (list): The sequence of nodes visited, including hotels and nodes.
            hotel_order (list): The actual hotel orders to replace indices of hotels in the sequence.
            coordinates (numpy array): Coordinates of all nodes and hotels (37 entries: 7 hotels + 30 nodes).
            scores (list): Scores for all entries (37 entries: 7 zeros for hotels + 30 prizes for nodes).
        """
        # Replace only hotel indices in the sequence
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


###################################################################################
  
    instance_path = r"../../../T1-65-5-3.ophs"
    # instance_path = r"100-210-15-10.ophs"
    # instance_path = r"100-50-12-6.ophs"
    # instance_path = r"../../../T1-65-2-3.ophs"

    def is_valid_line(line):
        parts = line.strip().split()
        if len(parts) != 3:
            return False
        try:
            # Try converting all three parts to floats
            float(parts[0])
            float(parts[1])
            float(parts[2])
            return True
        except ValueError:
            return False

    # Read the data from the file
    with open(instance_path, 'r') as file:
        # Read the first line for n, h, d
        first_line = file.readline().strip().split()
        n = int(first_line[0])  # number of nodes
        h = int(first_line[1])  # number of hotels
        day_number = int(first_line[2])  # number of days
        
        # Skip the third line (as it's unrelated based on the new structure)
        file.readline()
        # Read the second line for t
        t_max = float(file.readline().split()[0])  # max time
        # Skip the third line (as it's unrelated based on the new structure)
        file.readline()
        # Read the remaining lines and filter only valid numeric lines
        coordinates_scores = [line.strip().split() for line in file.readlines() if is_valid_line(line)]

    # Convert the x, y, score into separate lists and convert to floats
    x_coords = [float(line[0]) for line in coordinates_scores]
    y_coords = [float(line[1]) for line in coordinates_scores]
    scores = [float(line[2]) for line in coordinates_scores]

    # Calculate the number of nodes
    hotels_number = h + 2  # Include 2 extra nodes for start and end
    nodes_number = n + hotels_number - 2  # Total nodes

    # Create index lists for hotels and points of interest
    all_nodes_index = list(range(nodes_number))
    hotel_nodes_index = all_nodes_index[:hotels_number]

    # Stack the coordinates into a single array
    coordinates = np.vstack((x_coords, y_coords)).T
    
    # Calculate the pairwise distances using pdist
    distances = pdist(coordinates)
    # Convert the distance matrix to a squareform matrix
    distance_matrix = squareform(distances)

    #####################################################   FUNCTIONS   ###################################################################################

    def create_hps_matrix(hotels_number, hotel_nodes_index, all_nodes_index, distance_matrix, scores):
            
        # Making hotel selection probability matrix
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
                    hps[a,b] += scores[node_i]
                    if a != b: 
                        hps[b,a] += scores[node_i]

        hps = torch.tensor(hps)  # Convert to tensor

        return hps
    
    def order_to_sequence(order, N, K):
        sequence = [0]  # Start with the fixed start point
        current_order = order
        for _ in range(K):
            point = current_order % N
            sequence.append(point)
            current_order //= N
        sequence.append(1)  # Add the fixed end point
        return sequence

    def sequence_to_order(sequence, h):

        K = len(sequence) - 2  # Exclude start and end points
        order = 0
        for i in range(K):
            order += sequence[i + 1] * (h ** i)  # Use i + 1 to skip the start point (0)
        return order

    def RL_inference(input_order):
        # Initialize the OPTester instance
        self = OPTester(env_params=env_params, model_params=model_params, tester_params=tester_params)

        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(
                self.tester_params['test_data_load']['filename'], 
                self.device, 
                hotel_swap=self.tester_params['test_data_load']['hotel_swap'], 
                order=input_order  # Ensure order is passed here
            )
        # Run the RL model
        self.run(batch_size=1)

        # Extract the results from self after running
        pomo = torch.argmax(self.reward)
        complete_order = [int(self.path[i][0][pomo]) for i in self.path]
        best_score = float(self.reward[0, pomo].item())
        prize_per_day = torch.stack(self.prize_per_day, dim=0)[..., pomo].squeeze()

        return complete_order, best_score, prize_per_day

    def simulated_annealing(hps, n_days, initial_solution=None, T0=1000, T_min=0.1, alpha=0.99):
        """
        Apply Simulated Annealing to the hotel sequence problem.

        Args:
            hps (torch.Tensor): Hotel Profitability Score matrix.
            n_days (int): Number of days in the trip.
            initial_solution (list): An initial hotel sequence.
            T0 (float): Initial temperature.
            T_min (float): Minimum temperature for stopping.
            alpha (float): Cooling rate.

        Returns:
            best_solution (list): The best hotel sequence found.
            best_score (float): The score of the best hotel sequence.
        """
        def evaluate(solution):
            score = 0
            for i in range(len(solution) - 1):
                score += hps[solution[i], solution[i + 1]].item()
            return score

        # If no initial solution is provided, generate one randomly
        if initial_solution is None:
            available_hotels = list(range(0, hps.shape[0]))
            initial_solution = [0] + random.sample(available_hotels[1:], n_days - 1) + [1]

        current_solution = initial_solution
        best_solution = current_solution[:]
        best_score = evaluate(current_solution)

        T = T0
        while T > T_min:
            # Generate a neighbor
            new_solution = current_solution[:]
            i, j = random.sample(range(1, n_days + 1), 2)  # Pick two hotels to swap
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
    
    #####################################################   Main loop   ###################################################################################
    # Function to run the optimization multiple times and store results
    def run_repeats_and_save(hps, hotel_size, n_days, max_no_improve_values, repeats, output_file):
        results = []

        # Loop over different max_no_improve values
        for max_no_improve in max_no_improve_values:
            print(f"Running for max_no_improve = {max_no_improve}")

            # Repeat the optimization for the given number of repeats
            for repeat in range(repeats):

                start_time = time.time()
                best_order, best_score, best_solution, final_hps, iterations_to_convergence, rate_of_improvement, percentage_of_improvement = optimize_trip(hps.clone(), hotel_size, n_days, max_no_improve)
                end_time = time.time()
                runtime = end_time - start_time

                same_indices = (hps == final_hps).sum().item()  # Count matching elements
                total_indices = hps.numel()                   # Total number of elements
                metric = same_indices / total_indices

                # Save results for this repeat
                results.append({
                    "Max_No_Improve": max_no_improve,
                    "Repeat": repeat + 1,
                    "Final_Score": best_score,
                    "Runtime": runtime,
                    "Percentage_of_Unchanged_indexes": metric*100,
                    "Iterations_to_Convergence": iterations_to_convergence,
                    "Rate_of_Improvement": rate_of_improvement,
                    "Percentage_of_Improvement": percentage_of_improvement,
                })

        # Convert results to a DataFrame
        df = pd.DataFrame(results)

        # Compute summary statistics
        summary = df.groupby("Max_No_Improve").agg(
            Mean_Final_Score=("Final_Score", "mean"),
            Std_Final_Score=("Final_Score", "std"),
            Mean_Runtime=("Runtime", "mean"),
            Std_Runtime=("Runtime", "std"),
            Mean_Iterations_to_Convergence=("Iterations_to_Convergence", "mean"),
            Std_Iterations_to_Convergence=("Iterations_to_Convergence", "std"),
            Mean_Rate_of_Improvement=("Rate_of_Improvement", "mean"),
            Std_Rate_of_Improvement=("Rate_of_Improvement", "std"),
            Mean_Percentage_of_Improvement=("Percentage_of_Improvement", "mean"),
            Std_Percentage_of_Improvement=("Percentage_of_Improvement", "std"),
        ).reset_index()

        # Save raw results and summary to an Excel file
        with pd.ExcelWriter(output_file) as writer:
            df.to_excel(writer, index=False, sheet_name="Raw Results")
            summary.to_excel(writer, index=False, sheet_name="Summary Statistics")

        print(f"Results saved to {output_file}")


    def optimize_trip(hps, hotel_size, n_days, max_no_improve=5):

        hotel_order = greedy_trip_with_penalty(hps, n_days)  # Get hotel order
        best_order_number = sequence_to_order(hotel_order, hotel_size)  # Convert sequence to order
        complete_solution, best_score, prize_per_day = RL_inference(best_order_number)

        best_order = best_order_number
        best_complete_solution = complete_solution
        no_improve_count = 0
        updated_mask = torch.zeros_like(hps, dtype=torch.bool)  # Initialize the updated mask
        scores_history = [best_score] 

        while no_improve_count < max_no_improve:
            for day in range(len(prize_per_day)):
                from_hotel = hotel_order[day]
                to_hotel = hotel_order[day + 1]
                prize = prize_per_day[day]

                if not updated_mask[from_hotel, to_hotel]:  # First-time update
                    hps[from_hotel, to_hotel] = prize
                    hps[to_hotel, from_hotel] = prize
                    updated_mask[from_hotel, to_hotel] = True
                    updated_mask[to_hotel, from_hotel] = True
                else:       # Subsequent updates only if the prize is greater 
                    hps[from_hotel, to_hotel] = max(hps[from_hotel, to_hotel], prize)
                    hps[to_hotel, from_hotel] = max(hps[to_hotel, from_hotel], prize)

            hotel_order = greedy_trip_with_penalty(hps, n_days)  # Get new hotel order
            new_order_number = sequence_to_order(hotel_order, hps.size(0))  # Convert sequence to order
            complete_solution, new_score, prize_per_day = RL_inference(new_order_number)
            scores_history.append(new_score)

            if new_score > best_score:
                best_order, best_score, best_complete_solution = new_order_number, new_score, complete_solution
                no_improve_count = 0  
            else:
                no_improve_count += 1 

        iterations_to_convergence = len(scores_history) - max_no_improve
        rate_of_improvement = (best_score - scores_history[0]) / iterations_to_convergence
        precentage_rate_of_improvement = (best_score - scores_history[0]) / (scores_history[0])

        return best_order, best_score, best_complete_solution, hps, iterations_to_convergence, rate_of_improvement, precentage_rate_of_improvement

    ####################################### testing #######################

    # hps = create_hps_matrix(hotels_number, hotel_nodes_index, all_nodes_index, distance_matrix, scores)
    # hps_old = hps.clone()
    # start_time = time.time()
    # order_number, reward, solution, final_hps, iteration, rate, p = optimize_trip(hps, hotels_number, day_number, max_no_improve=10)
    # end_time = time.time()
    # runtime = end_time - start_time

    # hotel_sequence = order_to_sequence(order_number, hotels_number, day_number-1)

    # cleaned_solution = []
    # for value in solution:
    #     if value not in cleaned_solution:
    #         cleaned_solution.append(value)

    # mapping = {i: hotel_sequence[i] for i in range(len(hotel_sequence))}
    # clean_solution = [mapping[value] if value in mapping else value for value in cleaned_solution]

    # same_indices = (hps_old == hps).sum().item()  # Count matching elements
    # total_indices = hps.numel()                   # Total number of elements
    # metric = same_indices / total_indices

    # print(f"\n {same_indices}/{total_indices} did not change ({metric*100}%)")
    # print(f"\n first hps:\n {hps_old}\n final hps:\n {final_hps}\n\n iter:{iteration}\n rate:{rate}\n per:{p}%\n Sequence: {hotel_sequence}\n total reward: {reward}\n raw solution:   {solution}\n final solution: {clean_solution}\n Runtime: {runtime} seconds\n")


    # plot_trip(solution, hotels_number, hotel_sequence, coordinates, scores)
    # trip = greedy_trip_with_exploration(hps,day_number)
    # order = sequence_to_order(trip, hotels_number)
    # print(trip, order)
#################################################################################################
    # max_no_improve_values = [5, 10]
    max_no_improve_values = [5, 10, 20, 50, 100]
    repeats = 20
    output_file = "output_results/greedy_with_penalty_32.xlsx"
    hps = create_hps_matrix(hotels_number, hotel_nodes_index, all_nodes_index, distance_matrix, scores)

    # Run the experiment
    run_repeats_and_save(hps, hotels_number, day_number, max_no_improve_values, repeats, output_file)



from dataclasses import dataclass
import torch
from OPHSSPProblemDef import get_random_problems, augment_xy_data_by_8_fold, augment_xy_data_by_16_fold

@dataclass 
class Reset_state : 
    depot_xy: torch.Tensor = None
    # shape: (batch, hotel, 2)
    day_number: torch.Tensor = None
    # shape: (batch, 1)
    trip_length: torch.Tensor = None
    # shape: (batch, day, 1)
    node_xy : torch.Tensor = None
    # shape: (batch, problem, 2)
    node_prize : torch.Tensor = None
    # shape: (batch, problem)
    
@dataclass
class Step_state : 
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    HOTEL_IDX: torch.Tensor = None
    # shape: (batch, pomo, hotel)

    current_node: torch.Tensor = None
    selected_count: int = None
    remaining_len : torch.Tensor = None
    ninf_mask: torch.Tensor  = None
    # shape: (batch, pomo, node)
    
    finished: torch.Tensor = None
    
class OPHSSPEnv: 
    def __init__(self, **env_params) : 
        
        # Const @INIT
        ####################################       
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size'] 
        self.hotel_size = env_params['hotel_size']                   
        self.stochastic_prize = env_params['stochastic_prize']

        
        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_prize = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.HOTEL_IDX = None

        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+hotel , 2)
        self.depot_node_prize = None
        # shape: (batch, problem+hotel)
        
        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)
        
        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.remaining_len = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+hotel)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+hotel)
        self.finished = None
        # shape: (batch, pomo)
        
        # states to return
        ################.####################
        self.reset_state = Reset_state()
        self.step_state = Step_state()

    def use_saved_problems(self, filename, device, hotel_swap: bool = False, order: int=0):             
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)

        self.saved_day_number = loaded_dict['day_number']
        self.saved_depot_xy = loaded_dict['hotel_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_remain_len = loaded_dict['remain_len']
        if self.stochastic_prize:
            self.saved_node_prize = torch.stack((loaded_dict['mean'], loaded_dict['deviation']), dim=-1)    
        else:
            self.saved_node_prize = loaded_dict['node_prize']
        
        self.saved_index = 0   

        if hotel_swap:
            coords = self.saved_depot_xy.squeeze(0)

            total_states = self.hotel_size**(self.saved_day_number)
            if order >= total_states:
                raise ValueError(f"Order {order} exceeds the total number of states {total_states}.")

            # Compute the state corresponding to the given order
            state = [0]  # Start with the fixed start point
            current_order = order
            for _ in range(self.saved_day_number-1):
                point = current_order % self.hotel_size
                state.append(point)
                current_order //= self.hotel_size
            state.append(1)  # Add the fixed end point

            # Ensure the sequence length matches hotel_size by appending 1s
            while len(state) < self.hotel_size:
                state.append(1)

            # Extract coordinates for the computed state
            self.saved_depot_xy = coords[state, :].unsqueeze(0)

    def load_problems(self, batch_size, aug_factor=1) : 
        self.batch_size = batch_size
        
        if not self.FLAG__use_saved_problems:
            day_number, depot_xy, node_xy, node_prize, trip_length = get_random_problems(batch_size, self.problem_size, self.hotel_size)  
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_prize = self.saved_node_prize[self.saved_index:self.saved_index+batch_size]
            trip_length = self.saved_remain_len[self.saved_index:self.saved_index+batch_size]
            day_number = self.saved_day_number
            self.saved_index += batch_size
       
        self.trip_length = trip_length
        self.depot_xy = depot_xy
        self.node_xy = node_xy
        self.day_number = day_number

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.depot_xy = augment_xy_data_by_8_fold(depot_xy)
                self.node_xy = augment_xy_data_by_8_fold(node_xy)
                self.trip_length = self.trip_length.repeat(8, 1)           
                if self.stochastic_prize:
                    node_prize = node_prize.repeat(8, 1, 1)
                else:
                    node_prize = node_prize.repeat(8, 1)
            elif aug_factor == 16:
                self.batch_size = self.batch_size * 16
                self.depot_xy = augment_xy_data_by_16_fold(depot_xy)
                self.node_xy = augment_xy_data_by_16_fold(node_xy)
                self.trip_length = self.trip_length.repeat(16, 1)
                if self.stochastic_prize:
                    node_prize = node_prize.repeat(16, 1, 1)
                else:
                    node_prize = node_prize.repeat(16, 1)
            else:
                raise NotImplementedError
                
        self.depot_node_xy = torch.cat((self.depot_xy, self.node_xy), dim=1)
        # shape: (batch, problem+hotel, 2)
        if self.stochastic_prize:
            depot_prize = torch.zeros(size=(self.batch_size, self.hotel_size, 2))                            
        else:
            depot_prize = torch.zeros(size=(self.batch_size, self.hotel_size))                            
        # shape: (batch, hotel, 2) or # shape: (batch, hotel)
        self.depot_node_prize = torch.cat((depot_prize, node_prize), dim=1)
        # shape: (batch, problem+hotel, 2)
        
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.HOTEL_IDX = torch.arange(self.hotel_size)[None, None, :].expand(self.batch_size, self.pomo_size, self.hotel_size)      # new IDX to use in OPHSmodel

        self.reset_state.depot_xy = self.depot_xy
        self.reset_state.node_xy = self.node_xy
        self.reset_state.day_number = self.day_number
        self.reset_state.node_prize = node_prize
        self.reset_state.trip_length = trip_length
        
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.HOTEL_IDX = self.HOTEL_IDX
        
    def reset(self) : 
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)
        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.ninf_mask_first_step = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.remaining_len = torch.ones(size=(self.batch_size, self.pomo_size))               
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + self.hotel_size))      
        # shape: (batch, pomo, problem+hotel)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + self.hotel_size))               
        # shape: (batch, pomo, problem+hotel)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.collected_prize = torch.zeros(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        
        self.day_finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.int64)              # 5 new tensors        
        # shape: (batch, pomo)
        self.last_step_mask = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        #shape: (batch, pomo)
        self.zeros_to_add = torch.zeros(self.batch_size, 1)  
        # shape: (batch, 1)  
        self.max_day_number = self.day_number.max().item()
        self.depots_ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.max_day_number+1))             #new 
        # shape: (batch, pomo, max_day)
        self.end_day_flag = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.finishing_depot_index = torch.ones(size=(self.batch_size, self.pomo_size), dtype= torch.long)
        # shape: (batch, pomo)
        self.batch_indices = torch.arange(self.batch_size).unsqueeze(1)  
        # Shape: (batch_size, 1)
        self.pomo_indices = torch.arange(self.pomo_size).unsqueeze(0).expand(self.batch_size, -1)  
        # Shape: (batch_size, pomo_size)
        self.depots_ninf_mask[:, :, :] = float('-inf')
        self.prize_per_day = []
        self.previous_prize = None
        self.flag = True
        
        reward = None
        done = False
        return self.reset_state, reward, done        
    
    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.remaining_len = self.remaining_len
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done    
    
    def step(self, selected) : 
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        
        if self.current_node is not None : 
            self.last_visited_node = self.current_node.clone()
        else : 
            self.last_visited_node = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.int64)
            
        self.current_node = selected.clone()
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        self.remaining_len[self.selected_count == 1] = self.trip_length[:, None, 0]             # Assign length for the first hotel (first step)       
        
        if self.selected_count == 1 :                     #first step(pomo) mask
            self.len_to_starting_depot, self.len_to_finishing_depot = self.calculate_len_to_depot()
            # shape len_to_start: (batch, pomo)
            # shape len_to_finish: (batch, hotel, pomo)
            
            self.first_step_len_too_large = (self.remaining_len < self.len_to_starting_depot + self.len_to_finishing_depot[:,0,:])          #infeasible first step condition
            self.ninf_mask_first_step[self.first_step_len_too_large] = True
            # shape: (batch, pomo)

            self.visited_ninf_flag[self.ninf_mask_first_step.unsqueeze(2).expand_as(self.visited_ninf_flag)] = float('-inf')
            self.finished[self.ninf_mask_first_step] = True

        # Dynamic-2
        ####################################

        self.at_the_depot = (selected == (self.finishing_depot_index))                   #change

        if self.stochastic_prize:
            self.prize_list = self.depot_node_prize[:, None, :, :].expand(self.batch_size, self.pomo_size, -1, -1)
            # shape: (batch, pomo, problem+hotel , 2)
            self.gathering_index = selected[:, :, None, None].expand(-1, -1, 1, 2)
            # shape: (batch, pomo, 1, 1)
            self.selected_prize = self.prize_list.gather(dim=2, index=self.gathering_index).squeeze(dim=2)
            # shape: (batch, pomo, 2)
            raw_rewards = torch.randn(self.batch_size, self.problem_size) * self.selected_prize[:,:,1] + self.selected_prize[:,:,0]

            non_zero_mask = raw_rewards != 0                                    # Clamp only the non-zero values
            node_prizes = raw_rewards.clone()
            node_prizes[non_zero_mask] = torch.clamp(torch.round(raw_rewards[non_zero_mask]), min=0.02, max=0.99)
            self.reward_tensor = node_prizes
            self.collected_prize += self.reward_tensor
        else:
            self.prize_list = self.depot_node_prize[:, None, :].expand(self.batch_size, self.pomo_size, -1)
            # shape: (batch, pomo, problem+hotel)
            self.gathering_index = selected[:, :, None]
            # shape: (batch, pomo, 1)
            self.selected_prize = self.prize_list.gather(dim=2, index=self.gathering_index).squeeze(dim=2)
            # shape: (batch, pomo)
            self.collected_prize += self.selected_prize


        selected_len = self.calculate_two_distance()

        self.remaining_len -= selected_len

        # self.end_day_flag = (self.ninf_mask[:, :, self.hotel_size:] == -float('inf')).all(dim=2) 
        # # shape: (batch, pomo)
        self.day_finished[self.selected_count > 1] += self.at_the_depot                                     # day count   

        self.finishing_depot_index = self.day_finished + 1 
        day_number_expanded = self.day_number.expand(-1, self.pomo_size)
        self.finishing_depot_index = torch.minimum(self.finishing_depot_index, day_number_expanded)

        self.trip_length = torch.cat((self.trip_length, self.zeros_to_add), dim=1)  
        # shape: (batch, day + 0~)
        self.gathering_index_day = self.day_finished.unsqueeze(-1)
        # shape: (batch, pomo, 1)  
        trip_length_expanded = self.trip_length.unsqueeze(1).expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, day)  
        self.trip_length_extracted = trip_length_expanded.gather(2, self.gathering_index_day).squeeze(-1)
        # shape: (batch, pomo)
        mask = self.at_the_depot & (self.day_finished < self.day_number)                                        # mask for length reset
        reward_mask = self.remaining_len < 0                                                            # negative reward

        self.remaining_len[mask] = self.trip_length_extracted[mask]

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+hotel)

        self.depots_ninf_mask[:, :, :] = float('-inf')
        hotel_mask = torch.cat([self.depots_ninf_mask, torch.full((self.batch_size, self.pomo_size, self.hotel_size - self.depots_ninf_mask.shape[-1]), float('-inf'))], dim=-1)

        self.last_step_mask[(self.day_finished >= self.day_number)] = True
        finish_mask = self.last_step_mask & self.at_the_depot                            

        batch_indices_flat_visited = self.batch_indices.expand_as(self.finished)[~finish_mask & ~self.ninf_mask_first_step]
        pomo_indices_flat_visited = self.pomo_indices[~finish_mask & ~self.ninf_mask_first_step]
        finishing_depot_index_flat_visited = self.finishing_depot_index[~finish_mask & ~self.ninf_mask_first_step]
        hotel_mask[batch_indices_flat_visited, pomo_indices_flat_visited, finishing_depot_index_flat_visited] = 0  # depot X is considered unvisited, unless you are AT the end

        self.visited_ninf_flag[:, :, :self.hotel_size] = hotel_mask

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
                    
        _, self.len_to_finishing_depot = self.calculate_len_to_depot()
        # shape len_to_finish: (batch, pomo, problem)
        self.future_len = self.calculate_future_len()       
        # shape: (batch, pomo, problem)

        self.remaining_len_expanded = self.remaining_len.unsqueeze(dim=2).expand(-1,-1,self.problem_size)
        # shape: (batch, pomo, problem)
        self.possible_dists = self.len_to_finishing_depot + self.future_len  
        self.len_too_large = self.remaining_len_expanded + round_error_epsilon < self.possible_dists
        # shape: (batch, pomo, problem)

        self.len_too_large_expanded = torch.cat((torch.zeros_like(self.len_too_large[:,:,:self.hotel_size], dtype=torch.bool), self.len_too_large), dim=-1)       #@chanage
        # shape: (batch, pomo, problem+hotel) 

        self.ninf_mask[self.len_too_large_expanded] = float('-inf')
        self.ninf_mask[:, :, :self.hotel_size] = hotel_mask
        # shape: (batch, pomo, problem+hotel)
        
        #prize for each day
        # if self.previous_prize is None:
        #     self.prize_per_day.append(self.collected_prize.clone())
        # else:
        #     difference = self.collected_prize - self.previous_prize
        #     self.prize_per_day.append(difference)

        # self.previous_prize = self.collected_prize.clone()
    
        self.newly_finished = (self.ninf_mask == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + self.newly_finished
        # shape: (batch, pomo)

        batch_indices_flat = self.batch_indices.expand_as(self.finished)[self.finished & ~self.ninf_mask_first_step]            # Flatting for masking in advanced indexing contex
        pomo_indices_flat = self.pomo_indices[self.finished & ~self.ninf_mask_first_step]
        finishing_depot_index_flat = self.finishing_depot_index[self.finished & ~self.ninf_mask_first_step]

        self.ninf_mask[batch_indices_flat, pomo_indices_flat, finishing_depot_index_flat] = 0               # do not mask finishing depot for finished episode
      
        self.ninf_mask[:, :, 0][self.finished & self.ninf_mask_first_step] = 0      #first step masking  

        self.step_state.selected_count = self.selected_count
        self.step_state.remaining_len = self.remaining_len
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        # done = (self.day_finished >= self.day_number).all()
        if done:
            reward_mask = self.remaining_len < 0                                # negative reward
            self.collected_prize[reward_mask] = (self.remaining_len[reward_mask]*1000.00) / self.collected_prize[reward_mask]

            reward = self.collected_prize
            # print(f'########selected nodes######## \n{self.selected_node_list}\n\n########trip length#######\n{self.trip_length}\n\n######final reward#####\n{reward}')    #for testing
            # print(self.trip_length)
            # print(f'\n{self.trip_length}\n{self.selected_node_list[0,2]}\n{self.prize_list[0,0]}\nprizes:   {reward[0,2]}\n')
        else:
            reward = None

        # return self.step_state, reward, self.prize_per_day, done
        return self.step_state, reward, done


    def calculate_len_to_depot(self) :
        
        self.starting_depot_xy_expanded = self.depot_xy[:, 0, :].unsqueeze(1).expand(-1, self.problem_size, -1)
        # Shape: (batch, problem, 2)   
        self.finishing_depot_xy_expanded = self.depot_xy.unsqueeze(2).expand(-1, -1, self.problem_size, -1)
        #shape: (batch, hotel, problem, 2)

        # Calculate squared differences and sum along the last dimension
        squared_diff = (self.starting_depot_xy_expanded - self.node_xy) ** 2
        #shape: (batch, problem, 2)
        self.node_xy_expanded = self.node_xy.unsqueeze(1).expand(-1, self.hotel_size, -1, -1)
        #shape: (batch, hotel, problem, 2)
        squared_diff2 = (self.finishing_depot_xy_expanded - self.node_xy_expanded) ** 2
        #shape: (batch, hotel, problem, 2)

        self.distance_sums = torch.sum(squared_diff, dim=2)
        #shape: (batch, problem)
        self.distance_sums2_unextracted = torch.sum(squared_diff2, dim=3)
        #shape: (batch, hotel, problem)
        self.distance_sums2 = self.distance_sums2_unextracted[self.batch_indices, self.finishing_depot_index]
        #shape: (batch, pomo, problem)

        # Square root to get the Euclidean distances
        len_to_starting_depot = torch.sqrt(self.distance_sums)
        #shape: (batch, problem)
        len_to_finishing_depot = torch.sqrt(self.distance_sums2)
        #shape: (batch, pomo, problem)

        return len_to_starting_depot, len_to_finishing_depot 

    def calculate_future_len(self) : 
        self.node_xy_expanded = self.node_xy.unsqueeze(dim=1).expand(-1, self.problem_size, -1, -1)
        #shape: (batch, pomo, problem, 2)
        self.current_xy_expanded = self.node_xy_current.unsqueeze(dim=2).expand(-1, -1, self.problem_size, -1)
        #shape: (batch, pomo, problem, 2)

        # Calculate squared differences and sum along the last dimension
        squared_diff = (self.current_xy_expanded - self.node_xy_expanded) ** 2
        #shape: (batch, pomo, problem, 2)
        distance_sums = torch.sum(squared_diff, dim=3)                                  
        #shape: (batch, problem)

        # Square root to get the Euclidean distances
        future_len = torch.sqrt(distance_sums)
        #shape: (batch, pomo, problem)
        return future_len 
    
    def calculate_two_distance(self) : 
        
        current_node_zero_indexed = self.current_node
        last_visited_node_zero_indexed = self.last_visited_node 
        
        # Expanding the dimensions of current_node to match the dimensions of node_xy for gathering
        current_node_expanded = current_node_zero_indexed.unsqueeze(dim=2).expand(-1, -1, 2)
        #shape: (batch, problem, 2)

        # Using gather to fill the node_xy_current tensor
        self.node_xy_current = torch.gather(self.depot_node_xy, 1, current_node_expanded)
        #shape: (batch, problem, 2)

        # Expanding the dimensions of last_visited_node to match the dimensions of node_xy for gathering
        last_visited_node_expanded = last_visited_node_zero_indexed.unsqueeze(dim=2).expand(-1, -1, 2)
        #shape: (batch, problem, 2)
        
        # Using gather to fill the node_xy_last tensor
        self.node_xy_last_visited = torch.gather(self.depot_node_xy, 1, last_visited_node_expanded)
        #shape: (batch, problem, 2)

        self.squared_diff = (self.node_xy_last_visited - self.node_xy_current) ** 2
        #shape: (batch, problem, 2)
        self.distance_sums = torch.sum(self.squared_diff, dim=2)
        #shape: (batch, problem)

        # Square root to get the Euclidean distances
        selected_len = torch.sqrt(self.distance_sums)
        #shape: (batch, problem)
        
        return selected_len

from dataclasses import dataclass
import torch
from OPProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass 
class Reset_state : 
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy : torch.Tensor = None
    # shape: (batch, problem, 2)
    node_prize : torch.Tensor = None
    # shape: (batch, problem)
    
@dataclass
class Step_state : 
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    
    current_node: torch.Tensor = None
    
    selected_count: int = None
    remaining_len = torch.Tensor = None
    ninf_mask: torch.Tensor  = None
    # shape: (batch, pomo, node)
    
    finished: torch.Tensor = None
    
class OPEnv: 
    def __init__(self, **env_params) : 
        
        # Const @INIT
        ####################################       
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size'] 
        
        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1 , 2)
        self.depot_node_prize = None
        # shape: (batch, problem+1)
        
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
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)
        
        # states to return
        ################.####################
        self.reset_state = Reset_state()
        self.step_state = Step_state()
        
    def load_problems(self, batch_size, aug_factor=1) : 
        self.batch_size = batch_size
        
        depot_xy, node_xy, node_prize = get_random_problems(batch_size, self.problem_size)
        #@todo 
        self.depot_xy = depot_xy
        self.node_xy = node_xy
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_prize = node_prize.repeat(8, 1)
            else:
                raise NotImplementedError
                
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_prize = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_prize = torch.cat((depot_prize, node_prize), dim=1)
        # shape: (batch, problem+1)
    
        
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_prize = node_prize
        
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        
    def reset(self) : 
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.remaining_len = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

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
            self.last_node = self.current_node.clone()
        else : 
            self.last_node = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.int64)
            
        self.current_node = selected.clone()
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)
        
        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)
        #fully undrestood
        self.prize_list = self.depot_node_prize[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        self.gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        self.selected_prize = self.prize_list.gather(dim=2, index=self.gathering_index).squeeze(dim=2)
        
        #todo 
        selected_len = self.calculate_two_distance()
        print(f'selected_len : {selected_len}')
        print(f'remaining_len : {self.remaining_len} ')
        self.remaining_len -= selected_len
        print(f'remaining_len : {self.remaining_len} ')
        
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        
        round_error_epsilon = 0.00001
        # demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # # shape: (batch, pomo, problem+1)
        # self.ninf_mask[demand_too_large] = float('-inf')
        # # shape: (batch, pomo, problem+1)
        self.len_to_depot = self.calculate_len_to_depot()
        # self.possible_dists = self.len_to_depot + self.dist_to_selected  
        self.possible_dists = self.len_to_depot
        self.len_too_large = self.remaining_len + round_error_epsilon < self.possible_dists 
        
        self.newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + self.newly_finished
        # shape: (batch, pomo)
        
    def calculate_len_to_depot(self) :
        
        self.depot_xy_expanded = self.depot_xy.expand_as(self.node_xy)
        
        # Calculate squared differences and sum along the last dimension
        self.squared_diff = (self.depot_xy_expanded - self.node_xy) ** 2
        self.distance_sums = torch.sum(self.squared_diff, dim=2)

        # Square root to get the Euclidean distances
        len_to_depot = torch.sqrt(self.distance_sums)
        return len_to_depot
        
    def calculate_future_len(self) : 
        
        self.current_xy_expanded = self.node_xy_current.expand_as(self.node_xy)
        
        # Calculate squared differences and sum along the last dimension
        squared_diff = (self.current_xy_expanded - self.node_xy) ** 2
        distance_sums = torch.sum(squared_diff, dim=2)

        # Square root to get the Euclidean distances
        future_len = torch.sqrt(distance_sums)
        return future_len 
    
    def calculate_two_distance(self) : 
        
        current_node_zero_indexed = self.current_node
        last_node_zero_indexed = self.last_node 
        
        #todo 
        # Expanding the dimensions of current_node to match the dimensions of node_xy for gathering
        current_node_expanded = current_node_zero_indexed.unsqueeze(2).expand(-1, -1, 2)

        # Using gather to fill the node_xy_current tensor
        self.node_xy_current = torch.gather(self.depot_node_xy, 1, current_node_expanded)

        # Expanding the dimensions of last_node to match the dimensions of node_xy for gathering
        last_node_expanded = last_node_zero_indexed.unsqueeze(2).expand(-1, -1, 2)

        # Using gather to fill the node_xy_last tensor
        self.node_xy_last = torch.gather(self.depot_node_xy, 1, last_node_expanded)
        
        squared_diff = (self.node_xy_last - self.node_xy_current) ** 2
        distance_sums = torch.sum(squared_diff, dim=2)

        # Square root to get the Euclidean distances
        selected_len = torch.sqrt(distance_sums)
        
        return selected_len

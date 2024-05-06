from dataclasses import dataclass
import torch
import numpy as np
from OPHSProblemDef import get_random_problems, augment_xy_data_by_8_fold


@dataclass 
class Reset_state : 
    hotel_xy: torch.Tensor = None
    # shape: (batch, hotel, 2)
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
    remaining_len : torch.Tensor = None
    ninf_mask: torch.Tensor  = None
    # shape: (batch, pomo, node)
    
    finished: torch.Tensor = None
    
class OPHSEnv: 
    def __init__(self, **env_params) : 
        
        # Const @INIT
        ####################################       
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.hotel_number = env_params['hotel_number'] 
        
        self.FLAG__use_saved_problems = False
        self.saved_hotel_xy = None
        self.saved_node_xy = None
        self.saved_node_prize = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.hotel_node_xy = None
        # shape: (batch, problem+hotel , 2)
        self.hotel_node_prize = None
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
        self.at_the_hotel = None
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

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_hotel_xy = loaded_dict['hotel_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_prize = loaded_dict['node_demand']
        self.saved_index = 0    

    def load_problems(self, batch_size, aug_factor=1) : 
        self.batch_size = batch_size
        
        if not self.FLAG__use_saved_problems:
            hotel_xy, node_xy, node_prize = get_random_problems(batch_size, self.problem_size, self.hotel_number)
        else:
            hotel_xy = self.saved_hotel_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_prize = self.saved_node_prize[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        self.hotel_xy = hotel_xy
        self.node_xy = node_xy

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.hotel_xy = augment_xy_data_by_8_fold(hotel_xy)
                self.node_xy = augment_xy_data_by_8_fold(node_xy)
                node_prize = node_prize.repeat(8, 1)
            else:
                raise NotImplementedError
                
        self.hotel_node_xy = torch.cat((self.hotel_xy, self.node_xy), dim=1)
        # shape: (batch, problem+hotel, 2)
        hotel_prize = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.hotel_node_prize = torch.cat((hotel_prize, node_prize), dim=1)
        # shape: (batch, problem+hotel)
    
        
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.hotel_xy = self.hotel_xy
        self.reset_state.node_xy = self.node_xy
        self.reset_state.node_prize = node_prize
        
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        
    def reset(self) : 
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_hotel = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.ninf_mask_first_step = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.remaining_len = 1*torch.ones(size=(self.batch_size, self.pomo_size))               
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+self.hotel_number))      
        # shape: (batch, pomo, problem+hotel)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+self.hotel_number))               
        # shape: (batch, pomo, problem+hotel)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        #new tensor 
        self.collected_prize = torch.zeros(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        
        self.day_finished = 0
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
        
        # Dynamic-2
        ####################################
        # self.at_the_hotel = (selected >= 0) & (selected < self.hotel_number)
        self.at_the_hotel = (selected == 0)
        
        selected_len = self.calculate_two_distance()

        self.first_step_len_too_large = (self.remaining_len/2 < selected_len)  
        print('bibbbbbbbbbbbbb')             #infeasible first step condition
        self.ninf_mask_first_step[self.first_step_len_too_large] = False
        print(self.ninf_mask_first_step)
        # shape: (batch, pomo)

        if self.selected_count == 2 :                                                       #first step    
            # print(f'first step mask {self.ninf_mask_first_step}')                            
            selected = torch.where(self.ninf_mask_first_step, selected, torch.tensor(0))        #using 'where' method to change only one element of tensor
            selected_len = torch.where(self.ninf_mask_first_step, selected_len, torch.tensor(0.0))
            self.visited_ninf_flag[~self.ninf_mask_first_step.unsqueeze(2).expand_as(self.visited_ninf_flag)] = float('-inf')
            self.finished = torch.where(self.ninf_mask_first_step, self.finished, torch.tensor(bool(True)))

        if self.selected_count == 3 :                                                        #second step (to correct wrong remaining length bug)
            selected_len = torch.where(self.ninf_mask_first_step, selected_len, torch.tensor(0.0))

        self.prize_list = self.hotel_node_prize[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+hotel)
        self.gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        self.selected_prize = self.prize_list.gather(dim=2, index=self.gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.collected_prize += self.selected_prize

        # print(f'\nselected_nodes \n {selected} \nvisited \n {self.visited_ninf_flag}')

        print(f'selected_nodes : {selected} \n selected_len : {selected_len} \n')
        # print(f'remaining_len before step : {self.remaining_len}\n')
        self.remaining_len -= selected_len
        print(f'remaining_len after step: {self.remaining_len}\n')
        # print(f'prize {self.collected_prize}')

        # self.remaining_len[self.at_the_hotel] = 1 # reset length at the hotel

        # print(self.BATCH_IDX,'\n',self.POMO_IDX)
        # test = torch.zeros(size=(self.batch_size, self.pomo_size))
        # test = self.POMO_IDX + self.hotel_number
        # print(test,'\n',self.POMO_IDX)
        # print(self.visited_ninf_flag.shape)
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected+self.hotel_number-1] = float('-inf')
        self.visited_ninf_flag[:, :, :self.hotel_number][selected == 0] = float('-inf')
        # shape: (batch, pomo, problem+1)
        # print(selected,'\n',self.visited_ninf_flag)
        # print(self.at_the_hotel)
        # self.visited_ninf_flag[:, :, 0][~self.at_the_hotel] = 0  # hotel is considered unvisited, unless you are AT the hotel
        self.visited_ninf_flag[:, :, :self.hotel_number][~self.at_the_hotel] = 0
        print('\n',self.visited_ninf_flag)
        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        

        self.len_to_hotel = self.calculate_len_to_hotel()
        # shape: (batch, hotel, problem)
        # print(self.len_to_hotel)
        selected_hotel = torch.full((self.batch_size, self.problem_size), self.day_finished+1)
        # shape: (batch, problem)
        self.hotel_gathering_index = selected_hotel[:, None, :]
        # shape: (batch, 1, problem)
        self.len_to_selected_hotel = self.len_to_hotel.gather(dim=1, index=self.hotel_gathering_index).squeeze(dim=1)
        # shape: (batch, problem)
        # print(self.len_to_hotel,'\n',self.len_to_selected_hotel)

        # self.len_to_hotel_expanded = self.len_to_hotel.unsqueeze(dim=2).expand(-1,-1,self.problem_size,-1)
        # shape: (batch, hotel, pomo, problem)

 

        self.len_to_hotel_expanded = self.len_to_selected_hotel.unsqueeze(dim=1).expand(-1,self.problem_size,-1)
        # shape: (batch, pomo, problem)
        self.future_len = self.calculate_future_len()       
        # shape: (batch, pomo, problem)

        self.remaining_len_expanded = self.remaining_len.unsqueeze(dim=2).expand(-1,-1,self.problem_size)
        # shape: (batch, pomo, problem)
        
        self.possible_dists = self.len_to_hotel_expanded + self.future_len  
        self.len_too_large = self.remaining_len_expanded + round_error_epsilon < self.possible_dists
        # shape: (batch, pomo, problem)

        self.len_too_large_expanded = torch.cat((torch.zeros_like(self.len_too_large[:,:,:self.hotel_number], dtype=torch.bool), self.len_too_large), dim=-1)
        # shape: (batch, pomo, problem+hotel) 
        self.ninf_mask[self.len_too_large_expanded] = float('-inf')                
        # shape: (batch, pomo, problem+hotel)
        self.newly_finished = (self.ninf_mask == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + self.newly_finished
        # shape: (batch, pomo)
        # print(f'finished {self.finished}')
        # self.day_finished = self.finished.all()
        # do not mask hotel for finished episode.
        self.ninf_mask[:, :, :self.hotel_number][self.finished] = 0          #@todo
        # print(self.ninf_mask)
        if self.finished.all() :
            self.day_finished += 1
            self.finished = False
            self.remaining_len[:,:] = 1 # reset length at the hotel

            print(f'prize {self.collected_prize}')
            print(f'################################### End of day = {self.day_finished+1} ###########################\n\n')


        # # do not mask hotel for finished episode.
        # self.ninf_mask[:, :, 0][self.finished] = 0
        
        self.step_state.selected_count = self.selected_count
        self.step_state.remaining_len = self.remaining_len
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = (self.day_finished == 2)
        if done:
            reward = self.collected_prize
            print(f' {self.collected_prize}\n\n')
        else:
            reward = None

        return self.step_state, reward, done

    def calculate_len_to_hotel(self) :
        
        self.hotel_xy_expanded = self.hotel_xy.unsqueeze(dim=2).expand(-1,-1,self.problem_size,-1)
        #shape: (batch, hotel, problem, 2)
        # self.hotel_xy_expanded = self.hotel_xy.expand_as(self.node_xy)

        self.node_xy_expanded = self.node_xy.unsqueeze(dim=1).expand(-1, 4,-1,-1)
        #shape: (batch, hotel, problem, 2)

        # Calculate squared differences and sum along the last dimension
        squared_diff = (self.hotel_xy_expanded - self.node_xy_expanded) ** 2
        #shape: (batch, hotel, problem, 2)
        self.distance_sums = torch.sum(squared_diff, dim=3)
        #shape: (batch, hotel, problem)

        # Square root to get the Euclidean distances
        len_to_hotel = torch.sqrt(self.distance_sums)
        #shape: (batch, hotel, problem)
        return len_to_hotel
        
    def calculate_future_len(self) : 
        self.node_xy_expanded = self.node_xy.unsqueeze(dim=1).expand(-1, self.problem_size, -1, -1)
        #shape: (batch, pomo, problem, 2)
        self.current_xy_expanded = self.node_xy_current.unsqueeze(dim=2).expand(-1, -1, self.problem_size, -1)
        #shape: (batch, pomo, problem, 2)

        # Calculate squared differences and sum along the last dimension
        squared_diff = (self.current_xy_expanded - self.node_xy_expanded) ** 2
        #shape: (batch, pomo, problem, 2)
        distance_sums = torch.sum(squared_diff, dim=3)                                  
        #shape: (batch, pomo, problem)

        # Square root to get the Euclidean distances
        future_len = torch.sqrt(distance_sums)
        #shape: (batch, pomo, problem)
        return future_len 
    
    def calculate_two_distance(self) : 
        
        current_node_zero_indexed = self.current_node + self.hotel_number
        last_visited_node_zero_indexed = self.last_visited_node + self.hotel_number
        
        #todo 
        # Expanding the dimensions of current_node to match the dimensions of node_xy for gathering
        current_node_expanded = current_node_zero_indexed.unsqueeze(dim=2).expand(-1, -1, 2)
        #shape: (batch, problem, 2)

        # Using gather to fill the node_xy_current tensor
        self.node_xy_current = torch.gather(self.hotel_node_xy, 1, current_node_expanded)       #@todo
        #shape: (batch, problem, 2)

        # Expanding the dimensions of last_visited_node to match the dimensions of node_xy for gathering
        last_visited_node_expanded = last_visited_node_zero_indexed.unsqueeze(dim=2).expand(-1, -1, 2)
        #shape: (batch, problem, 2)
        
        # Using gather to fill the node_xy_last tensor
        self.node_xy_last_visited = torch.gather(self.hotel_node_xy, 1, last_visited_node_expanded)
        #shape: (batch, problem, 2)

        self.squared_diff = (self.node_xy_last_visited - self.node_xy_current) ** 2
        #shape: (batch, problem, 2)
        self.distance_sums = torch.sum(self.squared_diff, dim=2)
        #shape: (batch, problem)

        # Square root to get the Euclidean distances
        selected_len = torch.sqrt(self.distance_sums)
        #shape: (batch, problem)
        
        return selected_len

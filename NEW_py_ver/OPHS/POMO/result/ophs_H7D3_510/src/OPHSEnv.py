from dataclasses import dataclass
import torch
from OPHSProblemDef import get_random_problems, augment_xy_data_by_8_fold

@dataclass 
class Reset_state : 
    depot_xy: torch.Tensor = None
    # shape: (batch, 2, 2)
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
    
class OPHSEnv: 
    def __init__(self, **env_params) : 
        
        # Const @INIT
        ####################################       
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.hotel_size = env_params['hotel_size']
        self.day_number = env_params['day_number'] 
        
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

    def use_saved_problems(self, filename, device):             
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_prize = loaded_dict['node_prize']
        self.saved_remaining_len = loaded_dict['remaining_len']
        self.saved_index = 0    

    def load_problems(self, batch_size, aug_factor=1) : 
        self.batch_size = batch_size
        
        if not self.FLAG__use_saved_problems:
            depot_xy, node_xy, node_prize, remaining_len_value = get_random_problems(batch_size, self.problem_size, self.hotel_size, self.day_number)
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_prize = self.saved_node_prize[self.saved_index:self.saved_index+batch_size]
            remaining_len_value = self.saved_remaining_len[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size
       
        self.remaining_len_value = remaining_len_value
        self.depot_xy = depot_xy
        self.node_xy = node_xy

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                self.depot_xy = augment_xy_data_by_8_fold(depot_xy)
                self.node_xy = augment_xy_data_by_8_fold(node_xy)
                node_prize = node_prize.repeat(8, 1)
            else:
                raise NotImplementedError
                
        self.depot_node_xy = torch.cat((self.depot_xy, self.node_xy), dim=1)
        # shape: (batch, problem+hotel, 2)
        depot_prize = torch.zeros(size=(self.batch_size, self.hotel_size))                            #@chanage
        # shape: (batch, hotel)
        self.depot_node_prize = torch.cat((depot_prize, node_prize), dim=1)
        # shape: (batch, problem+hotel)
    
        
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.HOTEL_IDX = torch.arange(self.hotel_size)[None, None, :].expand(self.batch_size, self.pomo_size, self.hotel_size)

        self.reset_state.depot_xy = self.depot_xy
        self.reset_state.node_xy = self.node_xy
        self.reset_state.node_prize = node_prize
        
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
        self.ninf_mask_first_step = torch.zeros(size=(self.batch_size, self.hotel_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, hotel, pomo)
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
        self.day_finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.int64)             #@chanage                           #TOP
        #shape: (batch, pomo)
        self.depots_ninf_mask = torch.zeros(size=(self.batch_size, self.hotel_size ,self.pomo_size))             #new 
        #shape: (batch, hotel, pomo)
        self.last_step_mask = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        #shape: (batch, pomo)
        
        self.zeros_to_add = torch.zeros(self.batch_size, 1)  
        # shape: (batch, 1)  

        # self.last_step_node_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size))
        # #shape: (batch, pomo, problem)
        
        
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

        self.len_to_starting_depot, self.len_to_finishing_depot = self.calculate_len_to_depot()
        # shape len_to_start: (batch, pomo)
        # shape len_to_finish: (batch, hotel, pomo)

        self.remaining_len[self.selected_count == 1] = self.remaining_len_value[:, None, 0]      # reset length for the first depot       
        
        if self.selected_count == 1 :                     #first step  
            self.remaining_first_step = self.remaining_len.unsqueeze(1).expand(-1, self.hotel_size, -1)
            # shape: (batch, hotel, pomo)
            len_to_starting_depot_expanded = self.len_to_starting_depot.unsqueeze(1).expand(-1, self.hotel_size, -1)
            # shape: (batch, hotel, pomo)
            self.first_step_len_too_large = (self.remaining_first_step < len_to_starting_depot_expanded + self.len_to_finishing_depot)          #infeasible first step condition
            # shape: (batch, hoetl, pomo)
            self.ninf_mask_first_step = (self.first_step_len_too_large == True).all(dim=1)
            # shape: (batch, pomo)

            self.visited_ninf_flag[self.ninf_mask_first_step.unsqueeze(2).expand_as(self.visited_ninf_flag)] = float('-inf')
            self.finished[self.ninf_mask_first_step] = True

        # Dynamic-2
        ####################################
        # print(f'{self.starting_depot_index}\n{self.finished}')
        # self.at_the_depot = (selected == (self.finishing_depot_index))                   #change
        self.at_the_depot = (selected < self.hotel_size)
        # print(f'\n{selected}\n')
        self.prize_list = self.depot_node_prize[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+hotel)
        self.gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        self.selected_prize = self.prize_list.gather(dim=2, index=self.gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        self.collected_prize += self.selected_prize
    
        selected_len = self.calculate_two_distance()

        self.remaining_len -= selected_len

        self.day_finished[self.selected_count > 1] += self.at_the_depot   
        # if self.selected_count > 1:
        #     self.day_finished[~self.ninf_mask_first_step & (self.day_finished < self.day_number - 1 )] += self.at_the_depot[~self.ninf_mask_first_step & (self.day_finished < self.day_number -1 )] 

        self.remaining_len_value = torch.cat((self.remaining_len_value, self.zeros_to_add), dim=1)  
        # shape: (batch, day + 0~)  

        self.gathering_index_day = self.day_finished.unsqueeze(-1)
        # shape: (batch, pomo, 1)  
        remaining_len_value_expanded = self.remaining_len_value.unsqueeze(1).expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, day)  
        self.remaining_len_value_extracted = remaining_len_value_expanded.gather(2, self.gathering_index_day).squeeze(-1)
        # shape: (batch, pomo)  
        
        # print(f'value: {self.remaining_len_value_extracted}\n{self.selected_node_list}\n{self.day_finished}\n')
        self.remaining_len[self.at_the_depot & (self.day_finished < self.day_number)] = self.remaining_len_value_extracted[self.at_the_depot & (self.day_finished < self.day_number)]      # reset length at the depot       
        
        # self.remaining_len[self.at_the_depot] = self.remaining_len_value_extracted[self.at_the_depot]      # reset length at the depot       
        # self.remaining_len[self.at_the_depot & (self.day_finished < self.day_number + 1)] = 1      # reset length at the depot       
        

        # if (self.at_the_depot).any():
        #     self.visited_ninf_flag[:, :, :4] = True
        #     self.visited_ninf_flag[:, range(self.pomo_size), self.starting_depot_index] = False
        #     self.visited_ninf_flag[:, range(self.pomo_size), self.starting_depot_index + 1] = False
        
        # self.starting_depot_index[self.at_the_depot & (self.day_finished < 3)] += 1
  

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+hotel)
        self.visited_ninf_flag[:, :, :self.hotel_size][~self.at_the_depot & ~self.ninf_mask_first_step] = 0  # depot 2 !!!! is considered unvisited, unless you are AT the depot       #@chanage tu code ghabli ino hazv kon bbin chi mishe

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        
        self.future_len = self.calculate_future_len()             
        # shape: (batch, pomo, problem)
        self.future_len_extended = self.future_len.unsqueeze(dim=1).expand(-1, self.hotel_size, -1, -1)
        # shape: (batch, hotel, pomo, problem)
        self.len_to_finishing_depot_expanded = self.len_to_finishing_depot.unsqueeze(dim=2).expand(-1, -1, self.problem_size, -1)
        # shape: (batch, hotel, pomo, problem)
        self.remaining_len_expanded = self.remaining_len.unsqueeze(1).expand(-1, self.hotel_size, -1)
        # shape: (batch, hotel, pomo)
        self.remaining_len_expanded_2 = self.remaining_len_expanded.unsqueeze(dim=3).expand(-1, -1, -1, self.problem_size)
        # shape: (batch, hotel, pomo, problem)
        
        self.possible_dists = self.len_to_finishing_depot_expanded + self.future_len_extended  
        self.len_too_large = self.remaining_len_expanded_2 + round_error_epsilon < self.possible_dists
        # shape: (batch, hotel, pomo, problem)
                                                                   
        self.len_too_large_extracted = (self.len_too_large == True).all(dim=1)                                                            
        # shape: (batch, pomo, problem)
        self.len_too_large_expanded = torch.cat((torch.zeros_like(self.len_too_large_extracted[:,:,:self.hotel_size], dtype=torch.bool), self.len_too_large_extracted), dim=-1)       #@chanage
        # shape: (batch, pomo, problem+hotel) 
        self.ninf_mask[self.len_too_large_expanded] = float('-inf')
        # shape: (batch, pomo, problem+hotel)



        # depots_ninf_mask_1 = (self.len_too_large == True).all(dim=3).transpose(1, 2)     #hotels masking 
        # # shape: (batch, pomo, hotel)
        self.depots_ninf_mask = self.remaining_len_expanded < self.len_to_finishing_depot
        # shape: (batch, hotel, pomo)
        # self.depots_ninf_mask = depots_ninf_mask_1 + depots_ninf_mask_2.transpose(1, 2)
        # # shape: (batch, pomo, hotel)
        self.ninf_mask[:, :, :self.hotel_size][self.depots_ninf_mask.transpose(1, 2)] = float('-inf')
        # shape: (batch, pomo, problem+hotel)

        # print(f'{self.ninf_mask}')



        # if (self.day_finished >= self.hotel_size-1).any():
        self.last_step_mask[(self.day_finished >= self.day_number-1)] = True        #last day mask
        # print(self.last_step_mask)
        # shape: (batch, pomo)
        # end_expaned = self.last_step_mask.unsqueeze(2).expand(-1, -1, self.hotel_size)
        # # shape: (batch, pomo, hotel)
        self.ninf_mask[:, :, :self.hotel_size][self.last_step_mask] = float('-inf')
        self.ninf_mask[:, :, 1][~self.at_the_depot & ~self.ninf_mask_first_step] = 0  # hotel 1 is considered unvisited, unless you are AT the depot       #@chanage tu code ghabli ino hazv kon bbin chi mishe

        # shape: (batch, pomo, hotel)
        # end_expaned2 = self.last_step_mask.unsqueeze(2).expand(-1, -1, self.problem_size)
        # # shape: (batch, pomo, problem)
        self.last_step_node_mask = self.visited_ninf_flag[:, :, self.hotel_size:].clone()
        self.last_step_node_mask[self.len_too_large[:, 1, :, :]] = float('-inf')                # hotel 1 as finishing depot
        # shape: (batch, pomo, problem)
        # print(self.len_too_large, self.last_step_node_mask)
        self.ninf_mask[:, :, self.hotel_size: ][self.last_step_mask] = self.last_step_node_mask[self.last_step_mask]
        # shape: (batch, pomo, problem)

        # if self.day_finished.any() > 3 & self.day_finished.any() < 5:
        # if self.day_finished.any() == 4:
        
        # print(f'{self.last_step_mask}')
        # print(f'\n\nfutur len: {self.len_to_finishing_depot}\nfutur extended: {self.len_to_finishing_depot_expanded}\n{self.remaining_len}')
        # print(f'\n\nfutur len: {self.future_len}\nlen to finish: {self.len_to_finishing_depot}\npossible dist{self.possible_dists[:, 1, :, :]}\n{self.remaining_len}')
        # print(f'\n\n{self.remaining_len}\n{self.day_finished}\n{self.selected_node_list}')
        # print(f'\n\n{self.remaining_len}')
        # print(f'mask: {self.ninf_mask}\n') 

        # self.ninf_mask[:, :, self.hotel_size:][self.last_step_mask.unsqueeze(-1).expand_as(self.ninf_mask[:, :, self.hotel_size:])] = self.last_step_node_mask[self.last_step_mask]
        # self.ninf_mask[self.last_step_mask, :, self.hotel_size:] = self.last_step_node_mask[self.last_step_mask]
        # self.ninf_mask[self.last_step_mask, :, self.hotel_size:] = self.last_step_node_mask.unsqueeze(1)[self.last_step_mask]
        # self.ninf_mask[self.last_step_mask, self.hotel_size:, :] = self.last_step_node_mask[self.last_step_mask]
        # self.ninf_mask[self.last_step_mask.unsqueeze(1), :, self.hotel_size:] = self.last_step_node_mask[self.last_step_mask]


        # print(self.ninf_mask[:, :, self.hotel_size: ].shape, end_expaned2.shape, self.last_step_node_mask.shape)
        # self.ninf_mask[:, :, self.hotel_size: ][self.len_too_large[:, 2, :, :]] = float('-inf')
        # # print(f'\n\n{self.ninf_mask}\n{self.finished}\n{self.selected_node_list}')
        # time.sleep(1.5)
        # print(f'{self.last_step_mask}\n{self.ninf_mask}')
        # print(f'selected: {self.selected_node_list}\n')
        # print(f'maskkkk: {self.ninf_mask}\n remaining: {self.remaining_len}\n')

        self.newly_finished = (self.ninf_mask == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + self.newly_finished
        # shape: (batch, pomo)

        # do not mask finishing depot for finished episode.
        self.ninf_mask[:, :, 1][self.finished & ~self.ninf_mask_first_step] = 0     # hotel number 1 as finishing node         
        
        # self.ninf_mask[self.ninf_mask_first_step, :] = -float('inf')                           
        self.ninf_mask[:, :, 0][self.finished & self.ninf_mask_first_step] = 0      #first step masking  
       

        # print(f'\n\n{self.remaining_len}\n{self.selected_node_list}\n{self.ninf_mask}')
        # print(f'\n\n{self.remaining_len}')
        # print(f'\n{self.day_finished}')
        # print(f'{selected}')
        # print(f'{self.visited_ninf_flag}')
        # print(f'selected: {selected_len} remaining{self.remaining_len}')
        # time.sleep(2)
        
        # # print(f'\n{self.finished}\n')
        # if self.finished.all() :
        #     self.day_finished += 1
        #     self.finished[:, :] = False
        #     self.remaining_len[:,:] = 1.5 # reset length at the depot
        #     self.depots_ninf_mask[:, :, :] = float('-inf')
        #     self.flag = True
        #     # self.finishing_depot_index[self.finishing_depot_index < 3] += 1
        #     # print(f'################################### End of day = {self.day_finished} ###########################\n\n')
        #     if self.finishing_depot_index < 3:
        #         self.finishing_depot_index += 1

        # if self.flag:
        #     self.depots_ninf_mask[:, :, self.finishing_depot_index] = 0             ################### just added
        #     self.depots_ninf_mask[self.ninf_mask_first_step, :] = -float('inf')                           
        #     # print(f'\n{self.visited_ninf_flag}\n')
        #     self.visited_ninf_flag[:, :, :4] = self.depots_ninf_mask
        #     self.flag = False
            
        # print(f'{self.finishing_depot_index}')
        # self.ninf_mask_first_step_expanded = self.ninf_mask_first_step.unsqueeze(2).expand(-1, -1, 4)
        # self.ninf_mask[:, :, :4] = self.depots_ninf_mask
        # print(f'\n visited:{self.visited_ninf_flag}')
        # time.sleep(1)

        self.step_state.selected_count = self.selected_count
        self.step_state.remaining_len = self.remaining_len
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # print(f'\n{self.step_state.selected_count}{self.step_state.remaining_len}{self.step_state.current_node}{self.step_state.ninf_mask}{self.step_state.finished}\n')
        # returning values
        # done = (self.day_finished >= self.hotel_size-1).all()
        done = self.finished.all()
        if done:
            # if (self.remaining_len < 0).any():
            # print(f'\n##################################################################################\nreward befor chnage: {self.collected_prize}\n')
            self.collected_prize[self.remaining_len < 0] = (self.remaining_len[self.remaining_len < 0]*1000.00) / self.collected_prize[self.remaining_len < 0]

            reward = self.collected_prize
            # print(f'\n########selected nodes######## \n{self.selected_node_list}\n##############reward######### \n{reward}\n{self.depot_node_xy}')    #for testing
            # print(f'########selected nodes######## \n{self.selected_node_list}\n\n########trip length#######\n{self.remaining_len_value}\n\n######final reward#####\n{reward}')    #for testing
        else:
            reward = None

        return self.step_state, reward, done


    def calculate_len_to_depot(self) :
        
        self.starting_depot_xy_expanded = self.depot_xy[:, 0 , :].unsqueeze(1).expand(-1, self.problem_size, -1)
        #shape: (batch, problem, 2)
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
        self.distance_sums2 = torch.sum(squared_diff2, dim=3)
        #shape: (batch, hotel, problem)

        # Square root to get the Euclidean distances
        len_to_starting_depot = torch.sqrt(self.distance_sums)
        #shape: (batch, problem)
        len_to_finishing_depot = torch.sqrt(self.distance_sums2)
        #shape: (batch, hotel, problem)

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

import torch

def get_random_problems(batch_size, problem_size, hotel_size):
 
    # day_number = torch.full((batch_size, 1), 3)
    day_number = torch.randint(2 ,5 , size=(batch_size, 1))


    if problem_size == 32:
        t_max = 0.6 
    elif problem_size == 64:
        t_max = 0.8 
    elif problem_size == 100:
        t_max = 1
    else:
        raise NotImplementedError


    trip_length = (t_max/day_number).unsqueeze(dim=1).expand(-1, day_number.max().item(), -1)
    day_mask = torch.arange(day_number.max().item()).unsqueeze(0) < day_number  # Shape: (1, max_days) < (batch_size, 1)
    trip_length = trip_length * day_mask.expand(batch_size, -1).unsqueeze(-1)  # Shape: (batch_size, max_days, 1)
 

    
    depot_xy = torch.rand(size=(batch_size, hotel_size, 2))
    

    node_xy = torch.rand(size=(batch_size, problem_size, 2))

    mean_generator = torch.rand(batch_size, problem_size) * (95.5 - 4.5) + 4.5  # Mean between 4.5 and 95.5
    deviation_generator = torch.rand(batch_size, problem_size) * (16.5 - 1.5) + 1.5  # Deviation between 1.5 and 16.5
    node_prize = torch.stack((mean_generator, deviation_generator), dim=-1)
    
    return day_number, depot_xy, node_xy , node_prize, trip_length

def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data


# day_number, depot_xy, node_xy , node_prize, trip_length = get_random_problems(2, 32, 5)

# # print(day_number, depot_xy, node_xy , node_prize, trip_length)
# print(day_number, trip_length)

import torch

def get_random_problems(batch_size, problem_size):
 
    if problem_size == 32:

        hotels = torch.tensor([3, 4, 5, 7, 8])
        days = torch.tensor  ([2, 3, 4, 3, 4])

        indices = torch.randint(len(hotels), size=(batch_size,))
        hotel_size = hotels[indices]
        day_number = days[indices].unsqueeze(1)

        # Generate and Padding for depot_xy
        max_hotel_size = hotel_size.max().item()
        depot_xy = torch.rand(batch_size, max_hotel_size, 2)
        mask = torch.arange(max_hotel_size).expand(batch_size, -1) >= hotel_size.unsqueeze(1)
        depot_xy[mask.unsqueeze(-1).expand_as(depot_xy)] = float('nan')  

    else:
        hotel_size = 5
        day_number = 3
        depot_xy = torch.rand(size=(batch_size, hotel_size ,2))

    if problem_size == 32:
        t_max = 0.6 * 1 * day_number
    elif problem_size == 64:
        t_max = 0.8 * 1 * day_number
    elif problem_size == 100:
        t_max = 1 * 1 * day_number
    else:
        raise NotImplementedError

    def distribute_tmax(batch_size, max_days, tmax, day_number):
        mask = torch.arange(max_days).unsqueeze(0) < day_number
        samples = torch.distributions.Dirichlet(torch.ones(max_days) * 100).sample((batch_size,)) * mask
        samples[torch.arange(batch_size), day_number - 1] *= 1 - (torch.rand(batch_size) < 0.3) * 0.1
        return ((samples / samples.sum(dim=1, keepdim=True)) * tmax).unsqueeze(-1)

    trip_length = distribute_tmax(batch_size, day_number.max().item(), t_max, day_number)

    # trip_length = torch.full((batch_size, day_number, 1), t_max)
    # trip_length = 0.7 + (0.7 * torch.rand(batch_size, day_number, 1))    #0.7 , 1.4
    # trip_length = torch.full((batch_size, day_number, 1), 1.5)       #0.7 6

    node_xy = torch.rand(size=(batch_size, problem_size, 2))

    mean_generator = torch.rand(batch_size, problem_size) * (95.5 - 4.5) + 4.5  # Mean between 4.5 and 95.5
    deviation_generator = torch.rand(batch_size, problem_size) * (16.5 - 1.5) + 1.5  # Deviation between 1.5 and 16.5
    node_prize = torch.stack((mean_generator, deviation_generator), dim=-1)
    
    print(depot_xy)

    return hotel_size, day_number, depot_xy, node_xy , node_prize, trip_length

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


# hotel_size, day_number, depot_xy, node_xy , node_prize, trip_length = get_random_problems(8, 32)

# print(hotel_size, day_number, depot_xy, node_xy , node_prize, trip_length)

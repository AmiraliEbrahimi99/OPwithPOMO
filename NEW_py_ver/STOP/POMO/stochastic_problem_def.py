import torch

depot_xy_tensor = torch.tensor([[[50, 50]]])

node_xy_tensor = torch.tensor([[[6, 14], 
                                [9, 31], 
                                [9, 52], 
                                [7, 69], 
                                [15, 92],
                                [34, 15], 
                                [27, 28],
                                [25, 53],
                                [32, 70],
                                [32, 92],
                                [50, 9],
                                [51, 35],
                                [37, 44],
                                [48, 69],
                                [44, 85],
                                [71, 5],
                                [66, 25],
                                [67, 45],
                                [66, 65],
                                [70, 94],
                                [94, 15],
                                [93, 31],
                                [94, 51],
                                [92, 73],
                                [87, 85],
                                [58, 82],
                                [80, 57],
                                [54, 92],
                                [80, 18],
                                [14, 12]]])

node_prize_tensor = torch.tensor([[[17, 22],
                                    [15, 22],
                                    [11, 22],
                                    [11, 20],
                                    [20, 21],
                                    [12, 23],
                                    [11, 19],
                                    [14, 19],
                                    [13, 21], 
                                    [15, 18], 
                                    [14, 336], 
                                    [20, 329], 
                                    [10, 21], 
                                    [15, 17], 
                                    [10, 22], 
                                    [14, 333], 
                                    [11, 296], 
                                    [15, 273], 
                                    [20, 353], 
                                    [20, 288], 
                                    [14, 233], 
                                    [19, 380], 
                                    [15, 276], 
                                    [18, 324], 
                                    [13, 329], 
                                    [16, 24], 
                                    [18, 383], 
                                    [19, 21], 
                                    [17, 225], 
                                    [16, 25]]]).type(torch.float)

remaining_length = float(200)


# Rescale node_xy_tensor to have values between 0 and 1
x_min = node_xy_tensor[:, :, 0].min()
x_max = node_xy_tensor[:, :, 0].max()
y_min = node_xy_tensor[:, :, 1].min()
y_max = node_xy_tensor[:, :, 1].max()

node_xy_rescaled = (node_xy_tensor - torch.tensor([x_min, y_min])) / torch.tensor([x_max - x_min, y_max - y_min])

# Rescale depot_xy_tensor to have values between 0 and 1
depot_xy_rescaled = (depot_xy_tensor - torch.tensor([x_min, y_min])) / torch.tensor([x_max - x_min, y_max - y_min])

# Compute min and max for each array separately
prize_min_0 = node_prize_tensor[:, :, 0].min()
prize_max_0 = node_prize_tensor[:, :, 0].max()
prize_range_0 = prize_max_0 - prize_min_0

prize_min_1 = node_prize_tensor[:, :, 1].min()
prize_max_1 = node_prize_tensor[:, :, 1].max()
prize_range_1 = (prize_max_1 - prize_min_1)

# Rescale each array separately
node_prize_tensor[:, :, 0] = (node_prize_tensor[:, :, 0] - prize_min_0) / prize_range_0 * 2 + 4.5
# node_prize_tensor[:, :, 0] = (node_prize_tensor[:, :, 0]) / 2.2222
node_prize_tensor[:, :, 1] = (node_prize_tensor[:, :, 1] - prize_min_1) / prize_range_1 * 2 + 1.5

# Rescale remain_length based on the same scaling factors used for node coordinates
remaining_length_rescaled = remaining_length / torch.tensor([x_max - x_min, y_max - y_min]).max()

remain_len = remaining_length_rescaled
depot_xy = depot_xy_rescaled
node_xy = node_xy_rescaled
node_prize = node_prize_tensor
# print(depot_xy.shape,node_xy.shape)
# Create a dictionary to store the tensors
saved_dict = {
    'remain_len': remain_len,
    'depot_xy': depot_xy,
    'node_xy': node_xy,
    'node_prize': node_prize 
}

# print(saved_dict)
# Save the dictionary to a file
filename = 'stochastic_2008_problem.pt'
torch.save(saved_dict, filename)

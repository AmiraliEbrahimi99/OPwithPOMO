import torch

# Assume your text file is named "instance.txt" and is in the same directory
with open("instance.txt", "r") as f:
    first_line = f.readline()
    remaining_length = float(first_line.split()[0])
    lines = f.readlines()

# Extract the depot xy coordinates from the second line
depot_xy_list = list(map(float, lines[0].split()[:2]))
depot_xy_tensor = torch.tensor(depot_xy_list)

# Initialize tensors for node xy and prizes
node_xy_list = []
prizes = []

# Iterate over the lines starting from the fourth line
for line in lines[2:]:
    nums = list(map(float, line.split()))
    node_xy_list.append([nums[0], nums[1]])
    prizes.append(nums[2])

# Convert node xy and prizes to tensors
node_xy_tensor = torch.tensor(node_xy_list)
node_prize_tensor = torch.tensor(prizes)

# print("Remaining lenth rescaled:", remaining_length)
# print("Depot xy tensor:", depot_xy_tensor)
# print("Node xy tensor:", node_xy_tensor)
# print("Prizes tensor:", node_prize_tensor)

# Rescale node_xy_tensor to have values between 0 and 1
x_min = node_xy_tensor[:, 0].min()
x_max = node_xy_tensor[:, 0].max()
y_min = node_xy_tensor[:, 1].min()
y_max = node_xy_tensor[:, 1].max()

node_xy_rescaled = (node_xy_tensor - torch.tensor([x_min, y_min])) / torch.tensor([x_max - x_min, y_max - y_min])

# Rescale depot_xy_tensor to have values between 0 and 1
depot_xy_rescaled = (depot_xy_tensor - torch.tensor([x_min, y_min])) / torch.tensor([x_max - x_min, y_max - y_min])

# Rescale node_prize_tensor to have values between 1 and 10
prize_min = node_prize_tensor.min()
prize_max = node_prize_tensor.max()
prize_range = prize_max - prize_min
prize_rescaled = (node_prize_tensor - prize_min) / prize_range * 9 + 1

# Rescale remain_length based on the same scaling factors used for node coordinates
remaining_length_rescaled = remaining_length / torch.tensor([x_max - x_min, y_max - y_min]).max()

# print("\nRemaining lenth rescaled:", remaining_length_rescaled)
# print("Depot xy rescaled tensor:", depot_xy_rescaled)
# print("Node xy rescaled tensor:", node_xy_rescaled)
# print("Prize rescaled tensor:", prize_rescaled)

remain_len = remaining_length_rescaled
depot_xy = depot_xy_rescaled.unsqueeze(0).unsqueeze(1)
node_xy = node_xy_rescaled.unsqueeze(0)
node_prize = prize_rescaled.unsqueeze(0)

# Create a dictionary to store the tensors
saved_dict = {
    'remain_len': remain_len,
    'depot_xy': depot_xy,
    'node_xy': node_xy,
    'node_prize': node_prize 
}

# Save the dictionary to a file
filename = 'op19_instance_tsiligirides_problem_2_budget_15.pt'
torch.save(saved_dict, filename)

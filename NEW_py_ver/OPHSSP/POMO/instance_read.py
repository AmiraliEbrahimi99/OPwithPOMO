import torch

# Assume your text file is named "instance.txt" and is in the same directory
with open("100-160-15-8.ophs", "r") as f:
    first_line = f.readline()

    problem_size = int(first_line.split()[0])
    hotel_size = 2 + int(first_line.split()[1])
    day_number = int(first_line.split()[2])
    lines = f.readlines()

# Extract the remaining_len from the third line
remaining_length = list(map(float, lines[1].split()[:3]))
remaining_length_tensor = torch.tensor(remaining_length)

# Initialize tensors for node xy and prizes
hotel_xy_list = []
node_xy_list = []
prizes = []

# Iterate over the lines starting from the fifth line to hotel_size
for line in lines[3 : (3 + hotel_size)]:
    nums = list(map(float, line.split()))
    hotel_xy_list.append([nums[0], nums[1]])

# # Iterate over the lines starting from the fifth line + hotel_size
for line in lines[(3 + hotel_size) : (1 + hotel_size + problem_size)]:
    nums = list(map(float, line.split()))
    node_xy_list.append([nums[0], nums[1]])
    prizes.append(nums[2])

# # Convert node xy and prizes to tensors
hotel_xy_tensor = torch.tensor(hotel_xy_list)
node_xy_tensor = torch.tensor(node_xy_list)
node_prize_tensor = torch.tensor(prizes)

all_nodes = torch.cat((hotel_xy_tensor, node_xy_tensor), dim=0)

# Rescale tensors to have values between 0 and 1
x_min = all_nodes[:, 0].min()
x_max = all_nodes[:, 0].max()
y_min = all_nodes[:, 1].min()
y_max = all_nodes[:, 1].max()

node_xy_rescaled = (node_xy_tensor - torch.tensor([x_min, y_min])) / torch.tensor([x_max - x_min, y_max - y_min])
hotel_xy_rescaled = (hotel_xy_tensor - torch.tensor([x_min, y_min])) / torch.tensor([x_max - x_min, y_max - y_min])

# Rescale node_prize_tensor to have values between 1 and 9
prize_min = node_prize_tensor.min()
prize_max = node_prize_tensor.max()
prize_range = prize_max - prize_min
prize_rescaled = (node_prize_tensor - prize_min) / prize_range * 8 + 1

# Rescale remain_length based on the same scaling factors used for node coordinates
remaining_length_rescaled = remaining_length_tensor / torch.tensor([x_max - x_min, y_max - y_min]).max()


# print("\nRemaining lenth rescaled:", remaining_length_rescaled)
# print("Hotel xy rescaled tensor:", hotel_xy_rescaled)
# print("Node xy rescaled tensor:", node_xy_rescaled)
# print("Prize rescaled tensor:", prize_rescaled)

remaining_len = remaining_length_rescaled.unsqueeze(0).unsqueeze(2)
hotel_xy = hotel_xy_rescaled.unsqueeze(0)
node_xy = node_xy_rescaled.unsqueeze(0)
node_prize = node_prize_tensor.unsqueeze(0)

# Create a dictionary to store the tensors
saved_dict = {
    'remain_len': remaining_len,
    'hotel_xy': hotel_xy,
    'node_xy': node_xy,
    'node_prize': node_prize 
}

# Save the dictionary to a file
filename = '100-160-15-8.pt'
torch.save(saved_dict, filename)
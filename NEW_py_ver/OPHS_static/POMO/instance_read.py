import torch
import os
import glob
import torch

def process_ophs_file(filepath, output_dir):
    with open(filepath, "r") as f:
        first_line = f.readline()

        problem_size = int(first_line.split()[0]) - 2
        hotel_size = 2 + int(first_line.split()[1])
        day_number = int(first_line.split()[2])
        lines = f.readlines()

    # Extract the remaining_len from the third line
    remaining_length = list(map(float, lines[1].split()))
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
    for line in lines[(3 + hotel_size) : (3 + hotel_size + problem_size)]:
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

    # Rescale node_prize_tensor to have values between 0.02 and 0.99
    prize_min = node_prize_tensor.min()
    prize_max = node_prize_tensor.max()
    prize_rescaled = (node_prize_tensor - prize_min) / (prize_max - prize_min) * (0.99 - 0.02) + 0.02

    # Rescale remain_length based on the same scaling factors used for node coordinates
    remaining_length_rescaled = remaining_length_tensor / torch.tensor([x_max - x_min, y_max - y_min]).max()

    # print("\nRemaining lenth rescaled:", remaining_length_rescaled)
    # print("Hotel xy rescaled tensor:", hotel_xy_rescaled)
    # print("Node xy rescaled tensor:", node_xy_rescaled)
    # print("Prize rescaled tensor:", prize_rescaled)

    remaining_len = remaining_length_rescaled.unsqueeze(0)
    hotel_xy = hotel_xy_rescaled.unsqueeze(0)
    node_xy = node_xy_rescaled.unsqueeze(0)
    node_prize = prize_rescaled.unsqueeze(0)

    # Create a dictionary to store the tensors
    saved_dict = {
        'remain_len': remaining_len,
        'hotel_xy': hotel_xy,
        'node_xy': node_xy,
        'node_prize': node_prize 
    }

    # Save the dictionary to a file
    output_filename = os.path.join(output_dir, os.path.basename(filepath).replace('.ophs', '.pt'))
    torch.save(saved_dict, output_filename)


def process_ophssp_file(filepath, output_dir):
    with open(filepath, "r") as f:
        first_line = f.readline()

        problem_size = int(first_line.split()[0]) - 2
        hotel_size = 2 + int(first_line.split()[1])
        day_number = int(first_line.split()[2])
        lines = f.readlines()

    # Extract the remaining_len from the third line
    remaining_length = list(map(float, lines[1].split()))
    remaining_length_tensor = torch.tensor(remaining_length)

    # Initialize tensors for node xy and prizes
    hotel_xy_list = []
    node_xy_list = []
    mean = []
    variance = []

    # Iterate over the lines starting from the fifth line to hotel_size
    for line in lines[3 : (3 + hotel_size)]:
        nums = list(map(float, line.split()))
        hotel_xy_list.append([nums[0], nums[1]])

    # # Iterate over the lines starting from the fifth line + hotel_size
    for line in lines[(3 + hotel_size) : (3 + hotel_size + problem_size)]:
        nums = list(map(float, line.split()))
        node_xy_list.append([nums[0], nums[1]])
        mean.append(nums[2])
        variance.append(nums[3])

    # # Convert node xy and prizes to tensors
    hotel_xy_tensor = torch.tensor(hotel_xy_list)
    node_xy_tensor = torch.tensor(node_xy_list)
    mean_tensor = torch.tensor(mean)
    vaiance_tensor = torch.tensor(variance)
    deviation_tensor = torch.sqrt(vaiance_tensor)

    all_nodes = torch.cat((hotel_xy_tensor, node_xy_tensor), dim=0)

    # Rescale tensors to have values between 0 and 1
    x_min = all_nodes[:, 0].min()
    x_max = all_nodes[:, 0].max()
    y_min = all_nodes[:, 1].min()
    y_max = all_nodes[:, 1].max()

    node_xy_rescaled = (node_xy_tensor - torch.tensor([x_min, y_min])) / torch.tensor([x_max - x_min, y_max - y_min])
    hotel_xy_rescaled = (hotel_xy_tensor - torch.tensor([x_min, y_min])) / torch.tensor([x_max - x_min, y_max - y_min])

    # Rescale node_prize_tensor to have values between 0.02 and 0.99
    mean_min = mean_tensor.min()
    mean_max = mean_tensor.max()
    mean_rescaled = (mean_tensor - mean_min) / (mean_max - mean_min) * (0.955 - 0.045) + 0.045

    deviation_min = deviation_tensor.min()
    deviation_max = deviation_tensor.max()
    deviation_rescaled = (deviation_tensor - deviation_min) / (deviation_max - deviation_min) * (0.16 - 0.015) + 0.015

    remaining_length_rescaled = remaining_length_tensor / torch.tensor([x_max - x_min, y_max - y_min]).max()

    # print("\nRemaining lenth rescaled:", remaining_length_rescaled)
    # print("Hotel xy rescaled tensor:", hotel_xy_rescaled)
    # print("Node xy rescaled tensor:", node_xy_rescaled)
    # print("mean rescaled tensor:", mean_rescaled)
    # print("deviation rescaled tensor:",deviation_rescaled)

    remaining_len = remaining_length_rescaled.unsqueeze(0)
    hotel_xy = hotel_xy_rescaled.unsqueeze(0)
    node_xy = node_xy_rescaled.unsqueeze(0)
    mean = mean_rescaled.unsqueeze(0)
    deviation = deviation_rescaled.unsqueeze(0)

    saved_dict = {
        'remain_len': remaining_len,
        'hotel_xy': hotel_xy,
        'node_xy': node_xy,
        'mean': mean, 
        'deviation': deviation 
    }

    # Save the dictionary to a file
    output_filename = os.path.join(output_dir, os.path.basename(filepath).replace('.ophs', '.pt'))
    torch.save(saved_dict, output_filename)


def ophssp_create_new_file(filepath, output_dir):
    with open(filepath, "r") as f:
        first_line = f.readline()

        problem_size = int(first_line.split()[0]) - 2
        hotel_size = 2 + int(first_line.split()[1])
        day_number = int(first_line.split()[2])
        lines = f.readlines()

    # Extract the remaining_len from the third line
    remaining_length = list(map(float, lines[1].split()))

    # Initialize tensors for node xy and prizes
    hotel_xy_list = []
    node_xy_list = []
    prizes = []
    hotel_prizes = []

    # Iterate over the lines starting from the fifth line to hotel_size
    for line in lines[3 : (3 + hotel_size)]:
        nums = list(map(float, line.split()))
        hotel_xy_list.append([nums[0], nums[1]])
        hotel_prizes.append(nums[2])

    # Iterate over the lines starting from the fifth line + hotel_size
    for line in lines[(3 + hotel_size) : (3 + hotel_size + problem_size)]:
        nums = list(map(float, line.split()))
        node_xy_list.append([nums[0], nums[1]])
        prizes.append(nums[2])

    hotel_prizes_tensor = torch.tensor(hotel_prizes)
    node_prize_tensor = torch.tensor(prizes)

    # Rescale node_prize_tensor to have values between 0.02 and 0.99
    prize_min = node_prize_tensor.min()
    prize_max = node_prize_tensor.max()
    prize_rescaled = ((node_prize_tensor - prize_min) / (prize_max - prize_min) * (95.5 - 4.5) + 4.5)

    coefficient_of_variation = (torch.rand(problem_size)) * (0.5 - 0.1) + 0.1
    deviation = prize_rescaled * coefficient_of_variation 
    rescaled_deviation = ((deviation - 4.5) * (prize_max - prize_min) / (95.5 - 4.5)) + prize_min
    rescaled_variance = (rescaled_deviation) ** 2

    node_variance = torch.cat((hotel_prizes_tensor, rescaled_variance), dim=0)
    # print(f'\n{node_prize_tensor}\n{rescaled_deviation}\n{coefficient_of_variation} \n')

    modified_lines = [first_line]  # Include the first line at the beginning
    modified_lines.extend(lines[:3])  # Keep the first 3+hotel_size lines unchanged

    for i, line in enumerate(lines[(3) : (3 + hotel_size + problem_size)]):
        nums = line.split()
        x = float(nums[0]) if "." in nums[0] else int(nums[0])
        y = float(nums[1]) if "." in nums[1] else int(nums[1])
        z = int(nums[2])  # Always an integer

        formatted_line = f"{x:<7} {y:<7} {z:<4d} {node_variance[i].item():<7.2f}\n"
        modified_lines.append(formatted_line)

    modified_lines.append('-------------------------------------------------------------------\n')
    # Save the modified content
    output_filepath = os.path.join(output_dir, os.path.basename(filepath))
    with open(output_filepath, "w") as f:
        f.writelines(modified_lines)

###########################################################################################################################

root_dir = "Instances/raw_OPHS_instances"
output_dir = "Instances/OPHS_pt"
# output_dir = "Instances/OPHSSP"
os.makedirs(output_dir, exist_ok=True)

for filepath in glob.glob(os.path.join(root_dir, "**/*.ophs"), recursive=True):
    process_ophs_file(filepath, output_dir)

################ test #############################

# ophssp_create_new_file(root_dir, output_dir)
# process_ophssp_file(root_dir, output_dir)
# process_ophssp_file(root_dir, output_dir)
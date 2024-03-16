import os
import json

def ensure_file_exists(filepath, default_content):
    """
    Ensure that a file exists at the specified filepath.
    If the file does not exist, create it with the default content.
    
    :param filepath: The full path to the file to check or create.
    :param default_content: A Python dictionary that will be saved as JSON if the file does not exist.
    """
    dirname = os.path.dirname(filepath)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if not os.path.exists(filepath):
        with open(filepath, 'w') as file:
            json.dump(default_content, file, indent=4)

# Example usage within your main function or initialization logic
def main():
    # Example file paths and default contents
    log_image_style_paths = [
        ('/home/soroush/nita_pro/tsp/POMO/NEW_py_ver/utils/log_image_style/style_OP_5.json', {"default": "content"}),
        ('/home/soroush/nita_pro/tsp/POMO/NEW_py_ver/utils/log_image_style/style_loss_1.json', {"default": "content"})
    ]
    
    for path, default_content in log_image_style_paths:
        ensure_file_exists(path, default_content)



if __name__ == "__main__":
    main()

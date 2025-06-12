# POMO for CO

We provide code for three CO (Combinatorial Optimization) problems, with some variations for each:

- Orienteering Problem (OP)  
- Team Orienteering Problem  
  - Deterministic prize (TOP)  
  - Stochastic prize (STOP)  
- Orienteering Problem with Hotel Selection  
  - Normal version (OPHS)  
  - OPHS with predefined (fixed) hotel order (OPHS_static)  
  - OPHS with predefined (fixed) hotel order and variable day number (OPHS_static_DD)  

> Note: Each variant of OPHS can be trained and used for both stochastic and deterministic prize types.

### Changes from original POMO

- Replaced standard attention with **Flash Attention**
- Modified input dimensions to match each problem and its prize type
- Updated the code for compatibility with 2025 GPUs and recent libraries
- Added a new test file for OPHS variants to evaluate trained models in combination with a heuristic method
- Added `instance_read.py` for transforming `.txt` input files into `.pt` format with improved readability

### Basic Usage

To train a model for any of the problems, run `train.py`.  
You can modify parameters such as problem size, training settings, and prize type directly in `train.py`.

To test a model, run `test.py`.  
You can specify the model inside `test.py`. It can be set to use either the saved model provided in the `result` folder or any custom-trained model.

To test a model for OPHS variants, run `Inference_main.py`.  
You can set parameters for the heuristic algorithm, such as the number of repetitions, augmentation count, and more, within that file.

### Used Libraries

All required packages are listed in the `requirements.txt` file.

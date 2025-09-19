# Navigation DQN Project

This project trains an agent to navigate and collect bananas in a large, square world using Deep Q-Network (DQN) and Rainbow DQN reinforcement learning algorithms.

## Project Details

### Environment Description

The navigation environment is a Unity ML-Agents environment where an agent moves in a large, square world to collect bananas.

### State Space
- **Size**: 37-dimensional continuous state space
- **Contains**: Agent's velocity, along with ray-based perception of objects around the agent's forward direction
- **Type**: Vector observation (no visual input required)

### Action Space
- **Size**: 4 discrete actions
- **Actions Available**:
  - `0` - Move forward
  - `1` - Move backward
  - `2` - Turn left
  - `3` - Turn right

### Reward Structure
- **+1** reward for collecting a yellow banana
- **-1** reward for collecting a blue banana
- **0** reward for all other actions

### Solving Criteria
The environment is considered solved when the agent achieves an **average score of +13 over 100 consecutive episodes**.

### Agents Implemented
This project includes two different DQN implementations:
1. **Standard DQN Agent** - Classic Deep Q-Network with experience replay and target network
2. **Rainbow DQN Agent** - Advanced DQN variant incorporating multiple improvements including

## Getting Started

### Prerequisites
- Python 3.6 or higher
- PyTorch
- Unity ML-Agents toolkit
- NumPy, Matplotlib, and other scientific computing libraries

### Step 1: Set Up Python Environment

Create and activate a new conda environment with Python 3.6:

```bash
conda create --name drlnd python=3.6 -y
conda activate drlnd
```

### Step 2: Install PyTorch

Install a compatible PyTorch version (the original PyTorch 0.4.0 is no longer available):

```bash
pip install torch==1.7.1 torchvision==0.8.2
```

**Note**: This installs PyTorch 1.7.1 which is compatible with the Unity ML-Agents package, though newer than the originally specified version.

### Step 3: Install Unity ML-Agents

Clone the Udacity Value-based Methods repository and install the Unity agents:

```bash
git clone https://github.com/udacity/Value-based-methods.git
cd Value-based-methods/python
pip install . --no-deps
pip install docopt pandas pytest pyyaml scipy matplotlib tqdm jupyter protobuf==3.5.2 grpcio==1.11.0
```

**Note**: We use `--no-deps` to avoid PyTorch version conflicts, then install the required dependencies separately. You can verify the installation by running:
```bash
python -c "from unityagents import UnityEnvironment; print('UnityEnvironment imported successfully')"
```

### Step 3: Download the Unity Environment

Download the appropriate Unity environment for your operating system:

- **Linux**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- **Mac OSX**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- **Windows (32-bit)**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- **Windows (64-bit)**: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

**For AWS users**: If training on AWS without a virtual screen, use [this headless version](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip).

Extract the downloaded file and place it in the appropriate directory relative to your project.

### Step 4: Verify Installation

All required dependencies should now be installed. You can verify everything is working by testing the imports:
```bash
python -c "import torch; import numpy as np; from unityagents import UnityEnvironment; print('All imports successful! PyTorch version:', torch.__version__)"
```

## Instructions

### Training the Agent

To train both DQN and Rainbow DQN agents and compare their performance:

1. **Update the environment path** in `main.py`:
   ```python
   env = UnityEnvironment(file_name="path/to/your/Banana.x86_64")
   ```

2. **Activate the environment and run the training script**:
   ```bash
   conda activate drlnd
   python main.py
   ```

### Output Files

After training, you'll find:
- `dqn_checkpoint.pth` - Trained DQN model weights
- `rainbow_dqn_checkpoint.pth` - Trained Rainbow DQN model weights
- `dqn_training.png` - DQN training progress plot
- `rainbow_training.png` - Rainbow DQN training progress plot
- `comparison.png` - Side-by-side comparison of both agents


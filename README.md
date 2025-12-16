# Reinforcement Learning Walker Project

This project implements Fall Recovery of Bipedal Locomotion via Modular Task 
Decomposition and Force-Decay Curriculum for a Walker environment using Unity ML-Agents.

## Prerequisites

- **Python 3.10.10**
- **Unity**

## Setup Information

### Python Virtual Environment
It is **strict requirement** to use a Python virtual environment (`venv`) for training to prevent library collisions and ensure the correct Python version is used.

1.  **Activate the Virtual Environment**:
    ```powershell
    .\venv\Scripts\activate
    ```
    *If the `venv` folder does not exist, create it first: `python -m venv venv`*

2.  **Install Dependencies**:
    Once the environment is activated, install the required packages:
    ```powershell
    python.exe -m pip install --upgrade pip
    pip3 install torch~=2.2.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install mlagents==1.1.0
    ```

## Training Commands

Run the following commands from the root directory of the project (ensure your `venv` is activated).

### Train Method 1 Agent (Combined Stand Up & Walking)
This uses the curriculum configuration to train the Method 1 agent. 
(This config can be used to train the Method 1 agent and the Recovery Expert in Method 2 with Scene changes inside Unity)
```powershell
mlagents-learn .\Configs\ppo\WalkerCurriculum.yaml --run-id=CombinedStandUpWalking
```

### Train Walking Model (Expert)
This uses the standard Walker configuration to train the walking model.
```powershell
mlagents-learn .\Configs\ppo\Walker.yaml --run-id=WalkerExpert
```

## Scripts Overview

All training and evaluation scripts are located in `Project\Assets\ML-Agents\Examples\Walker\Scripts`. Key files include:

-   **WalkerAgent.cs**: The core agent script for the Walker.
-   **WalkerAgentCurriculum.cs**: Handles curriculum-based training logic (used for Method 1 and Recovery Expert in Method 2).
-   **WalkerAgentFast.cs**: Misc Evaluation Script
-   **ExpertWalker.cs**: Implementation for the expert walker.

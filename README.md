# CRG-STL: Signal Temporal Logic for Robotic Manipulation

This repository contains the implementation for a thesis project focused on using Signal Temporal Logic (STL) to guide and verify robotic manipulation tasks. It features a Safe Funnel Controller based on Potential Fields and a Reinforcement Learning (RL) pipeline using Soft Actor-Critic (SAC) with Behavior Cloning.

The primary environment used is `PandaPickAndPlace-v3` from `panda_gym`.

---
## Project Overview

The goal of this project is to satisfy complex temporal logic specifications, such as:
> "Eventually approach the object, then eventually grasp it, and finally move it."

The repository explores two main approaches:
1.  **Safe Funnel Controller:** A potential field-based controller that navigates the robot through specific phases defined by the STL specification.
2.  **Reinforcement Learning with Behavior Cloning:** An SAC agent trained to satisfy the STL specification, bootstrapped with expert demonstrations collected from the Safe Funnel Controller.

---
## Installation

This project is designed to run using Docker and Docker Compose.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/gawtamcr/crgstl.git
    cd crgstl
    ```
2.  **Build the Docker environment:**
    ```bash
    docker build -t crgstl .
    ```
3.  **Run the Docker container:**
    - Modify the compose file between run_PF_controller.py and train_RL_with_BC.py
    ```bash
    docker compose up
    ```
4. **To visualize the results, in a separate terminal:**
    ```bash
    tensorboard --logdir models/training/sac_crgstl_tensorboard/
    ```
--- 

This script will:
1.  Initialize the environment with an STL wrapper.
2.  Collect expert demonstrations using the `SafeFunnelController`.
3.  Pre-fill the SAC replay buffer.
4.  Train the agent for a specified number of timesteps.
5.  Save the trained model to `../models/training/`.

## Project Structure

- **`src/`**: Main source code directory.
    - **`controller/`**: Contains the `SafeFunnelController` logic.
    - **`behavior_cloning/`**: Scripts for collecting expert data and gym wrappers.
    - **`common/`**: Shared utilities, including STL predicates and planners.

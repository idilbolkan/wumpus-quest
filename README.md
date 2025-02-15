# Assignment 2.3: Wumpus Quest

This project implements an AI agent for Wumpus Quest, a Markov Decision Process (MDP)-based game where the agent navigates a cave, collects gold, and returns safely while avoiding hazards such as pits, Wumpuses, and bridges. The agent must learn an optimal policy for decision-making using policy iteration, a reinforcement learning technique.

The agent operates in an uncertain environment where movements may not always succeed as intended, and some actions (such as fighting or crossing bridges) depend on skill-based probability rolls.


## Table of Contents

- [Introduction](#introduction)
  - Key Features
- [Setup](#setup)
  - Repository content
  - How to run the code
  - Used libraries
- [Code Structure](#code-structure)
- [Self Evaluation and Design Decisions](#design-decision)
- [Output Format](#output-format)

## Introduction

### Key Features 
- **Markov Decision Process (MDP) Framework:** The agent models the game as an MDP and applies policy iteration to determine the best strategy.
- **State Representation:** The state consists of the agent’s position and gold collected in the cave.
- **Reward System:** The agent receives rewards for collecting gold and penalties for movement costs and hitting walls.
- **Skill Allocation:** The agent can allocate skill points for agility (crossing bridges) and fighting (defeating the Wumpus).
- **Deterministic Transition Model:** The movement follows deterministic rules, but skill-dependent actions use probability-based success/failure.
- **Grid-Based Navigation:** The cave map is represented as a 2D grid where different symbols (S, G, P, W, B, X) define terrain and obstacles.

## Setup
### This repository contains:
 1) **`example.py`**: Core implementation of navigation logic
 2) **`client.py`**: A Python implementation of the AISysProj server protocol
 3) **agent-configs/**: Configuration files for different game scenarios.

### How to run the code: 
1) **`example.py`**, **`client.py`** and **agent-configs/** folder must all be on the same folder
2) Run the **cmd** on the current path.
3) Run the following command **python example.py agent-configs/env-*.json**

### Used libraries:
**_random:_**
Used for random dice rolls in skill checks.
**_itertools:_**
Helps in generating state subsets for MDP state representation.
**_logging:_**
Tracks runtime events and debugging information during environment interactions.


## **Code Overview**

### **1. Imports and Constants**
```python
import random
import logging
from itertools import chain, combinations

# Constants
GAMMA = 0.95  # Discount factor
EPSILON = 1e-6  # Convergence threshold
ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST", "EXIT"]
```
- **Imports**:
  - `random`: Used for random choices (e.g., initial policy).
  - `logging`: For logging information during execution.
  - `sys`: For system-related operations (e.g., command-line arguments).
  - `itertools`: Provides utility functions like `chain` and `combinations` for generating subsets.
  - `client.run`: Assumed to be a function provided by the server to run the agent.
- **Constants**:
  - `GAMMA`: Discount factor for future rewards.
  - `EPSILON`: Threshold for convergence in Policy Iteration.
  - `ACTIONS`: List of possible actions the agent can take.

---

### **2. Helper Functions**
#### **`powerset(iterable)`**
```python
def powerset(iterable):
  #...
```
- Generates all possible subsets of collected gold (e.g., [G1, G2] → {(), (G1), (G2), (G1,G2)}).
- Represents states in the MDP as (position, frozenset(gold_collected))
- Used to represent all possible states of collected gold.

#### **`parse_map(raw_map)`**
```python
def parse_map(raw_map):
  #...
```
- Parses the raw map string into a 2D grid.
- Extracts the positions of gold (`G`) and the starting position (`S`).

#### **`get_walkable_positions(grid)`**
```python
get_walkable_positions(grid):
  #...
```
- Returns a list of all walkable positions (non-wall cells) in the grid.

#### **`is_position_walkable(position, grid)`**
```python
is_position_walkable(position, grid):
  #...
```
- Checks if a position is within the grid bounds and not a wall.

#### **`get_reward(position, action, next_position, gold_collected, gold_locations, start_pos)`**
```python
get_reward(position, action, next_position, gold_collected, gold_locations, start_pos):
  #...
```
- Bellman Equation Context: Computes immediate reward R(s,a,s') for value updates:
  - Small penalty for each step (-0.01).
  - Additional penalty for hitting a wall (-0.1).
  - Reward for collecting gold (+1).
  - Reward for exiting the cave with collected gold (+len(gold_collected)).

#### **`get_possible_next_positions(position, action, grid)`**
```python
get_possible_next_positions(position, action, grid):
  #...
```
- Returns all possible next positions for a given action.
- For `EXIT`, the agent can only stay at the current position if it's at the stairs (`S`).

#### **`get_transition_prob(position, action, next_position, grid)`**
```python
get_transition_prob(position, action, next_position, grid):
  #...
```
- Computes the transition probability for a given action and next position.
- Since the environment is deterministic, the probability is either `1.0` or `0.0`.

---

### **3. Policy Iteration**
```python
policy_iteration(grid, gold_locations, start_pos):
  #...
```
  - Bellman Equation: Explicitly used in value updates during policy evaluation.
  - Alternates between **Policy Evaluation** (updating the value function) and **Policy Improvement** (updating the policy).
  - Stops when the policy stabilizes (no further changes).

---

### **4. Bridge Handling**
```python
def cross_bridge(agility_skill):
  #...
```
 - Rolls agility_skill dice, takes top 3.
 - Success if sum ≥ 12.
 - Called automatically when agent enters a bridge (B).

---

### **5. Agent Function**
```python
def agent_function(request_data, request_info):
  #...
```
- Skill Allocation: Prioritizes agility for bridge survival.
- State Tracking:
    - current_position: Updated from server’s history.
    - gold_collected: Tracked via collected-gold-at in outcomes.
- Computes the optimal policy using Policy Iteration.
- Follows the policy to move the agent and collect gold.
- Returns the chosen action.

---

### **5. Main Function**
- Runs the agent using the `client.run` function.
- Sets up logging and runs the agent for a maximum of 1000 iterations.

---

## **How It Works**
1. The agent receives the game state (map, history, etc.) from the server.
2. It parses the map to identify walkable positions, gold locations, and the starting position.
3. Using **Policy Iteration**, the agent computes the optimal policy for maximizing rewards.
4. The agent follows the policy to navigate the grid, collect gold, and exit the cave.
5. The agent returns the chosen action to the server.

---

## Self Evaluation and Design Decisions

## Output Format
The code currently works for env-1, env-2 

### env-1
<img width="291" alt="image" src="https://github.com/user-attachments/assets/3ca1d048-4418-43f3-90f0-78662cf7ff29" />

### env-2
<img width="308" alt="image" src="https://github.com/user-attachments/assets/61436460-065d-40ee-b96d-247041dee9ed" />


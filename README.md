# Assignment 2.3: Wumpus Quest

This project implements an AI agent for Wumpus Quest, a Markov Decision Process (MDP)-based game where the agent navigates a cave, collects gold, and returns safely while avoiding hazards such as pits and bridges. The agent uses policy iteration, a reinforcement learning technique, to learn an optimal decision-making policy.

The agent operates in an uncertain environment where movements may not always succeed as intended, and some actions (such as crossing bridges) depend on skill-based probability rolls.

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
- **State Representation:** The state consists of the agent’s position and gold collected in the cave `(position, frozenset(gold_collected))`.
- **Reward System:** The agent receives rewards for collecting gold and penalties for movement costs and hitting walls.
- **Skill Allocation:** The agent can allocate skill points for agility (crossing bridges) and fighting (defeating the Wumpus).
- **Deterministic Transition Model:** The movement follows deterministic rules, but skill-dependent actions use probability-based success/failure.
- **Grid-Based Navigation:** The cave map is represented as a 2D grid where different symbols (S, G, P, B, X) define terrain and obstacles.

## Setup
### This repository contains:
 1) **`example.py`**: Core implementation of navigation logic
 2) **`client.py`**: A Python implementation of the AISysProj server protocol
 3) **agent-configs/**: Configuration files for different game scenarios.

### How to run the code: 
1) Ensure **`example.py`**, **`client.py`** and **agent-configs/** folder are in the same directory.
2) Run the **cmd** in the directory
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
GAMMA = 0.99  # Discount factor
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
- This function returns a list of all walkable positions in the grid. A position is walkable if it is not a wall (X). The positions are stored as (column, row) tuples..

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
  - Small penalty for each step (-0.1).
  - Additional penalty for hitting a wall (-0.5).
  - Reward for collecting gold (+10).
  - Reward for exiting the cave with collected gold (+len(gold_collected)* 10).

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
  - **Initialization:**
    -  The policy is initialized randomly for each state.
    -  The value function `V` is initialized to 0 for all states.
  - **Policy Evaluation:**
    - The value function is updated iteratively using the **Bellman equation** until convergence.
  - **Policy Improvement:**
    - The policy is updated to choose the action that maximizes the expected value.
  - **Termination:**
    - The process stops when the policy stabilizes (no further changes).
---

### **4. Bridge Handling**
```python
def get_safe_next_position(current_position, action, grid, skill_points):
  #...
```
This function determines the next position for a given action, considering skill-based obstacles like bridges:
 - For movement actions, it checks if the new position is walkable.
 - For bridges (B), it performs an agility check using dice rolls:
    - The agent rolls dice equal to its agility skill and sums the top 3 rolls.
    - If the sum is ≥ 12, the bridge is crossed successfully.
    - If the sum is < 12, the agent retries the roll.
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

In this project, the challenges we faced was ensuring the agent could navigate the environment safely, particularly when dealing with pits ("P") and bridges ("B"). Both elements introduced unique complexities: pits were deadly and had to be avoided at all costs, while bridges required successful agility checks to cross. Below, we outline our attempts to solve these problems, the challenges we encountered, and the final solutions we implemented.

### 1. Pit Problem:
---
- **Initial Approach:**
Our first step was to ensure the agent could detect and avoid pits. We modified the is_position_walkable function to exclude pits from walkable positions. This prevented the agent from moving directly into pits but did not account for neighboring pits or unsafe paths.
```python
def is_position_walkable(position, grid):
    col, row = position
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        cell = grid[row][col]
        if cell not in ('X', 'P'):  # Exclude walls and pits
            return True
    return False
```
**Issue:** <br>
While this prevented direct movement into pits, the agent could still end up near pits, leading to unsafe paths.

- **Detecting Neighboring Pits:**
To address this, we introduced a leads_to_pit function to check if an action would move the agent into a pit. This allowed us to filter out dangerous actions before the agent executed them.
```python
def leads_to_pit(position, action, grid):
    directions = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
    dc, dr = directions[action]
    new_col = position[0] + dc
    new_row = position[1] + dr
    if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
        return grid[new_row][new_col] == 'P'
    return False
```
**Issue:** <br>
This approach improved pit detection but was not consistently integrated into the agent's decision-making process, leading to occasional unsafe actions.

- **Policy Iteration with Pit Avoidance:**
We attempted to integrate pit avoidance into the policy iteration algorithm. During policy improvement, we skipped actions that led to pits, ensuring the agent prioritized safe paths.
```python
for action in ACTIONS:
    if leads_to_pit(position, action, grid):
        continue  # Skip dangerous actions
``` 
**Issue:** <br>
While this reduced the likelihood of the agent falling into pits, it did not eliminate the problem entirely, especially in edge cases where safe actions were limited.

- **Final Outcome:**
Despite our efforts, the agent occasionally fell into pits in certain scenarios. The primary challenges were:
  - Consistently detecting and avoiding pits.
  - Integrating pit avoidance into the agent's decision-making process effectively.
  - Handling edge cases where safe actions were unavailable.
We implemented a fallback mechanism where the agent defaults to a safe action (e.g., staying in place) if no safe moves are available.
```python
safe_actions = [a for a in ACTIONS if not leads_to_pit(current_position, a, grid)]
action = random.choice(safe_actions) if safe_actions else "NORTH"
```

### 2. Bridge Problem:
---
- **Initial Approach:**
Initially, we treated bridges as walkable cells, allowing the agent to consider them in its pathfinding. However, this ignored the probabilistic nature of bridge crossings, leading to frequent failures.
```python
def is_position_walkable(position, grid):
    return grid[position[1]][position[0]] not in ('X', 'P')
```
**Issue:** <br>
The agent attempted to cross bridges without considering the risk of failure, resulting in poor performance.

- **Adding Agility Checks:**
To address this, we introduced agility checks in the `get_safe_next_position` function. The agent would roll dice based on its agility skill to determine if it could successfully cross a bridge.
```python
if next_cell == 'B':
    agility_skill = skill_points.get("agility", 0)
    dice_rolls = [random.randint(1, 6) for _ in range(agility_skill)]
    score = sum(sorted(dice_rolls, reverse=True)[:3])
    if score >= 12:
        return new_position  # Successful crossing
    else:
        return current_position  # Failed crossing
```
**Issue:** <br>
This approach improved pit detection but was not consistently integrated into the agent's decision-making process, leading to occasional unsafe actions.

- **Policy Iteration with Pit Avoidance:**
We attempted to integrate pit avoidance into the policy iteration algorithm. During policy improvement, we skipped actions that led to pits, ensuring the agent prioritized safe paths.
```python
for action in ACTIONS:
    if leads_to_pit(position, action, grid):
        continue  # Skip dangerous actions
```
**Issue:** <br>
This approach sometimes led to infinite loops if the agent repeatedly failed the agility check.

- **Final Solution: Incorporating Probabilities:**
To address the probabilistic nature of bridge crossings, we updated the get_transition_prob function to account for the success and failure probabilities of crossing a bridge. This allowed the agent to make informed decisions by considering the risk of failing to cross a bridge. However, despite our efforts, the agent still occasionally falls off bridges due to the inherent randomness of the agility checks. Additionally, in some environments, the agent struggles to find safe paths that avoid both pits and bridges, leading to suboptimal performance.
```python
def get_transition_prob(position, action, next_position, grid):
    if action == "EXIT":
        return 1.0 if grid[position[1]][position[0]] == 'S' and next_position == position else 0.0

    directions = {"NORTH": (0, -1), "SOUTH": (0, 1), "EAST": (1, 0), "WEST": (-1, 0)}
    dc, dr = directions[action]
    new_pos = (position[0] + dc, position[1] + dr)

    if not is_position_walkable(new_pos, grid):
        return 1.0 if next_position == position else 0.0

    # Handle bridge crossings
    if grid[new_pos[1]][new_pos[0]] == 'B':
        p = compute_bridge_success_probability(skill_points.get("agility", 0))  # Probability of success
        return p if next_position == new_pos else (1 - p) if next_position == position else 0.0
    else:
        return 1.0 if next_position == new_pos else 0.0
```
**Explanation:** <br>
- **Bridge Handling:** If the next position is a bridge ('B'), the function calculates the probability of successfully crossing it (p) based on the agent's agility skill. If the agent succeeds, it moves to the bridge cell (new_pos). If it fails, it stays in the current position.

- **Non-Bridge Cells:** For non-bridge cells, the function allows the agent to move to the next position if it is walkable.

**Conclusion:** <br>
This approach effectively modeled the probabilistic outcomes of bridge crossings, enabling the agent to balance the risk of crossing bridges against potential rewards. However, due to the randomness of the agility checks, the agent still occasionally falls off bridges. This is an inherent limitation of the probabilistic model, as even with a high success probability, there is always a chance of failure.

**Challenges in Complex Environments:** <br>
In some environments, the agent still faces difficulties:

  - **Bridge Crossings:** In scenarios with multiple bridges or low agility skill points, the agent struggles to cross bridges successfully, often getting stuck or falling repeatedly.
  - **Pit Avoidance:** The agent sometimes fails to find safe paths that avoid pits, especially in environments where pits are densely placed or block critical paths.
  - **Combined Challenges:** In environments with both pits and bridges, the agent's decision-making becomes more complex, and it occasionally fails to balance the risks effectively.


## Output Format
The code currently works for env-1, env-2 

### env-1
<img width="430" alt="image" src="https://github.com/user-attachments/assets/bc451b11-8dee-4509-893a-a658a2dac782" />

### env-2
<img width="432" alt="image" src="https://github.com/user-attachments/assets/3a04ebd9-8af8-481a-874e-392762df1085" />

### env-3
<img width="433" alt="image" src="https://github.com/user-attachments/assets/b4806568-c0d8-4d76-af8a-b99e4675bed6" />



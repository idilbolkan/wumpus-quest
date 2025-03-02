import random
import logging
from itertools import chain, combinations

# Constants
GAMMA = 0.99  # Increased discount factor to prioritize future rewards
EPSILON = 1e-6  # Convergence threshold
ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST", "EXIT"]

# Helper functions
#-----------------------------------------------------------------------------------------
"""Generate all possible subsets of a given iterable."""
def powerset(gold_locations):
    s = list(gold_locations)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
#-----------------------------------------------------------------------------------------
"""Parse the map into a 2D list and extract key locations."""
def parse_map(raw_map):
    grid = [list(row) for row in raw_map.split('\n') if row.strip()]
    start_pos = None
    gold_locations = []
    for row_idx, line in enumerate(grid):
        for col_idx, cell in enumerate(line):
            if cell == 'G':
                gold_locations.append((col_idx, row_idx))  # Store as (column, row)
            elif cell == 'S':
                start_pos = (col_idx, row_idx)  # Store as (column, row)
    return grid, gold_locations, start_pos
#-----------------------------------------------------------------------------------------
"""Return a list of all coordinates (column, row) in the grid that are walkable."""
def get_walkable_positions(grid):
    walkable_positions = []
    for row_idx in range(len(grid)):
        for col_idx in range(len(grid[0])):
            if grid[row_idx][col_idx] != 'X':  # Walkable if not a wall
                walkable_positions.append((col_idx, row_idx))  # Store as (column, row)
    return walkable_positions
#-----------------------------------------------------------------------------------------
"""Check if a position is within bounds and not a wall."""
def is_position_walkable(position, grid):
    col, row = position
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] != 'X' and grid[row][col] != 'P':
        return True
    return False
#-----------------------------------------------------------------------------------------
"""Compute the reward for a given transition."""
def get_reward(position, action, next_position, gold_collected, gold_locations, start_pos):
    reward = -0.1  # Base step penalty

    # Penalize for invalid moves (blocked by wall/pit)
    if next_position == position:
        reward -= 0.5

    if next_position in gold_locations and next_position not in gold_collected:
        reward += 10

    if action == "EXIT" and next_position == start_pos:
        total_gold = len(gold_collected)
        exit_reward = total_gold * 10
        if total_gold == len(gold_locations):
            exit_reward += 100
        reward += exit_reward

    return reward
#-----------------------------------------------------------------------------------------
"""Determine valid next positions based on the action, avoiding pits and walls."""
def get_possible_next_positions(position, action, grid):
    if action == "EXIT":
        if grid[position[1]][position[0]] == 'S':
            return {position}
        else:
            return set()

    directions = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0)
    }

    dc, dr = directions[action]
    new_col = position[0] + dc
    new_row = position[1] + dr
    new_pos = (new_col, new_row)

    # Only return walkable positions (pits are treated as walls)
    if is_position_walkable(new_pos, grid):
        return {new_pos}
    else:
        return {position}
#-----------------------------------------------------------------------------------------
"""Compute the transition probability for a given action."""
def get_transition_prob(position, action, next_position, grid):
    if action == "EXIT":
        if grid[position[1]][position[0]] == 'S' and next_position == position:
            return 1.0
        else:
            return 0.0

    directions = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0)
    }
    dc, dr = directions[action]
    new_col = position[0] + dc
    new_row = position[1] + dr
    new_pos = (new_col, new_row)
    actual_next_pos = new_pos if is_position_walkable(new_pos, grid) else position
    return 1.0 if next_position == actual_next_pos else 0.0
#-----------------------------------------------------------------------------------------
"""Perform Policy Iteration to compute the optimal policy."""
def policy_iteration(grid, gold_locations, start_pos):
    walkable_positions = get_walkable_positions(grid)
    states = [(pos, frozenset(gold_collected)) for pos in walkable_positions for gold_collected in powerset(gold_locations)]
    
    # Initialize policy and value function
    policy = {state: random.choice(ACTIONS) for state in states}
    V = {state: 0 for state in states}

    while True:
        # Policy Evaluation with more iterations
        for _ in range(1000):  # Increased iterations for better convergence
            delta = 0
            for state in states:
                position, gold_collected = state
                v = V[state]
                action = policy[state]
                total = 0
                for next_position in get_possible_next_positions(position, action, grid):
                    next_gold_collected = set(gold_collected)
                    if next_position in gold_locations and next_position not in gold_collected:
                        next_gold_collected.add(next_position)
                    reward = get_reward(position, action, next_position, gold_collected, gold_locations, start_pos)
                    prob = get_transition_prob(position, action, next_position, grid)
                    next_state = (next_position, frozenset(next_gold_collected))
                    total += prob * (reward + GAMMA * V[next_state])
                V[state] = total
                delta = max(delta, abs(v - V[state]))
            if delta < EPSILON:
                break

        # Policy Improvement
        policy_stable = True
        for state in states:
            position, gold_collected = state
            old_action = policy[state]
            best_action = None
            best_value = -float('inf')
            for action in ACTIONS:
                total = 0
                for next_position in get_possible_next_positions(position, action, grid):
                    # Skip actions that lead directly into pits
                    if grid[next_position[1]][next_position[0]] == 'P':
                        continue

                    next_gold_collected = set(gold_collected)
                    if next_position in gold_locations and next_position not in gold_collected:
                        next_gold_collected.add(next_position)

                    reward = get_reward(position, action, next_position, gold_collected, gold_locations, start_pos)
                    total += get_transition_prob(position, action, next_position, grid) * (reward + GAMMA * V[(next_position, frozenset(next_gold_collected))])

                if total > best_value:
                    best_value = total
                    best_action = action

                policy[state] = best_action
            if old_action != best_action:
                policy_stable = False

        if policy_stable:
            break

    return policy
#-----------------------------------------------------------------------------------------
"""Print the grid with the agent's position marked as 'A'."""
def print_grid(grid, agent_position):
    grid_copy = [row[:] for row in grid]
    col, row = agent_position
    grid_copy[row][col] = 'A'  # Mark the agent's position with 'A'
    for row in grid_copy:
        print(''.join(row))
    print()
#-----------------------------------------------------------------------------------------
"""Determines if the next position is safe. Treats pits as walls."""
def get_safe_next_position(current_position, action, grid, skill_points):
    directions = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "EAST": (1, 0),
        "WEST": (-1, 0)
    }

    if action not in directions:
        return current_position

    dc, dr = directions[action]
    new_col = current_position[0] + dc
    new_row = current_position[1] + dr

    if not (0 <= new_row < len(grid) and 0 <= new_col < len(grid[0])):
        return current_position

    next_cell = grid[new_row][new_col]

    # Handle bridges with agility checks
    if next_cell == 'B':
        agility_skill = skill_points.get("agility", 0)
        if agility_skill <= 0:
            print("Agility skill too low. Cannot attempt crossing.")
            return None

        while True:
            dice_rolls = [random.randint(1, 6) for _ in range(agility_skill)]
            dice_rolls.sort(reverse=True)
            top_dice = dice_rolls[:3]
            score = sum(top_dice)

            print(f"Rolling Dice: {dice_rolls}, Top 3 = {top_dice}, Score = {score}")

            if score >= 12:
                print("✅ Bridge crossing successful!")
                return (new_col, new_row)
            else:
                print("❌ Failed roll. Retrying...")

    # Pits and walls are not walkable
    if next_cell in ('X', 'P'):
        return current_position

    return (new_col, new_row)
#-----------------------------------------------------------------------------------------
"""Main agent function."""
def agent_function(request_data, request_info):
    print('_________________________________________________________')

    # Parse game state
    game_map = request_data.get('map', '')
    grid, gold_locations, start_pos = parse_map(game_map)
    free_skill_points = request_data.get("free-skill-points", 0)
    history = request_data.get("history", [])
    skill_points = request_data.get("skill-points", {})

    # Allocate skill points if needed (first action)
    if free_skill_points > 0:
        skill_allocation = {"agility": free_skill_points, "fighting": 0}
        return skill_allocation

    # Extract current position and gold collected from history
    current_position = start_pos
    gold_collected = set()

    if history:
        for event in history:
            outcome = event.get('outcome', {})
            if 'position' in outcome:
                current_position = tuple(outcome['position'])
            if 'collected-gold-at' in outcome:
                gold_pos = tuple(outcome['collected-gold-at'])
                gold_collected.add(gold_pos)

    # Print the amount of collected gold
    print(f"COLLECTED GOLD: {len(gold_collected)}")

    # Debugging: Print current position and grid
    print(f"Current Position: {current_position}")
    print(f"Grid Layout:")
    print_grid(grid, current_position)

    # Check if the agent is on the stairs and has collected gold
    if grid[current_position[1]][current_position[0]] == 'S' and gold_collected:
        return "EXIT"  # Return plain string for EXIT action
    
    if grid[current_position[1]][current_position[0]] == 'P':
        print("Agent fell into a pit and died.")

    # Compute the optimal policy using Policy Iteration
    policy = policy_iteration(grid, gold_locations, start_pos)
    
    # Override EXIT action unless all gold is collected
    state = (current_position, frozenset(gold_collected))
    action = policy.get(state, "NORTH")

    # Check if the next move is safe (especially for bridges)
    next_position = get_safe_next_position(current_position, action, grid, skill_points)
    
    # If next_position is None, it means we can't safely cross a bridge
    if next_position is None:
        print("Cannot safely cross bridge - looking for alternative route")
        # Find an alternative action that doesn't lead to an unsafe bridge
        for alt_action in ACTIONS:
            alt_next_position = get_safe_next_position(current_position, alt_action, grid, skill_points)
            if alt_next_position is not None:
                action = alt_action
                break
    
    return action
#-----------------------------------------------------------------------------------------
"""Run the agent."""
if __name__ == '__main__':
    import sys
    import logging
    from client import run

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run the agent
    run(
        agent_config_file=sys.argv[1],
        agent=agent_function,
        parallel_runs=True,
        run_limit=100000000  # Stop after 1000 runs
    )
import numpy as np
import time

# Grid size
ROWS = 3
COLS = 4

# Initialize grid world values
grid_world = np.zeros((ROWS, COLS))
# Define positions of the black square, green, and red endings
wall_position = (1, 1)
green_ending = (0, 3)
red_ending = (1, 3)

# Define rewards
reward_normal = -0.04
reward_good_end = 1
reward_bad_end = -1

# Define transition probabilities for stochastic policy
action_probs = {'UP': {'UP': 0.8, 'LEFT': 0.1, 'RIGHT': 0.1},
                'DOWN': {'DOWN': 0.8, 'LEFT': 0.1, 'RIGHT': 0.1},
                'LEFT': {'LEFT': 0.8, 'UP': 0.1, 'DOWN': 0.1},
                'RIGHT': {'RIGHT': 0.8, 'UP': 0.1, 'DOWN': 0.1}}


def get_neighbors(i, j):
    """
    Get neighboring states for a given state
    """
    neighbors = {
        'UP': (max(i - 1, 0), j),
        'DOWN': (min(i + 1, ROWS - 1), j),
        'LEFT': (i, max(j - 1, 0)),
        'RIGHT': (i, min(j + 1, COLS - 1))
    }
    return neighbors


def is_valid_move(state, action):
    """
    Check if a move from the current state in the given direction is valid
    """
    next_state = get_neighbors(*state)[action]
    if next_state == wall_position:
        return False
    return True


def evaluate_policy(policy):
    """
    Evaluate the given policy to get the value function
    """
    delta = float('inf')
    theta = 0.001
    discount_factor = 0.5
    while delta > theta:
        delta = 0
        for i in range(ROWS):
            for j in range(COLS):
                if (i, j) == wall_position:
                    continue
                v = grid_world[i, j]
                new_v = 0
                # Calculate the value for each action in the policy
                for action, prob in action_probs[policy[i, j]].items():
                    next_state = get_neighbors(i, j)[action]
                    if next_state == wall_position:
                        next_state = (i, j)

                    if((i,j)==green_ending or (i,j)==red_ending): next_state = (i,j)

                    if (i, j) == green_ending:
                        reward = reward_good_end
                    elif (i, j) == red_ending:
                        reward = reward_bad_end
                    else:
                        reward = reward_normal
                        
                    new_v += prob * (reward + discount_factor * grid_world[next_state[0], next_state[1]])
                grid_world[i, j] = new_v
                delta = max(delta, abs(v - new_v))


def improve_policy(policy):
    """
    Improve the policy based on the current value function
    """
    policy_stable = True
    for i in range(ROWS):
        for j in range(COLS):
            if (i, j) == wall_position or (i, j) == green_ending or (i, j) == red_ending:
                continue
            old_action = policy[i, j]
            max_v = -float('inf')
            # Find the action that maximizes the value function
            for action in action_probs.keys():
                new_v = 0
                for next_action, prob in action_probs[action].items():
                    next_state = get_neighbors(i, j)[next_action]
                    if next_state == wall_position:
                        next_state = (i, j)

                    if((i,j)==green_ending or (i,j)==red_ending): next_state = (i,j)

                    if (i, j) == green_ending:
                        reward = reward_good_end
                    elif (i, j) == red_ending:
                        reward = reward_bad_end
                    else:
                        reward = reward_normal
                    new_v += prob * (reward + 0.5 * grid_world[next_state[0], next_state[1]])
                if new_v > max_v:
                    max_v = new_v
                    policy[i, j] = action
            if old_action != policy[i, j]:
                policy_stable = False
    return policy_stable


def policy_iteration():
    """
    Perform policy iteration to find the optimal policy
    """
    
    policy = np.random.choice(list(action_probs.keys()), size=(ROWS, COLS))
    policy[1][1] = "-"
    policy_stable = False
    start_time = time.time()  # Start timing
    while not policy_stable:
        evaluate_policy(policy)
        policy_stable = improve_policy(policy)

    
    policy[1][3] = "-"
    policy[0][3] = "-"    
    
    end_time = time.time()  # End timing
    time_taken = end_time - start_time
    print("Time taken for convergence:", round(time_taken, 6), "seconds")
    return policy


print("")
print("")
print("Iterative policy iteration: ")
print("")
# Run policy iteration
optimal_policy = policy_iteration()
print("")
print("state values: ")
print(grid_world)
print("")
# Print results
print("Optimal Policy:")
print(optimal_policy)


print("")
print("")

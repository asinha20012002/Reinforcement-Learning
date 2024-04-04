import numpy as np

# Define the maze matrix with pillars represented by -2
maze = np.array([
    [0, 0, 0, 0],
    [0, -2, 0, -2],
    [0, 0, 0, -2],
    [-2, 0, 0, 10]
])

# Define parameters
gamma = 0.8  # Discount factor
alpha = 0.9  # Learning rate
epsilon = 0.1  # Exploration rate
num_actions = 4  # Number of possible actions
num_episodes = 10

# Initialize Q-values as a 3D array
Q = np.zeros((maze.shape[0], maze.shape[1], num_actions), dtype=np.float32)

# Helper function to get valid actions for a given state
def get_valid_actions(state):
    actions = []
    if state[0] > 0 and maze[state[0] - 1, state[1]] != -2:  # Up
        actions.append(0)
    if state[0] < maze.shape[0] - 1 and maze[state[0] + 1, state[1]] != -2:  # Down
        actions.append(1)
    if state[1] > 0 and maze[state[0], state[1] - 1] != -2:  # Left
        actions.append(2)
    if state[1] < maze.shape[1] - 1 and maze[state[0], state[1] + 1] != -2:  # Right
        actions.append(3)
    return actions

# Helper function to select an action based on epsilon-greedy policy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(get_valid_actions(state))
    else:
        return np.argmax(Q[state[0], state[1], :])

# Q-learning algorithm
for _ in range(num_episodes):
    state = (0, 0)
    while state != (3, 3):
        action = choose_action(state)
        next_state = None
        if action == 0:
            next_state = (state[0] - 1, state[1])
        elif action == 1:
            next_state = (state[0] + 1, state[1])
        elif action == 2:
            next_state = (state[0], state[1] - 1)
        elif action == 3:
            next_state = (state[0], state[1] + 1)

        if next_state[0] < 0 or next_state[0] >= maze.shape[0] or next_state[1] < 0 or next_state[1] >= maze.shape[1]:
            # Out of bounds, ignore this move
            continue

        if maze[next_state[0], next_state[1]] == -2:
            # Hit a pillar, ignore this move
            continue

        reward = maze[next_state[0], next_state[1]]
        max_next_Q = np.max(Q[next_state[0], next_state[1], :])
        Q[state[0], state[1], action] += alpha * (reward + gamma * max_next_Q - Q[state[0], state[1], action])
        state = next_state

# Extracting the optimal path
optimal_path = [(0, 0)]
state = (0, 0)
while state != (3, 3):
    action = np.argmax(Q[state[0], state[1], :])
    if action == 0:
        next_state = (state[0] - 1, state[1])
    elif action == 1:
        next_state = (state[0] + 1, state[1])
    elif action == 2:
        next_state = (state[0], state[1] - 1)
    elif action == 3:
        next_state = (state[0], state[1] + 1)
    optimal_path.append(next_state)
    state = next_state

# Print the optimal path
print("Optimal Path:")
for row, col in optimal_path:
    if(row == 3 and col == 3) :
        print(f"({row}, {col})")
    else: print(f"({row}, {col}) -> ", end="")

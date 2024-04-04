import numpy as np

# Define the maze as a 2D numpy array
maze = np.array([
    [-1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1],
    [-1, -1, 0, -1, 0, 0, -1, -1, 0, 0, -1],
    [-1, 0, -1, 0, -1, 0, 0, -1, -1, -1, -1],
    [0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1],
    [-1, -1, 0, -1, -1, 0, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, 0, -1, -1, 0, -1, -1],
    [-1, 0, -1, -1, -1, -1, -1, 0, -1, 0, 0],
    [-1, 0, -1, -1, -1, -1, -1, -1, 0, -1, 0],
    [-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1]
 ])

# Define the Q-table
Q = np.zeros_like(maze)

# Define parameters
alpha = 0.9 # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Epsilon for epsilon-greedy strategy
num_episodes = 1000

# Define epsilon-greedy function
def epsilon_greedy(state):
    if np.random.rand() < epsilon:
        # Explore
        return np.random.choice(np.arange(11))
    else:
        # Exploit
        return np.argmax(Q[state])

# Define the update rule for Q-learning
def update_Q(state, action, reward, next_state):
    Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

# Q-learning algorithm
for _ in range(num_episodes):
    current_state = 3 # Start from state 1
    done = False
    while not done:
        action = epsilon_greedy(current_state)
        if(maze[current_state, action] == -1):
            Q[current_state, action] = -1
            continue
        next_state = action
        reward = maze[current_state, next_state]
        if next_state == 10:
            done = True
            reward += 10
        update_Q(current_state, action, reward, next_state)
        current_state = next_state

# Find the optimal path
current_state = 3
optimal_path = [current_state]
while current_state != 10:
    action = np.argmax(Q[current_state])
    next_state = action
    current_state = next_state
    optimal_path.append(current_state)
#print Q table
# print(Q)
print("Optimal path:", optimal_path)

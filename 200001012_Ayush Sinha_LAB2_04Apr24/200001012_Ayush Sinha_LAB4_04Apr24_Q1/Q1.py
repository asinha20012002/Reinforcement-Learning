import numpy as np
import copy

def create_gridworld():
    # Initialize the grid world with default rewards
    grid = np.full((3, 4), -0.04)

    # Define rewards for terminal states: green and red square
    grid[0, 3] = 1  
    grid[1,3] = -1 
    
    # Print the initialized grid for visualization
    print(grid)
    return grid

def calculate_transition_probabilities(grid, terminal_states, pillars):
    # Initialize transition probabilities dictionary
    P = {}
    num_states = 12  # Number of states in the grid world
    
    # Loop through each state
    for s in range(num_states):
        # Initialize transition probabilities for each action
        P[s] = {a: [] for a in range(4)}  # 4 actions: Up, Right, Down, Left
        
        # Check if the state is a terminal state
        if (s // 4, s % 4) in terminal_states:
            # If terminal state, transition probability is 1.0 to itself with the respective reward
            for a in range(4):
                P[s][a] = [(1.0, s, grid[s // 4, s % 4], True)]
        else:
            x, y = s // 4, s % 4
            # Loop through each action (Up, Right, Down, Left)
            for a in range(4):
                dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][a] 
                outcomes = []
                # Determine possible outcomes for each action
                for da, db in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    new_x, new_y = max(0, min(2, x + da)), max(0, min(3, y + db))
                    # Check if the new position is a pillar
                    if [new_x,new_y] in pillars:
                        new_x, new_y = x, y 
                    new_s = new_x * 4 + new_y
                    # Calculate transition probability based on the action
                    if (da, db) == (dx, dy):
                        outcomes.append((0.8, new_s, grid[new_x, new_y], False))  
                    elif (da, db) == (-dy, dx):  # 90 degrees left
                        outcomes.append((0.1, new_s, grid[new_x, new_y], False))  
                    elif (da, db) == (dy, -dx):  # 90 degrees right
                        outcomes.append((0.1, new_s, grid[new_x, new_y], False))  
                    else:
                        outcomes.append((0.0, new_s, grid[new_x, new_y], False)) 
                # Assign calculated outcomes to the transition probabilities
                P[s][a] = outcomes
    # Print the transition probabilities for visualization
    # print(P)
    return P

def value_iteration(grid, P, gamma=0.5, theta=1e-3):
    num_states = 12  # Number of states in the grid world
    V = np.zeros(num_states)  # Initialize value function for all states
    
    n=0  # Counter for iterations
    while True:
        delta = 0  # Change in value function
        V_temp = copy.deepcopy(V)  # Create a copy of the current value function
        # Loop through each state
        for s in range(num_states):
            if s==5:  # Skip the pillar state
                continue
            v = V_temp[s]  # Current value for the state
            q_values = np.zeros(4)  # Initialize Q-values for each action
            new_v = 0
            # Calculate Q-values for each action using Bellman equation
            for a in range(4):  # 4 actions: Up, Right, Down, Left
                transitions = P[s][a]  # Possible transitions for the action
                for prob, next_state, reward, done in transitions:
                    q_values[a] += prob * (V[next_state])  # Update Q-value based on transition probabilities and rewards
            # Calculate the best action value using the maximum Q-value
            best_action_value = grid[s // 4, s % 4] + gamma*np.max(q_values)
            # Update the value function with the best action value
            V[s] = best_action_value
            # Update the change in value function
            delta = max(delta, abs(v - V[s]))

        n+=1  # Increment the iteration counter
        
        # Check for convergence
        if delta < theta:
            break  # Break the loop if convergence criterion is met
        
    print("Iterations : ",n)  # Print the number of iterations
    return V  # Return the optimal value function

if __name__ == "__main__":
    # Create the grid world
    grid = create_gridworld()
    terminal_states = [(0,3),(1,3)]  # Define terminal states: green and red square
    pillars = [[1,1]]  # Define pillars in the grid world
    # Calculate transition probabilities
    P = calculate_transition_probabilities(grid, terminal_states, pillars)
    # Perform value iteration to find the optimal value function
    V = value_iteration(grid, P)
    # Print the optimal value function, reshaped to match the grid world
    print("Value function:")
    print(V.reshape((3, 4)))

import numpy as np 
import time
import gym
import cv2
import pickle

from sklearn.preprocessing import KBinsDiscretizer
import math, random
from typing import Tuple

env = gym.make('CartPole-v1', render_mode='rgb_array')

policy = lambda *obs: 1

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
out = cv2.VideoWriter('cartpole_video.mp4', fourcc, 20.0, (600, 400))  # Adjust dimensions as per your environment

obs = env.reset()
while True:
    actions = policy(*obs)
    obs, reward, done, info = env.step(actions)[:4] 
    frame = env.render()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR color format
    out.write(frame)
    cv2.imshow('CartPole', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or done:
        break

out.release()
cv2.destroyAllWindows()
env.close()

n_bins = ( 6 , 12 )
lower_bounds = [ env.observation_space.low[2], -math.radians(60) ]
upper_bounds = [ env.observation_space.high[2], math.radians(60) ]

def discretizer( _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample= None)
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))

Q_table = np.zeros(n_bins + (env.action_space.n,))
Q_table.shape

def policy( state : tuple ):
    """Choosing action based on epsilon-greedy policy"""
    return np.argmax(Q_table[state])

def new_Q_value( reward : float ,  new_state : tuple , discount_factor=1 ) -> float:
    """Temperal diffrence for updating Q-value of state-action pair"""
    future_optimal_value = np.max(Q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value

def learning_rate(n : int , min_rate=0.01 ) -> float  :
    """Decaying learning rate"""
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

def exploration_rate(n : int, min_rate= 0.1 ) -> float :
    """Decaying exploration rate"""
    return max(min_rate, min(1, 1.0 - math.log10((n  + 1) / 25)))

# Training

n_episodes = 10000

for e in range(n_episodes):
    
    # Discretize state into buckets
    current_state, done = discretizer(*env.reset(), 0, 0), False
    
    while not done:
        
        # Policy action 
        action = policy(current_state) # exploit
        
        # Insert random action
        if np.random.random() < exploration_rate(e): 
            action = env.action_space.sample() # explore 
         
        # Increment environment
        obs, reward, done, _ = env.step(action)[:4]
        new_state = discretizer(*obs)
        
        # Update Q-Table
        lr = learning_rate(e)
        learnt_value = new_Q_value(reward, new_state)
        old_value = Q_table[current_state][action]
        Q_table[current_state][action] = (1 - lr) * old_value + lr * learnt_value
        
        current_state = new_state
        
        # Render the cartpole environment
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR format for compatibility with cv2
        cv2.imshow('CartPole', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the window if 'q' is pressed
            break
        time.sleep(0.02)  # Adjust the sleep time as needed for smoother rendering
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the window if 'q' is pressed
        break

cv2.destroyAllWindows()
env.close()
 
# QTable  
print("Q-Table after training:")
print(Q_table)

# Save Q-table to file after training
with open('Q_table.pkl', 'wb') as f:
    pickle.dump(Q_table, f)
print("Q-table saved successfully.")



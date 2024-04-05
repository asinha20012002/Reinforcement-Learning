import numpy as np
import cv2
import gym
import pickle
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import math, random
import time

# Load Q-table from file
try:
    with open('Q_table.pkl', 'rb') as f:
        Q_table = pickle.load(f)
    print("Q-table loaded successfully.")
except FileNotFoundError:
    print("No existing Q-table found. Exiting...")
    exit()

env = gym.make('CartPole-v1', render_mode='rgb_array')

n_testing_episodes = 10

n_bins = ( 6 , 12 )
lower_bounds = [ env.observation_space.low[2], -math.radians(60) ]
upper_bounds = [ env.observation_space.high[2], math.radians(60) ]

def discretizer( _ , __ , angle, pole_velocity ) -> Tuple[int,...]:
    """Convert continues state intro a discrete state"""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample= None)
    est.fit([lower_bounds, upper_bounds ])
    return tuple(map(int,est.transform([[angle, pole_velocity]])[0]))

def policy( state : tuple ):
    """Choosing action based on epsilon-greedy policy"""
    return np.argmax(Q_table[state])

for e in range(n_testing_episodes):
    current_state, done = discretizer(*env.reset(), 0, 0), False
    start_time = time.time()
    while not done:
        action = policy(current_state)
        obs, reward, done, _ = env.step(action)[:4]
        new_state = discretizer(*obs)
        current_state = new_state

        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('CartPole Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    end_time = time.time()
    episode_duration = end_time - start_time
    print(f"Episode {e + 1} duration: {episode_duration:.2f} seconds")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
env.close()

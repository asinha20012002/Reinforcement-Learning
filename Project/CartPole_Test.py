import numpy as np
import cv2
import gym
import pickle
from sklearn.preprocessing import KBinsDiscretizer
from typing import Tuple
import math
import time

# Load Q-table from file
try:
    with open('Q_table.pkl', 'rb') as f:
        Q_table = pickle.load(f)
    print("Q-table loaded successfully.")
except FileNotFoundError:
    print("No existing Q-table found. Exiting...")
    exit()

print(Q_table)

env = gym.make('CartPole-v1', render_mode='rgb_array')

n_testing_episodes = 20
n_bins = (6, 12)
lower_bounds = [env.observation_space.low[2], -math.radians(60)]
upper_bounds = [env.observation_space.high[2], math.radians(60)]

total_rewards = []
episode_durations = []
best_duration = 0
best_frames = []

def discretizer(_, __, angle, pole_velocity) -> Tuple[int, ...]:
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
    est.fit([lower_bounds, upper_bounds])
    return tuple(map(int, est.transform([[angle, pole_velocity]])[0]))

def policy(state: tuple):
    return np.argmax(Q_table[state])

for e in range(n_testing_episodes):
    current_state, done = discretizer(*env.reset(), 0, 0), False
    episode_reward = 0
    start_time = time.time()
    frames = []

    while not done:
        action = policy(current_state)
        obs, reward, done, _ = env.step(action)[:4]
        new_state = discretizer(*obs)
        current_state = new_state
        episode_reward += reward

        frame = env.render()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imshow('CartPole Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = time.time()
    episode_duration = end_time - start_time

    print(f"Episode {e + 1} duration: {episode_duration:.2f} seconds, Total Reward: {episode_reward}")

    total_rewards.append(episode_reward)
    episode_durations.append(episode_duration)

    if episode_duration > best_duration:
        best_duration = episode_duration
        best_frames = frames

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Average Total Reward: {np.mean(total_rewards)}")
print(f"Average Episode Duration: {np.mean(episode_durations)}")

# Save video of the best episode
if best_frames:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('best_cartpole_episode.mp4', fourcc, 20.0, (600, 400))
    for frame in best_frames:
        out.write(frame)
    out.release()

cv2.destroyAllWindows()
env.close()

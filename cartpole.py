# python cartpole.py > replay
import gym
import numpy as np
import matplotlib.pyplot as plt

# Save model
def save_list(q_table):
    with open('saved_q_list.npy', 'wb') as f:
        np.save(f, q_table)
        print('Q_TABLE WAS SAVED')
    f.close()

# Load model
def load_list():
    with open('saved_q_list.npy', 'rb') as f:
        q_table = np.load(f)
        print('USING SAVED Q_TABLE')
    f.close()
    return q_table


env = gym.make("MountainCar-v0")

USE_SAVED_LIST = True
SAVE_LIST = False

LEARNING_RATE = 0.025
GAMMA = 0.95
EPISODES = 15
SHOW = True
SHOW_THIS_OFTEN = 1

epsilon = 0
START_EPSILON_DECAYING = 3000
END_EPSILON_DECAYING = 6000
EPSILON_DECAY = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OS_SIZE = [50] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE

actions = { 0 : "Left", 1 : "Nothing", 2 : "Right" }

# Acc. Rewards per Episode List for Plotting
RewardPerEpisodeList = []

if USE_SAVED_LIST:
    q_table = load_list()
else:
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
    # print(q_table.shape)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):

    if SHOW and episode%SHOW_THIS_OFTEN == 0:
        print(f"episode={episode}, eps={epsilon}")
        render = True
    else:
        render = False

    rewardSum = 0
    done = False
    discrete_state = get_discrete_state(env.reset())

    if episode > 1 and episode > START_EPSILON_DECAYING and episode < END_EPSILON_DECAYING:
        epsilon = epsilon - EPSILON_DECAY

    # Run Episode
    while not done:

        # Action Selection
        if np.random.rand() < epsilon and episode > START_EPSILON_DECAYING and episode < END_EPSILON_DECAYING:
            # Explore (Pick random action)
            action = np.random.choice(3)

        else:
            # Pick greedy action
            action = np.argmax(q_table[discrete_state])

        # Observation
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        rewardSum += reward

        # Render
        if render:
            env.render()

        # Q Update
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + GAMMA * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0

        # Update State
        discrete_state = new_discrete_state

    RewardPerEpisodeList.append(rewardSum)

env.close()
if SAVE_LIST:
    save_list(q_table)

# Plot
plt.figure(figsize=(20,10))
plt.plot(RewardPerEpisodeList, label='Accumulated reward per Episode', color='green', linewidth=0.4)
plt.title('Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

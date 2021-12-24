# python cartpole.py > replay
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
GAMMA = 0.95
EPISODES = 20000
SHOW_THIS_OFTEN = 1000

epsilon = 0.7
START_EPSILON_DECAYING = 6000
END_EPSILON_DECAYING = 11000
EPSILON_DECAY = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high-env.observation_space.low)/DISCRETE_OS_SIZE

actions = { 0 : "Left", 1 : "Nothing", 2 : "Right" }

# Acc. Rewards per Episode List for Plotting
RewardPerEpisodeList = []

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# print(q_table.shape)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):

    if episode%SHOW_THIS_OFTEN == 0:
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

# Plot
plt.figure(figsize=(20,10))
plt.plot(RewardPerEpisodeList, label='Accumulated reward per Episode', color='green', linewidth=0.4)
plt.title('Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

# Save model
# data = q_table
# with open('model.txt', 'w') as outfile:
#     outfile.write('# Array shape: {0}\n'.format(data.shape))
#     for data_slice in data:
#         np.savetxt(outfile, data_slice, fmt='%-7.2f')
#         outfile.write('# New slice\n')
# outfile.close()

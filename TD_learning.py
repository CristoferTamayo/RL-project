from MazeEnv import MazeEnv
from Maze import Maze
from plot_utils import plot_different_scales
from plot_utils import plot_heatmap
import numpy as np
import matplotlib.pyplot as plt
import random
import time

class TD_learning():
    def __init__(self, env, epsilon, gamma, alpha): # alpha is learning rate
        self.Q = {}
        self.env = env
        self.epsilon_min = 0.001
        self.epsilon_max = epsilon
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
    
    def initialize_Q(self):
        for row in range(self.env.rows):
            for col in range(self.env.cols):
                self.Q[(row, col)] = [0, 0, 0, 0] # State (row, col), list with values for each action

    def argmax(self, action_value_list):
        ties = []
        for index in range(len(action_value_list)):
            if action_value_list[index] == max(action_value_list):
                ties.append(index)
        return ties

    def epsilon_greedy(self, state):
        prob = np.random.random()
        action = 0
        if prob < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = random.choice(self.argmax(self.Q.get(state))) ## In the case of ties, pick randomly
        return action

    def sarsa(self, number_of_episodes):
        self.initialize_Q()
        steps_per_episode = []
        trajectories = []
        q_tables = []
        # trajectory_actions = []
        for episode in range(number_of_episodes):
            self.env.reset() # Reward = 0 and current state is starting state, i.e. (0,0)
            total_steps = 0
            current_trajectory = []
            current_state = self.env.maze.current_position
            current_action = self.epsilon_greedy(current_state)
            done = False
            while not done: # Loop until episode is done
                next_state, reward, done = self.env.step(current_action)
                total_steps += 1
                next_action = self.epsilon_greedy(next_state)
                self.Q[current_state][current_action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state][next_action] - self.Q[current_state][current_action])
                # if episode >= 499 and episode <= 514: # Collecting trajectories episodes 500-515
                current_trajectory.append(current_state)
                # trajectory_actions.append(current_action)
                current_state = next_state
                current_action = next_action
            steps_per_episode.append(total_steps)
            # if episode >= 499 and episode <= 514: # Collecting trajectories episodes 500-515
            q_tables.append(self.Q)
            trajectories.append(current_trajectory)
            #print("Episode " + str(episode) + " done!")

            # Including a trap
            if episode == 499:
                self.env.change_environment([(2,0), (2,1), (2,2), (2,3)])

        return q_tables, steps_per_episode, trajectories #, trajectory_actions

    def q_learning(self, number_of_episodes):
        self.initialize_Q()
        steps_per_episode = []
        trajectories = []
        q_tables = []
        # trajectory_actions = []
        for episode in range(number_of_episodes):
            self.env.reset()
            current_state = self.env.maze.current_position
            total_steps = 0
            current_trajectory = []
            done = False
            while not done:
                current_action = self.epsilon_greedy(current_state)
                next_state, reward, done= self.env.step(current_action)
                greedy_action = random.choice(self.argmax(self.Q[next_state])) ## For ties
                self.Q[current_state][current_action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state][greedy_action] - self.Q[current_state][current_action])
                # if episode >= 499 and episode <= 514: # Collecting trajectories episodes 500-515
                current_trajectory.append(current_state)
                # trajectory_actions.append(current_action)
                current_state = next_state
                total_steps += 1
            steps_per_episode.append(total_steps)
            # if episode >= 499 and episode <= 514: # Collecting trajectories episodes 500-515
            trajectories.append(current_trajectory)
            q_tables.append(self.Q)
            #print("Episode " + str(episode) + " done!")

            ## Including a trap
            if episode == 499:
                self.env.change_environment([(2,0), (2,1), (2,2), (2,3)])

        return q_tables, steps_per_episode, trajectories #, trajectory_actions

    def get_optimal_policy(self, Q):
        optimal_policy = []
        for action_value_list in Q.values():
            optimal_action = self.argmax(action_value_list)
            optimal_policy.append(optimal_action)
        return optimal_policy
    
    

# starting_traps = [(2,1), (2,2), (2,3), (2,4)]
# ending_traps = [(2,0), (2,1), (2,2), (2,3)]
# env = MazeEnv(5,5, starting_traps)
# TD = TD_learning(env, 0.8, 1, 0.1) #epsilon, gamma, alpha

# Sarsa training + Heatmap
# q_sarsa, steps_per_episode_sarsa, trajectories_sarsa = TD.sarsa(1000) # , actions_traj_sarsa
# print(trajectories_sarsa)
# print(len(trajectories_sarsa))
# print(len(trajectories_sarsa[0]))
# print(len(trajectories_sarsa[1]))
# print(trajectories_sarsa)
# print("Sarsa is done.")
# plot_heatmap([trajectory_sarsa], ending_traps, "Heatmap SARSA - Episode 501")

##
# print("Sarsa trajectory: " + str(trajectory_sarsa))
# print("Sarsa actions trajectory: " + str(actions_traj_sarsa))
# print("Sarsa's trajectory is " + str(len(trajectory_sarsa)))
# print("Sarsa's trajectory at change is: ")
# time.sleep(2)
# env.maze.print_trajectory(trajectory_sarsa)

# Q-learning training + Heatmap
# env.change_environment(starting_traps) # Change grid so Q-learning starts with same starting grid as Sarsa

# print("Q-learning is being applied now")
# q_Q_learning, steps_per_episode_Q_learning, trajectories_Qlearning = TD.q_learning(1000) # , actions_traj_Qlearning 
# print(trajectories_Qlearning)
# print(len(trajectories_Qlearning))
# print(len(trajectories_Qlearning[0]))
# print(len(trajectories_Qlearning[1]))

# print("Q-learning is done")
# plot_heatmap([trajectory_Qlearning], ending_traps, "Heatmap SARSA - Episode 501")

##
# print("Q-learning trajectory: " + str(trajectory_Qlearning))
# print("Q-learning actions trajectory: " + str(actions_traj_Qlearning))
# print("Q-learning's trajectory is " + str(len(trajectory_Qlearning)))
# print("Q-learning's trajectory at change is: ")
# time.sleep(2)
# env.maze.print_trajectory(trajectory_Qlearning)



# optimal_policy_sarsa = TD.get_optimal_policy(q_sarsa)
# env.print_optimal_policy(optimal_policy_sarsa, "1000 episodes Sarsa")

# env.change_environment([(0,2), (1,4), (2,3), (2,4), (3,1), (4,2)]) # Change so Q-laerning has same starting grid

# q_Q_learning, steps_per_episode_Q_learning = TD.q_learning(1000)
# optimal_policy_Q_learning = TD.get_optimal_policy(q_Q_learning)
# env.print_optimal_policy(optimal_policy_Q_learning, "1000 episodes Q-learning")

# start = 500
# end = 550
# plt.plot(steps_per_episode_sarsa[start:end], label="Sarsa", color="magenta")
# plt.plot(steps_per_episode_Q_learning[start:end], label="Q-learning", color="orange")
# plt.xlabel("Episode")
# plt.ylabel("Number of steps")
# plt.title('Learning Curve Lin-Lin')
# plt.legend()
# plt.grid(True)
# plt.show()

# x_axis = np.arange(start, end)
# plot_different_scales(x_axis, steps_per_episode_sarsa[start:end], steps_per_episode_Q_learning[start:end], epsilon=0.8, info="steps") #episodes_axis, avg_sarsa, avg_qlearning, epsilon, info


# Changing environment 2

# env = MazeEnv(5, 5, traps=[(0,2), (1,4), (2,3), (2,4), (3,1), (4,2)]) # 
# TD = TD_learning(env, 0.1, 1, 0.1) # Order: epsilon, gamma = 1 no discount, alpha


# q_sarsa, steps_per_episode_sarsa = TD.sarsa(1000)
# optimal_policy_sarsa = TD.get_optimal_policy(q_sarsa)
# env.print_optimal_policy(optimal_policy_sarsa, "1000 episodes Sarsa")

# env.change_environment([(0,2), (1,4), (2,3), (2,4), (3,1), (4,2)])

# q_Q_learning, steps_per_episode_Q_learning = TD.q_learning(1000)
# optimal_policy_Q_learning = TD.get_optimal_policy(q_Q_learning)
# env.print_optimal_policy(optimal_policy_Q_learning, "1000 episodes Q-learning")


# ##Steps per episode for SARSA
# plt.plot(steps_per_episode_sarsa, label="Sarsa")
# plt.axhline(y=8, color='black', linestyle='--', label="Number of steps: 8")
# plt.xlabel("Episode")
# plt.ylabel("Number of steps")
# plt.title("Steps per episode with SARSA (500 episodes)")
# plt.legend()
# plt.grid(True)
# plt.show()



# # Steps per episode for Q-learning
# plt.plot(steps_per_episode_Q_learning)
# plt.axhline(y=8, color='black', linestyle='--', label="Number of steps: 8")
# plt.xlabel("Episode")
# plt.ylabel("Number of steps")
# plt.title("Steps per episode with Q-learning (500 episodes)")
# plt.grid(True)
# plt.show()





## Testing with multiprocessing

# import multiprocessing as mp


# def run_sarsa_all(runs, number_episodes):
#     sarsa_runs = []
#     env = MazeEnv(5, 5, traps=[(2,1), (2,2), (2,3), (2,4)]) # 
#     TD = TD_learning(env, 0.1, 1, 0.1) # Order: epsilon, gamma = 1 no discount, alpha

#     for run in range(runs):
#         steps_per_episode_sarsa = TD.sarsa(number_episodes)
#         env.change_environment([(2,1), (2,2), (2,3), (2,4)]) # Changing the env back to starting grid
#         sarsa_runs.append(steps_per_episode_sarsa)
#         print(f"Sarsa run {run} done!")
#     np.save('sarsa_steps_'+str(runs)+'_runs.npy', np.vstack(sarsa_runs))
#     #print("Sarsa guardado en ", os.path.abspath('sarsa_steps_'+str(runs)+'_runs.npy'))

# def run_qlearning_all(runs, number_episodes):
#     qlearning_runs = []
#     env = MazeEnv(5, 5, traps=[(2,1), (2,2), (2,3), (2,4)]) # 
#     TD = TD_learning(env, 0.1, 1, 0.1) # Order: epsilon, gamma = 1 no discount, alpha

#     for run in range(runs):
#         steps_per_episode_qlearning = TD.sarsa(number_episodes)
#         env.change_environment([(2,1), (2,2), (2,3), (2,4)]) # Changing the env back to starting grid
#         qlearning_runs.append(steps_per_episode_qlearning)
#         print(f"Qlearning run {run} done!")
#     np.save('qlearning_steps_'+str(runs)+'_runs.npy', np.vstack(qlearning_runs))
#     #print("Qlearning guardado en ", os.path.abspath('qlearning_steps_'+str(runs)+'_runs.npy'))



# if __name__ == "__main__":

#     try:
#         mp.set_start_method('spawn')
#     except RuntimeError:
#         pass

#     runs = 10000
#     episodes = 1000

#     sarsa_process = mp.Process(target=run_sarsa_all, args=(runs, episodes))
#     qlearning_process = mp.Process(target=run_qlearning_all, args=(runs, episodes))

#     sarsa_process.start()
#     qlearning_process.start()

#     sarsa_process.join()
#     qlearning_process.join()

#     steps_sarsa = np.load('sarsa_steps_'+str(runs)+'_runs.npy')
#     steps_qlearning = np.load('qlearning_steps_'+str(runs)+'_runs.npy')

#     avg_steps_sarsa = steps_sarsa.mean(axis=0)
#     avg_steps_qlearning = steps_qlearning.mean(axis=0)

#     plot_different_scales(avg_steps_sarsa, avg_steps_qlearning)

    # print("Existe?", os.path.exists('sarsa_steps_'+str(runs)+'_runs.npy'))
    # print("Existe?", os.path.exists('qlearning_steps_'+str(runs)+'_runs.npy'))









## Simulations

# env = MazeEnv(5, 5, traps=[(2,1), (2,2), (2,3), (2,4)]) # 
# TD = TD_learning(env, 0.1, 1, 0.1) # Order: epsilon, gamma = 1 no discount, alpha

# runs = 1000
# episodes = 1000
# sarsa_runs = []
# qlearning_runs = []
# # action_value_func_sarsa = {}
# # action_value_func_q_learning = {}

# for run in range(runs):
#     steps_per_episode_sarsa = TD.sarsa(episodes)
#     #print(f"Sarsa run done!")
#     env.change_environment([(2,1), (2,2), (2,3), (2,4)])
#     # if run == runs-1:
#     #     action_value_func_sarsa = TD.Q
#     steps_per_episode_qlearning = TD.q_learning(episodes)
#     #print(f"Q-learning run done!")
#     env.change_environment([(2,1), (2,2), (2,3), (2,4)])
#     # if run == runs-1:
#     #     action_value_func_q_learning = TD.Q

#     sarsa_runs.append(steps_per_episode_sarsa)
#     qlearning_runs.append(steps_per_episode_qlearning)

#     print(f"Run number {run} is done!")

# env.change_environment([(2,0), (2,1), (2,2), (2,3)])
# # Optimal policy from the last run
# # optimal_policy_sarsa = TD.get_optimal_policy(action_value_func_sarsa)
# # env.print_optimal_policy(optimal_policy_sarsa, "Optimal policy Sarsa")

# # optimal_policy_sarsa = TD.get_optimal_policy(action_value_func_q_learning)
# # env.print_optimal_policy(optimal_policy_sarsa, "Optimal policy Q-learning")


# steps_runs_sarsa = np.vstack(sarsa_runs)
# steps_runs_qlearning = np.vstack(qlearning_runs)

# avg_steps_sarsa = steps_runs_sarsa.mean(axis=0)
# avg_steps_qlearning = steps_runs_qlearning.mean(axis=0)

# avg_steps_sarsa_list = avg_steps_sarsa.tolist()
# avg_steps_qlearning_list = avg_steps_qlearning.tolist()

# # print(f"Average steps sarsa: {avg_steps_sarsa_list}")
# #print(f"Length of list above: {len(avg_steps_sarsa_list)}")

# # print(f"Average steps qlearning: {avg_steps_qlearning_list}")
# #print(f"Length of list above: {len(avg_steps_qlearning_list)}")


# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4))

# ax[0].plot(avg_steps_sarsa, label="Sarsa", color="magenta")
# ax[0].plot(avg_steps_qlearning, label="Q-learning", color="orange")
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[0].set_xlabel('Episodes (log scale)')
# ax[0].set_ylabel('Average steps per episode (log scale)')
# ax[0].set_title('Learning Curve Log-Log')
# ax[0].legend()

# ax[1].plot(avg_steps_sarsa, label="Sarsa", color="magenta")
# ax[1].plot(avg_steps_qlearning, label="Q-learning", color="orange")
# ax[1].set_xscale('log')
# ax[1].set_xlabel('Episodes (log scale)')
# ax[1].set_ylabel('Average steps per episode (linear)')
# ax[1].set_title('Learning Curve Log-Lin')
# ax[1].legend()

# ax[2].plot(avg_steps_sarsa, label="Sarsa", color="magenta")
# ax[2].plot(avg_steps_qlearning, label="Q-learning", color="orange")
# ax[2].set_yscale('log')
# ax[2].set_xlabel('Episodes')
# ax[2].set_ylabel('Average steps per episode (log scale)')
# ax[2].set_title('Learning Curve Lin-Log')
# ax[2].legend()

# ax[3].plot(avg_steps_sarsa, label="Sarsa", color="magenta")
# ax[3].plot(avg_steps_qlearning, label="Q-learning", color="orange")
# ax[3].set_xlabel('Episodes')
# ax[3].set_ylabel('Average steps per episode ')
# ax[3].set_title('Learning Curve Lin-Lin')
# ax[3].legend()

# plt.tight_layout()
# plt.show()






# q_sarsa, steps_per_episode_sarsa = TD.sarsa(500) 
# q_Q_learning, steps_per_episode_Q_learning = TD.q_learning(500)

# Steps per episode for SARSA
# plt.plot(steps_per_episode_sarsa, label="Sarsa")
# plt.axhline(y=8, color='black', linestyle='--', label="Number of steps: 8")
# plt.xlabel("Episode")
# plt.ylabel("Number of steps")
# plt.title("Steps per episode with SARSA (500 episodes)")
# plt.legend()
# plt.grid(True)
# plt.show()

# optimal_policy_sarsa = TD.get_optimal_policy(q_sarsa)
# env.print_optimal_policy(optimal_policy_sarsa, "500_episodes_Sarsa")

# # Steps per episode for Q-learning
# plt.plot(steps_per_episode_Q_learning)
# plt.axhline(y=8, color='black', linestyle='--', label="Number of steps: 8")
# plt.xlabel("Episode")
# plt.ylabel("Number of steps")
# plt.title("Steps per episode with Q-learning (500 episodes)")
# plt.grid(True)
# plt.show()

# optimal_policy_Q_learning = TD.get_optimal_policy(q_Q_learning)
# env.print_optimal_policy(optimal_policy_Q_learning, "500 episodes Q-learning")



## TRYING DIFFERENT PLOTS AND SCALES
# env = MazeEnv(5, 5, traps=[(2,1), (2,2), (2,3), (2,4)])
# TD = TD_learning(env, 0.1, 1, 0.1) # Order: epsilon, gamma = 1 no discount, alpha
# # Epsilon will decrease exponentially after each episode. The max_epsilon is the one from input.

# q_sarsa, reward_per_episode_sarsa, steps_per_episode_sarsa = TD.sarsa(500) 
# q_Q_learning, reward_per_episode_Q_learning, steps_per_episode_Q_learning = TD.q_learning(500)
# print(f"Reward per episode Sarsa: {reward_per_episode_sarsa}")
# print(f"Steps per episode Sarsa: {steps_per_episode_sarsa}")
# print(f"Reward per episode Q-learning: {reward_per_episode_Q_learning}")
# print(f"Steps per episode Q-learning: {steps_per_episode_Q_learning}")

# steps_per_episode_sarsa = [23, 127, 30, 44, 71, 57, 46, 23, 31, 75, 100, 38, 24, 12, 77, 29, 12, 16, 26, 18, 21, 17, 64, 55, 32, 22, 12, 30, 8, 18, 32, 58, 12, 38, 13, 34, 14, 22, 14, 17, 27, 38, 8, 36, 12, 20, 10, 33, 18, 12, 18, 28, 10, 14, 26, 39, 8, 13, 20, 17, 20, 33, 9, 14, 24, 14, 10, 10, 26, 15, 12, 26, 14, 12, 10, 26, 37, 16, 9, 12, 19, 10, 10, 14, 12, 26, 12, 19, 36, 12, 14, 15, 18, 16, 12, 10, 11, 10, 12, 10, 14, 13, 25, 14, 14, 22, 10, 10, 10, 8, 9, 12, 19, 12, 10, 24, 17, 16, 10, 12, 20, 12, 12, 10, 14, 9, 9, 20, 14, 10, 8, 12, 10, 8, 11, 10, 10, 14, 8, 18, 10, 14, 14, 10, 20, 8, 15, 14, 8, 8, 16, 12, 8, 9, 8, 8, 19, 12, 12, 25, 11, 11, 10, 10, 26, 17, 14, 10, 14, 9, 9, 10, 8, 10, 16, 8, 8, 16, 14, 9, 20, 10, 14, 10, 12, 12, 11, 10, 8, 8, 9, 10, 8, 9, 8, 10, 9, 8, 9, 8, 14, 12, 12, 8, 8, 15, 10, 14, 10, 10, 22, 8, 24, 10, 8, 10, 8, 8, 8, 15, 8, 8, 10, 10, 10, 8, 8, 11, 10, 16, 10, 12, 22, 8, 10, 10, 8, 8, 12, 8, 8, 9, 10, 12, 15, 8, 9, 10, 12, 8, 15, 10, 10, 11, 8, 14, 13, 8, 12, 9, 8, 8, 8, 10, 9, 8, 8, 8, 10, 8, 10, 8, 10, 10, 8, 10, 8, 8, 8, 8, 11, 12, 13, 11, 8, 8, 8, 15, 14, 13, 14, 14, 8, 13, 10, 10, 10, 8, 10, 10, 8, 12, 9, 9, 8, 8, 9, 15, 10, 13, 8, 10, 10, 8, 13, 8, 9, 11, 8, 12, 10, 12, 10, 8, 8, 8, 8, 12, 10, 10, 8, 8, 11, 8, 9, 8, 9, 8, 8, 10, 10, 8, 8, 10, 8, 8, 10, 12, 8, 10, 11, 9, 8, 8, 10, 8, 8, 10, 10, 10, 10, 12, 8, 11, 8, 10, 8, 8, 10, 16, 8, 10, 9, 8, 12, 9, 8, 8, 8, 8, 8, 8, 8, 9, 8, 8, 10, 8, 8, 8, 8, 8, 11, 16, 13, 8, 8, 10, 10, 8, 11, 8, 8, 8, 8, 8, 10, 8, 8, 11, 8, 8, 8, 10, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10, 8, 8, 12, 8, 8, 9, 8, 8, 8, 14, 8, 10, 8, 9, 12, 10, 9, 8, 8, 8, 12, 14, 8, 8, 12, 11, 10, 10, 9, 8, 8, 9, 8, 8, 8, 8, 8, 8, 10, 8, 8, 9, 8, 8, 8, 8, 8, 8, 10, 9, 8, 8, 10, 8, 12, 10, 8, 8, 8, 10, 8, 10, 10, 11, 13, 12, 8, 8, 8, 8, 10, 8, 8, 10, 8]
# steps_per_episode_Q_learning = [32, 51, 86, 29, 126, 48, 26, 20, 17, 89, 67, 29, 38, 70, 28, 17, 22, 19, 49, 25, 23, 16, 32, 21, 49, 14, 62, 31, 17, 13, 28, 24, 14, 34, 26, 17, 39, 25, 29, 13, 35, 25, 9, 13, 31, 13, 21, 13, 19, 27, 16, 16, 23, 22, 35, 12, 18, 23, 10, 20, 16, 15, 18, 11, 21, 23, 23, 19, 9, 16, 20, 33, 19, 32, 18, 12, 10, 10, 12, 17, 8, 9, 9, 24, 15, 10, 10, 9, 25, 18, 22, 11, 11, 26, 19, 13, 16, 12, 13, 10, 8, 14, 10, 25, 16, 13, 12, 9, 15, 10, 17, 19, 8, 17, 12, 18, 8, 11, 12, 10, 11, 8, 31, 8, 12, 18, 10, 10, 14, 8, 10, 10, 10, 15, 9, 13, 9, 16, 8, 10, 12, 17, 12, 14, 19, 17, 10, 18, 8, 16, 9, 12, 9, 8, 13, 8, 20, 12, 11, 9, 10, 9, 12, 14, 11, 15, 12, 9, 8, 18, 14, 9, 17, 8, 8, 12, 8, 9, 11, 9, 8, 16, 15, 8, 16, 9, 10, 14, 8, 12, 12, 10, 8, 10, 10, 9, 10, 13, 15, 8, 10, 14, 8, 8, 8, 10, 12, 9, 13, 8, 11, 10, 14, 13, 11, 8, 8, 13, 8, 12, 8, 10, 9, 15, 8, 9, 13, 10, 8, 10, 12, 8, 10, 9, 10, 13, 15, 11, 8, 11, 8, 11, 10, 9, 20, 10, 9, 8, 10, 8, 8, 8, 16, 11, 9, 10, 11, 9, 9, 8, 8, 8, 10, 8, 11, 8, 10, 8, 10, 11, 11, 8, 8, 13, 8, 8, 9, 8, 8, 12, 10, 8, 12, 10, 9, 8, 11, 10, 8, 9, 15, 10, 13, 11, 8, 11, 8, 12, 11, 8, 11, 12, 8, 16, 8, 8, 16, 8, 8, 8, 12, 8, 9, 10, 10, 8, 8, 8, 12, 10, 8, 9, 10, 11, 8, 8, 8, 10, 8, 8, 8, 8, 9, 10, 13, 12, 9, 11, 16, 12, 9, 11, 8, 11, 8, 9, 12, 8, 8, 12, 9, 8, 8, 8, 8, 10, 8, 10, 10, 10, 8, 8, 8, 8, 8, 8, 11, 10, 8, 9, 8, 10, 10, 8, 10, 8, 8, 8, 8, 8, 8, 8, 8, 10, 10, 10, 9, 8, 8, 8, 10, 9, 8, 11, 8, 10, 9, 8, 8, 15, 10, 10, 10, 8, 11, 8, 8, 11, 8, 10, 8, 10, 10, 10, 10, 8, 10, 8, 9, 8, 8, 11, 8, 10, 8, 8, 8, 8, 8, 8, 8, 10, 10, 8, 11, 8, 8, 10, 10, 8, 8, 8, 12, 11, 10, 8, 8, 8, 8, 8, 10, 8, 8, 8, 10, 8, 10, 9, 10, 8, 10, 8, 11, 10, 8, 8, 8, 8, 10, 8, 8, 8, 9, 8, 8, 8, 10, 8, 8, 8, 8, 8, 9, 8, 8, 8, 10, 8, 8, 8, 8, 8, 9, 8, 8, 8, 12, 11, 9, 8]

# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,4))

# ax[0].plot(steps_per_episode_sarsa, label="Sarsa", color="magenta")
# ax[0].plot(steps_per_episode_Q_learning, label="Q-learning", color="orange")
# ax[0].set_xscale('log')
# ax[0].set_yscale('log')
# ax[0].set_xlabel('Episodes (log scale)')
# ax[0].set_ylabel('Steps per episode (log scale)')
# ax[0].set_title('Learning Curve Log-Log')
# ax[0].legend()

# ax[1].plot(steps_per_episode_sarsa, label="Sarsa", color="magenta")
# ax[1].plot(steps_per_episode_Q_learning, label="Q-learning", color="orange")
# ax[1].set_xscale('log')
# ax[1].set_xlabel('Episodes (log scale)')
# ax[1].set_ylabel('Steps per episode (linear)')
# ax[1].set_title('Learning Curve Log-Lin')
# ax[1].legend()

# ax[2].plot(steps_per_episode_sarsa, label="Sarsa", color="magenta")
# ax[2].plot(steps_per_episode_Q_learning, label="Q-learning", color="orange")
# ax[2].set_yscale('log')
# ax[2].set_xlabel('Episodes')
# ax[2].set_ylabel('Steps per episode (log scale)')
# ax[2].set_title('Learning Curve Lin-Log')
# ax[2].legend()

# ax[3].plot(steps_per_episode_sarsa, label="Sarsa", color="magenta")
# ax[3].plot(steps_per_episode_Q_learning, label="Q-learning", color="orange")
# ax[3].set_xlabel('Episodes')
# ax[3].set_ylabel('Steps per episode ')
# ax[3].set_title('Learning Curve Lin-Lin')
# ax[3].legend()

#plt.tight_layout()
# plt.show()






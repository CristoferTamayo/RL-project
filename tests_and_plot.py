from TD_learning import TD_learning
from MazeEnv import MazeEnv
from Maze import Maze
from plot_utils import plot_different_scales
from plot_utils import plot_multiple_heatmaps
from plot_utils import interactive_trajectory
from plot_utils import median_heatmap
from plot_utils import visits_boxplot
from plot_utils import checkpoint_plot
import multiprocessing as mp
import numpy as np
import time

def print_grid(rows=5, cols=5, traps=[]):
    env = MazeEnv(rows, cols, traps)
    env.render()
    env.maze.check_events()
    time.sleep(2) # Wait 2 seconds until the window is closed automatically.

def runs_sarsa(runs, number_episodes, epsilon, starting_traps=[], ending_traps=[]):
    sarsa_runs_steps = []
    sarsa_trajectories = []
    sarsa_q_tables = []
    # sarsa_trajectories_actions = []
    env = MazeEnv(5, 5, starting_traps)
    TD = TD_learning(env, epsilon, 1, 0.1) # Order: epsilon, gamma = 1 no discount, alpha

    for run in range(runs):
        q_tables, steps_per_episode_sarsa, trajectories = TD.sarsa(number_episodes)
        sarsa_runs_steps.append(steps_per_episode_sarsa)
        sarsa_trajectories.append(trajectories)
        sarsa_q_tables.append(q_tables)
        # sarsa_trajectories_actions.append(trajectory_actions)
        print(f"Sarsa run {run} done!")
        # if run == runs-1:
        #     optimal_policy_sarsa = TD.get_optimal_policy(action_value_func) # Optimal policy for last run
        env.change_environment(starting_traps) # Changing the env back to starting grid before next run

    np.save('sarsa_steps.npy', sarsa_runs_steps)
    np.save('sarsa_trajectories.npy', np.array(sarsa_trajectories, dtype=object), allow_pickle=True)
    np.save('sarsa_q_tables.npy', np.array(sarsa_q_tables, dtype=object), allow_pickle=True)
    # np.save('sarsa_trajectories_actions.npy', np.array(sarsa_trajectories_actions, dtype=object), allow_pickle=True)
    # env.change_environment(ending_traps) # Change environment to print optimal policy
    # env.print_optimal_policy(optimal_policy_sarsa, "Sarsa 1000 episodes")

def runs_qlearning(runs, number_episodes, epsilon, starting_traps=[], ending_traps=[]):
    qlearning_runs_steps = []
    qlearning_trajectories = []
    # qlearning_trajectories_actions = []
    env = MazeEnv(5, 5, starting_traps)
    TD = TD_learning(env, epsilon, 1, 0.1) # Order: epsilon, gamma = 1 no discount, alpha

    for run in range(runs):
        action_value_func, steps_per_episode_qlearning, trajectories = TD.q_learning(number_episodes)
        qlearning_runs_steps.append(steps_per_episode_qlearning)
        qlearning_trajectories.append(trajectories)
        # qlearning_trajectories_actions.append(trajectory_actions)
        print(f"Qlearning run {run} done!")
        if run == runs-1:
            optimal_policy_qlearning = TD.get_optimal_policy(action_value_func) #Optimal policy for last run
        env.change_environment(starting_traps) # Changing the env back to starting grid before next run

    np.save('qlearning_steps.npy', qlearning_runs_steps)
    np.save('qlearning_trajectories.npy', np.array(qlearning_trajectories, dtype=object), allow_pickle=True)
    # np.save('qlearning_trajectories_actions.npy', np.array(qlearning_trajectories_actions, dtype=object), allow_pickle=True)
    env.change_environment(ending_traps) # Change environment to print optimal policy
    env.print_optimal_policy(optimal_policy_qlearning, "Qlearning 1000 episodes")

def get_median(file_name):
    steps = np.load(file_name)
    median_steps_per_episode = np.median(steps, axis=0)
    return median_steps_per_episode

def get_greedy_policy_episode(q_tables, episodes, starting_traps, ending_traps):
    env = MazeEnv(5, 5, starting_traps)
    TD = TD_learning(env, 0.8, 1, 0.1) # Order: epsilon, gamma = 1 no discount, alpha
    for episode in episodes:
        if episode > 500:
            env.change_environment(ending_traps)
        else:
            env.change_environment(starting_traps)
        optimal_policy = TD.get_optimal_policy(q_tables[0][episode-1])
        env.print_optimal_policy(optimal_policy, "Title")

# def greedy_policy_episodes(self, q_tables, episodes):

#         for episode in episodes:
#             optimal_policy = self.get_optimal_policy(q_tables[0][episode-1])
#             self.env.print_optimal_policy(optimal_policy, "Title")
    

rows = 5
cols = 5
runs = 1
episodes = 1000
starting_traps = [(2,1), (2,2), (2,3), (2,4)]
ending_traps = [(2,0), (2,1), (2,2), (2,3)]
epsilon = 0.8

## Doing 10 runs and saving the results: steps per episode, trajectories at change, trajectory actions
# runs_sarsa(runs, episodes, epsilon, starting_traps, ending_traps) 
# runs_qlearning(runs, episodes, epsilon, starting_traps, ending_traps)

episodes = [499, 500]
sarsa_qtables = np.load('sarsa_q_tables.npy', allow_pickle=True)
get_greedy_policy_episode(sarsa_qtables, episodes, starting_traps, ending_traps)
# print(sarsa_qtables)
# print(len(sarsa_qtables[0][0]))

# ## Taking the average
# median_steps_sarsa = get_median('sarsa_steps.npy')
# median_steps_qlearning = get_median('qlearning_steps.npy')

# ## Plotting median in different scales
# title = 'Median steps per episode'
# # start = 0
# # end = 1000
# # plot_different_scales(start, end, median_steps_sarsa, median_steps_qlearning, epsilon, title)

# start = 499
# end = 550
# plot_different_scales(start, end, median_steps_sarsa, median_steps_qlearning, epsilon, title)

# ## Heatmaps of all trajectories at change
# trajectories_sarsa = np.load('sarsa_trajectories.npy', allow_pickle=True)
# trajectories_qlearning = np.load('qlearning_trajectories.npy', allow_pickle=True)
# plot_multiple_heatmaps(trajectories_sarsa, ending_traps, 5, 2, f"Trajectories at change - SARSA - epsilon {str(epsilon)}", (13, 7))
# plot_multiple_heatmaps(trajectories_qlearning, ending_traps, 5, 2, f"Trajectories at change - Q-learning - epsilon {str(epsilon)}", (13, 7))

# ## Median heatmaps
# trajectories_sarsa = np.load('sarsa_trajectories.npy', allow_pickle=True)
# trajectories_qlearning = np.load('qlearning_trajectories.npy', allow_pickle=True)
## Need to use function episode_trajectories BEFORE using median_heatmap!
# median_heatmap(trajectories_sarsa, ending_traps, f"Median of visits per state - SARSA - 10 runs - epsilon {str(epsilon)}")
# median_heatmap(trajectories_qlearning, ending_traps, f"Median of visits per state - Q-learning - 10 runs - epsilon {str(epsilon)}")

# Checkpoint plots
# checkpoints = [510,515, 525, 545, 585, 645]
# # checkpoints = [499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515]
# state = (1,0)
# checkpoint_plot(checkpoints, trajectories_sarsa, trajectories_qlearning, ending_traps, epsilon, state)

## Visits boxplot
# visits_boxplot(trajectories_sarsa, trajectories_qlearning, (1,0), epsilon, episode=500)

## Making interactive trajectory
# trajectories_sarsa = np.load('sarsa_trajectories.npy', allow_pickle=True)
# trajectory_actions_sarsa = np.load('sarsa_trajectories_actions.npy', allow_pickle=True)
# trajectories_qlearning = np.load('qlearning_trajectories.npy', allow_pickle=True)
# trajectory_actions_qlearning = np.load('qlearning_trajectories_actions.npy', allow_pickle=True)
# for i in range(10):
#     interactive_trajectory(trajectories_sarsa[i], trajectory_actions_sarsa[i], ending_traps, 100, f"Trajectory at change for SARSA from run {str(i)}")
#     interactive_trajectory(trajectories_qlearning[i], trajectory_actions_qlearning[i], ending_traps, 100, f"Trajectory at change for Q-learning from run {str(i)}")
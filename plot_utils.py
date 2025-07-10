import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import pandas as pd
import pygame
import Maze


trajectories_sarsa_sample = [[(0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), 
                              (1, 0), (1, 1), (0, 1), (1, 1), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), 
                              (1, 0), (1, 0), (1, 1), (0, 1), (1, 1), (1, 2), (1, 3), (1, 3), (0, 3), 
                              (0, 3), (0, 4), (0, 4), (0, 4), (1, 4), (1, 4), (0, 4), (0, 4), (0, 3), 
                              (0, 2), (0, 2), (0, 3), (0, 3), (1, 3), (1, 2), (0, 2), (0, 1), (0, 0), 
                              (0, 0), (1, 0), (1, 1), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), 
                              (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 1), (1, 0), (1, 0), 
                              (1, 0), (1, 0), (1, 1), (0, 1), (0, 2), (0, 1), (1, 1), (1, 1), (1, 0), 
                              (1, 1), (1, 1), (1, 2), (0, 2), (0, 3), (1, 3), (1, 3), (0, 3), (0, 4), 
                              (0, 4), (0, 4), (0, 4), (1, 4), (1, 3), (1, 3), (1, 4), (1, 3), (1, 4), 
                              (1, 3), (1, 2), (0, 2), (1, 2), (1, 1), (1, 0), (0, 0), (1, 0), (1, 0), 
                              (1, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0, 0), (0, 0), (0, 1), 
                              (0, 2), (0, 1), (0, 0), (0, 1), (0, 1), (0, 0), (1, 0), (1, 0), (1, 0), 
                              (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), 
                              (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), 
                              (1, 0), (1, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 1), 
                              (1, 1), (0, 1), (0, 2), (0, 1), (0, 2), (0, 2), (0, 2), (0, 2), (0, 1), 
                              (0, 1), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 1), (0, 1), (0, 2), 
                              (0, 1), (1, 1), (1, 0), (1, 0), (1, 1), (0, 1), (0, 1), (0, 1), (0, 0), 
                              (0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (0, 1), (0, 1), (0, 2), (1, 2), 
                              (1, 1), (1, 2), (1, 1), (1, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), 
                              (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 1), (1, 0), (1, 0), 
                              (1, 0), (1, 0), (1, 1), (1, 1), (0, 1), (0, 0), (1, 0), (1, 1), (1, 0), 
                              (1, 0), (1, 0), (1, 0), (1, 1), (0, 1), (1, 1), (0, 1), (0, 1), (1, 1), 
                              (1, 1), (1, 1), (1, 1), (1, 1), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), 
                              (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 0), (1, 1), (1, 2), 
                              (0, 2), (0, 2), (0, 3), (0, 3), (0, 3), (0, 3), (0, 2), (0, 1), (0, 0), 
                              (0, 1), (1, 1), (1, 1), (0, 1), (1, 1), (1, 2), (1, 1), (1, 1), (1, 0), 
                              (1, 0), (1, 0), (0, 0), (0, 1), (0, 0), (0, 1), (0, 0), (0, 0), (0, 0), 
                              (1, 0), (1, 0), (1, 1), (1, 2), (1, 2), (1, 1), (1, 2), (1, 3), (1, 2), 
                              (1, 3), (0, 3), (1, 3), (0, 3), (1, 3), (1, 3), (1, 3), (1, 4), (0, 4), 
                              (0, 4), (0, 3), (1, 3), (1, 3), (0, 3), (0, 4), (1, 4), (1, 4), (1, 4), 
                              (1, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 3), (1, 3), (1, 2), 
                              (1, 2), (1, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), 
                              (0, 0), (0, 1), (1, 1), (0, 1), (0, 0), (0, 1), (0, 1), (1, 1), (1, 0), 
                              (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), 
                              (0, 1), (1, 1), (0, 1), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), 
                              (1, 0), (1, 0), (0, 0), (0, 1), (0, 0), (1, 0), (1, 0), (1, 1), (1, 0), 
                              (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 1), (0, 0), (1, 0), (0, 0), 
                              (0, 0), (0, 1), (0, 0), (0, 1), (0, 1), (1, 1), (1, 0), (1, 1), (1, 0), 
                              (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 1), 
                              (1, 0), (1, 1), (0, 1), (1, 1), (1, 0), (1, 1), (1, 1), (1, 0), (1, 0), 
                              (1, 0), (1, 1), (1, 2), (1, 1), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), 
                              (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), 
                              (1, 1), (1, 2), (1, 1), (0, 1), (0, 1), (0, 1), (0, 1), (1, 1), (1, 2), 
                              (1, 1), (1, 2), (1, 1), (1, 1), (1, 1), (1, 0), (1, 0), (1, 0), (1, 1), 
                              (0, 1), (1, 1), (0, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), 
                              (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0), (0, 0), (0, 0), 
                              (1, 0), (1, 1), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 1), (0, 0), 
                              (1, 0), (1, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 3), (1, 2), (1, 3), 
                              (0, 3), (0, 4), (0, 3), (0, 2), (0, 3), (0, 3), (1, 3), (1, 4), (2, 4), 
                              (2, 4), (3, 4)]]

trajectories_qlearning_sample = [[(0, 0), (0, 1), (0, 0), (0, 0), (0, 0), (0, 1), (0, 2), (0, 2), 
                 (1, 2), (1, 2), (1, 1), (1, 2), (1, 2), (1, 3), (1, 3), (1, 2), 
                 (1, 3), (1, 4), (2, 4), (1, 4), (0, 4), (0, 4), (0, 4), (0, 3), 
                 (0, 2), (0, 2), (0, 3), (0, 4), (0, 4), (0, 4), (0, 4), (0, 3), 
                 (0, 3), (1, 3), (0, 3), (0, 2), (1, 2), (1, 1), (1, 0), (1, 1), 
                 (1, 1), (1, 2), (1, 2), (1, 2), (0, 2), (0, 1), (0, 1), (0, 1), 
                 (1, 1), (1, 0), (1, 0), (1, 1), (0, 1), (1, 1), (0, 1), (0, 0), 
                 (0, 1), (1, 1), (1, 2), (1, 1), (1, 0), (1, 1), (1, 2), (0, 2), 
                 (0, 2), (0, 1), (0, 0), (1, 0), (1, 0), (1, 1), (1, 2), (1, 2), 
                 (1, 2), (1, 3), (1, 4), (0, 4), (1, 4), (1, 3), (0, 3), (0, 4), 
                 (0, 4), (0, 3), (1, 3), (1, 3), (1, 2), (0, 2), (0, 2), (0, 2), 
                 (0, 1), (0, 2), (1, 2), (0, 2), (0, 2), (0, 2), (0, 2), (0, 3), 
                 (0, 3), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), 
                 (0, 4), (1, 4), (1, 4), (1, 3), (0, 3), (1, 3), (1, 3), (1, 4), 
                 (1, 4), (2, 4), (1, 4), (2, 4), (2, 4), (2, 4), (3, 4)]]

## PLOT WITH DIFFERENT SCALES
def plot_different_scales(start_x, end_x, sarsa_data, qlearning_data, epsilon, title):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(17,4))
    fig.suptitle(f'Epsilon = {epsilon}', fontsize=14)

    x_axis = np.arange(start_x, end_x)
    # if start_x != 0:
    #     x_axis_log_log = np.arange(start_x-start_x, end_x-start_x) # Shifting the x-axis to origo for log-log plot
    # else:
    #     x_axis_log_log = x_axis

    # ax[0].plot(x_axis_log_log, sarsa_data[start_x:end_x], label="Sarsa", color="magenta")
    # ax[0].plot(x_axis_log_log, qlearning_data[start_x:end_x], label="Q-learning", color="orange")
    # ax[0].set_xscale('log')
    # ax[0].set_yscale('log')
    # ax[0].set_xlabel('Episodes (log scale)')
    # ax[0].set_ylabel(title + ' (log scale)')
    # ax[0].set_title('Learning Curve Log-Log')
    # ax[0].legend()

    # ax[1].plot(episodes_axis, sarsa_data, label="Sarsa", color="magenta")
    # ax[1].plot(episodes_axis, qlearning_data, label="Q-learning", color="orange")
    # ax[1].set_xscale('log')
    # ax[1].set_xlabel('Episodes (log scale)')
    # ax[1].set_ylabel('Average ' + info + ' per episode (linear)')
    # ax[1].set_title('Learning Curve Log-Lin')
    # ax[1].legend()

    ax[0].plot(x_axis, sarsa_data[start_x:end_x], label="Sarsa", color="magenta")
    ax[0].plot(x_axis, qlearning_data[start_x:end_x], label="Q-learning", color="orange")
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel(title + ' (log scale)')
    ax[0].set_title('Learning Curve Lin-Log')
    ax[0].legend()

    ax[1].plot(x_axis, sarsa_data[start_x:end_x], label="Sarsa", color="magenta")
    ax[1].plot(x_axis, qlearning_data[start_x:end_x], label="Q-learning", color="orange")
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel(title)
    ax[1].set_title('Learning Curve Lin-Lin')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(f'{start_x}_{end_x}.png', dpi=200)
    plt.show()

## HEATMAPS
def count_visits_heatmap(trajectories, grid_size = (5,5)):
    heatmap_matrix = np.zeros(grid_size)
    # for traj i n trajectories:
    for (i,j) in trajectories:
        heatmap_matrix[i,j] += 1
    heatmap_matrix[0,0] -= 1 #We start counting AFTER applying one action and so at start, 
                            # the agent being in the starting state does not count as visit 
    return heatmap_matrix

def list_of_visit_matrix(trajectories): # Making a list of all the matrices with visits per state
    matrix_list = []
    for traj in trajectories:
        visit_matrix = count_visits_heatmap(traj)
        matrix_list.append(visit_matrix)
    return matrix_list

def fix_annotations(heatmap_matrix, traps):
    annotations = np.round(heatmap_matrix).astype(int).astype(str) # To int first for integers, then to string to add traps
    # annotations = heatmap_matrix.astype(str) # Without rounding
    # for (i,j) in traps:
    #     annotations[i,j] = "T" # Represent traps
    return annotations

def median_heatmap(trajectories, traps, algorithm, episode, epsilon): # Trajectories: list of trajectories for a certain episode multiple runs
    matrix_tuple = tuple(list_of_visit_matrix(trajectories))
    all_matrix = np.stack(matrix_tuple)
    median_matrix = np.median(all_matrix, axis=0)
    annotations = fix_annotations(median_matrix, traps)
    sns.heatmap(median_matrix, annot=annotations, cmap="Reds", cbar=True,
                linewidths=1, linecolor="black", fmt="",
                yticklabels=False, xticklabels=False)
    title = 'Median visits per state - ' + algorithm + f' - epsilon {epsilon} - episode {episode}'
    plt.title(title)
    if algorithm == "SARSA":
        save_name = "sarsa"
    else:
        save_name = "qlearning"
    plt.savefig("/Users/cristofertamayo/Desktop/checkpoints/" + save_name + f'_{episode}.png', dpi=200)
    plt.show()

def state_visits(trajectories, state): # Return a list of all the visits for a certain state
    state_visits = []
    all_visits = list_of_visit_matrix(trajectories)
    rows, cols = all_visits[0].shape
    for num in range(len(all_visits)):
        for row in range(rows):
            for col in range(cols):
                if (row, col) == state:
                    state_visits.append(all_visits[num][row][col])
                    break
    return state_visits

def algorithms_visits_dict(sarsa_trajectories, qlearning_trajectories, state):
    sarsa_visits = state_visits(sarsa_trajectories, state)
    qlearning_visits = state_visits(qlearning_trajectories, state)
    return {"Sarsa": sarsa_visits, "Q-learning": qlearning_visits}

def text_list(data, algorithm):
    mean = data[algorithm].mean()
    median = data[algorithm].median()
    std = data[algorithm].std(ddof=1)
    se = std/np.sqrt(len(data[algorithm]))
    q1 = np.quantile(data[algorithm], 0.25)
    q3 = np.quantile(data[algorithm], 0.75)
    iqr = q3 - q1
    text = ['Statistics for ' + algorithm, f'* Mean= {mean:.2f}', f'* Median: {median:.2f}',
            f'* Std: {std:.2f}', f'* IQR: {q3} - {q1} = {iqr:.2f}', f'* SE: {se:.2f}']
    return text, mean, se

def plot_heatmap(data, traps, title):
    heatmap_matrix = count_visits_heatmap(data)
    annotations = np.round(heatmap_matrix).astype(int).astype(str) # To int first for integers, then to string to add traps
    # annotations = heatmap_matrix.astype(str)
    for (i,j) in traps:
        annotations[i,j] = "T" # Represent traps
    
    sns.heatmap(heatmap_matrix, annot=annotations, cmap="Reds", cbar=True,
                linewidths=1, linecolor="black", fmt="",
                yticklabels=False, xticklabels=False)
    plt.title(title)
    plt.show()

def plot_multiple_heatmaps(trajectories, traps, plot_rows, plot_cols, title, figsize):
    fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize) # HUR GÖR VI MED FIGSIZE?
    fig.suptitle(title, fontsize=14)
    matrix_list =[]
    annot_list = []
    for traj in trajectories:
        heatmap_matrix = count_visits_heatmap(traj)
        annotations = fix_annotations(heatmap_matrix, traps)
        matrix_list.append(heatmap_matrix)
        annot_list.append(annotations)
    
    counter_x = 0
    counter_y = 0
    for index in range(len(trajectories)):
        sns.heatmap(matrix_list[index], annot=annot_list[index], cmap="Reds", cbar=True, linewidths=1,
                    linecolor="black", fmt="", yticklabels=False, xticklabels=False,
                    ax=axs[counter_x, counter_y])
        if counter_x == plot_rows-1 and counter_y == plot_cols-1:
            break
        elif counter_y == plot_cols-1:
            counter_x += 1
            counter_y = 0
        else:
            counter_y += 1
    plt.show()

## BOXPLOT
def visits_boxplot(sarsa_traj, qlearning_traj, state, epsilon, episode):
    result_dict = algorithms_visits_dict(sarsa_traj, qlearning_traj, state)
    plt.figure(figsize=(12, 5))
    df = pd.DataFrame(result_dict)
    sns.boxplot(data=df, orient='h', palette=["magenta", "orange"])

    text_s, mean_sarsa, se_sarsa = text_list(df, "Sarsa")
    text_q, mean_qlearning, se_qlearning = text_list(df, "Q-learning")
    text_sarsa = '\n'.join(text_s)
    text_qlearning= '\n'.join(text_q)

    xlim = plt.xlim()
    x_offset = xlim[1] + 0.8
    plt.text(x_offset, 0.2, text_sarsa, bbox=dict(boxstyle="square", facecolor="magenta"))
    plt.text(x_offset, 1.2, text_qlearning, bbox=dict(boxstyle="square", facecolor="orange"))
    plt.errorbar(mean_sarsa, 0, xerr=se_sarsa, fmt='o', color='black', capsize=5, label='Mean ±1 SE')
    plt.errorbar(mean_qlearning, 1, xerr=se_qlearning, fmt='o', color='black', capsize=5)
    plt.subplots_adjust(right=0.8)
    plt.xlabel("Number of visits")
    plt.title(f"Distribution of visits for state {state} - epsilon {epsilon} - episode {episode}") # - 10 runs
    plt.legend()
    plt.savefig("/Users/cristofertamayo/Desktop/checkpoints/" + f'box_{episode}.png', dpi=200)
    plt.show()

## INTERACTIVE TRAJECTORY
def show_action(font, action):
    action_text = ""
    if action == 0:
        action_text = "Up"
    elif action == 1:
        action_text = "Down"
    elif action == 2:
        action_text = "Right"
    elif action == 3:
        action_text = "Left"
    text_surface = font.render(f"Action: {action_text}", True, (255, 255,255)) # White
    screen = pygame.display.get_surface()
    screen.blit(text_surface, (10,230)) # Position
    pygame.display.update()

def get_point(state, cell_size=100):
    row, col = state
    line_point = (col * cell_size + cell_size//2, row * cell_size + cell_size//2)
    return line_point

def draw_line(points, color=(0,0,255), width=3):
    screen = pygame.display.get_surface()
    pygame.draw.lines(screen, color, False, points, width)
    pygame.display.update()

def interactive_trajectory(trajectory, trajectory_actions, traps, fps, title):
    maze = Maze.Maze(5,5, traps, fps=fps)
    maze.render()
    current_step = 0
    points = [(50,50)] # Start state
    pygame.display.set_caption(title)
    font = pygame.font.SysFont("Arial", 24)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    if current_step < len(trajectory)-1:
                        current_step += 1
                        maze.current_position = trajectory[current_step]
                        action_taken = trajectory_actions[current_step]
                        points.append(get_point(trajectory[current_step])) # Point to draw line
                        #print(points)
                        maze.render()
                        show_action(font, action_taken)
                        # draw_line(points)
                elif event.key == pygame.K_LEFT:
                    if current_step > 0:
                        current_step -= 1
                        maze.current_position = trajectory[current_step]
                        action = trajectory_actions[current_step]
                        maze.render()
                        show_action(font, action)

## CHECKPOINT TRAJECTORIES
def episode_trajectories(episode, trajectories): # Gets all the trajectories for a specific episode through all the runs
    episode_traj = []
    episode_index = episode - 1 
    for run in range(len(trajectories)):
        episode_traj.append(trajectories[run][episode_index])
    return episode_traj

def checkpoint_plot(checkpoints, sarsa_trajectories, qlearning_trajectories,
                    traps, epsilon, state):
    for episode in checkpoints:
        sarsa_traj = episode_trajectories(episode, sarsa_trajectories) # Get trajectories for specific episode
        qlearning_traj = episode_trajectories(episode, qlearning_trajectories)
        # Plot median heatmap and boxplot
        median_heatmap(sarsa_traj, traps, "SARSA", episode, epsilon)
        median_heatmap(qlearning_traj, traps, "Q-learning", episode, epsilon)
        visits_boxplot(sarsa_traj, qlearning_traj, state, epsilon, episode)

# checkpoints = [501, 502]
# traps = [(2,0), (2,1), (2,2), (2,3)]
# sarsa_trajectories = np.load('sarsa_trajectories.npy', allow_pickle=True)
# qlearning_trajectories = np.load('qlearning_trajectories.npy', allow_pickle=True)
# epsilon = 0.8
# state = (1,0)
# checkpoint_plot(checkpoints, sarsa_trajectories, qlearning_trajectories, traps, epsilon, state)


# traps = [(2,0), (2,1), (2,2), (2,3)]
# traj_sample2 = trajectories_sarsa_sample + trajectories_qlearning_sample
# plot_multiple_heatmaps(traj_sample2, traps, 5, 2, "Trying multiple trajectories")

# plot_heatmap(trajectories_sarsa_sample[0], traps, "Number of visits - SARSA episode 501")
# plot_heatmap(trajectories_qlearning_sample[0],traps, "Number of visits - Q-learning episode 501")


## TESTING INTERACTIVE_TRAJECTORY
# trajectory = trajectories_sarsa_sample[0]
# trajectory_actions_sample = [2, 3, 0, 2, 3, 3, 0, 1, 3, 3, 3, 1, 2, 3, 0, 0, 3, 3, 1, 1, 1, 1, 1, 0, 2, 2, 1, 1, 3, 2, 0, 3, 3, 1, 0, 0, 1, 1, 1, 0, 0, 3, 1, 1, 3, 1, 2, 0, 1, 3, 1, 1, 3, 3, 2, 3, 1, 1, 1, 2, 3, 1, 3, 0, 1, 1, 3, 1, 3, 1, 1, 2, 3, 3, 2, 3, 0, 1, 1, 2, 1, 1, 2, 3, 3, 1, 3, 1, 0, 0, 2, 2, 0, 3, 1, 1, 2, 1, 0, 3, 0, 0, 3, 1, 3, 1, 0, 1, 0, 3, 2, 3, 3, 3, 2, 3, 1, 0, 3, 3, 3, 1, 3, 2, 2, 3, 2, 1, 3, 0, 2, 0, 2, 3, 1, 3, 3, 3, 0, 1, 3, 1, 1, 0, 0, 1, 3, 1, 1, 3, 3, 2, 3, 1, 2, 1, 3, 3, 1, 0, 1, 0, 1, 3, 3, 1, 3, 2, 3, 1, 1, 3, 3, 1, 3, 2, 3, 2, 3, 3, 2, 3, 1, 3, 2, 3, 1, 3, 1, 3, 0, 1, 0, 1, 1, 1, 3, 0, 1, 2, 2, 2, 0, 0, 3, 1, 3, 3, 0, 1, 0, 0, 0, 1, 2, 0, 3, 1, 2, 3, 2, 3, 1, 3, 1, 1, 0, 3, 1, 0, 2, 1, 0, 3, 2, 3, 0, 3, 3, 3, 0, 2, 1, 0, 3, 1, 2, 3, 1, 0, 0, 0, 0, 3, 0, 1, 0, 2, 3, 0, 1, 2, 1, 0, 3, 1, 1, 2, 2, 1, 1, 2, 0, 1, 3, 3, 3, 1, 0, 3, 3, 1, 0, 2, 2, 0, 3, 3, 0, 3, 1, 2, 1, 1, 3, 1, 3, 0, 0, 2, 1, 3, 3, 0, 3, 2, 0, 1, 1, 1, 0, 0, 1, 1, 3, 1, 2, 3, 1, 3, 3, 1, 1, 3, 2, 0, 0, 3, 1, 0, 1, 2, 2, 0, 1, 2, 3, 3, 2, 0, 2, 0, 3, 1, 3, 0, 1, 3, 3, 0, 2, 1, 1, 1, 3, 2, 3, 1, 0, 2, 3, 2, 2, 2, 1, 1, 3, 3, 3, 2, 3, 3, 0, 1, 1, 3, 2, 3, 0, 0, 1, 1, 1, 2, 3, 3, 0, 3, 2, 1, 2, 1, 0, 2, 2, 1, 1, 1, 1]
# traps = [(2,0), (2,1), (2,2), (2,3)]
# interactive_trajectory(trajectory, trajectory_actions_sample, traps, 100)

# ## TESTING MEDIAN HEATMAP
# sarsa_trajectories = np.load('sarsa_trajectories.npy', allow_pickle=True)
# qlearning_trajectories = np.load('qlearning_trajectories.npy', allow_pickle=True)
# median_heatmap(sarsa_trajectories, traps, "Median of visits per state - SARSA - 10 runs - epsilon 0.8")

# epsilon = 0.8
# visits_boxplot(sarsa_trajectories, qlearning_trajectories, (1,0), epsilon)

# print(np.median([62, 4, 6, 43, 20, 24, 50, 22, 21, 18]))


## TESTING NUMPY

# steps_per_epi = [[1,2,3], [4,5,6], [2,3,10]]
#np.save('testing_numpy.npy', steps_per_epi)
# getting_steps_back = np.load('testing_numpy.npy')
# print(getting_steps_back)


# np.save('testing_vstack.npy', np.vstack(steps_per_epi))
# getting_vstack_back = np.load('testing_vstack.npy')
# print(getting_vstack_back)

# getting_steps_back = np.load('testing_numpy.npy')
# getting_vstack_back = np.load('testing_vstack.npy')
#np.save('last_test.npy', np.vstack((getting_steps_back, getting_vstack_back)))

# get_last_back = np.load('last_test.npy')
# print(get_last_back[0])

## Testing np.save with more arguments
# trajectories = [
#     [(0, 0), (0, 1), (0, 2)],
#     [(0, 0), (1, 0), (1, 1), (1,2)],
#     [(0, 0), (1, 0), (0,0), (0,1), (0,2), (0,3)]
# ]

# np.save("my_trajectories.npy", np.array(trajectories, dtype=object), allow_pickle=True)

# traps = [(2,0), (2,1), (2,2), (2,3)]
# trajectories_loaded = np.load("my_trajectories.npy", allow_pickle=True)

# for trajectory in trajectories_loaded:
#     plot_heatmap(trajectory, traps, "trying_with_npSave")

#plot_multiple_heatmaps(trajectories_loaded, traps, 3, 2, "Trying multiple trajectories")

## TESTING NP.MEAN

# steps_per_epi = [[1,2,3], [4,5,6], [2,3,10]]
# np.save('testing_steps.npy', steps_per_epi)
# getting_steps_back = np.load('testing_steps.npy')
# print(getting_steps_back)
# avg_steps = getting_steps_back.mean()
# print(avg_steps)
# avg_steps2 = getting_steps_back.mean(axis=0)
# print(avg_steps2)


## TESTING NP.MEDIAN with matrix of multiple dimensions

# A = np.array([
#     [140, 58, 70],
#     [14, 13, 10]
# ])
# B = np.array([
#     [1,2,3],
#     [4,5,6]
# ])
# C = np.array([
#     [5,6,7],
#     [8,9,10]
# ])

# all_matrix = np.stack((A, B, C))
# print(all_matrix)
# print(all_matrix.shape)
# print(all_matrix.ndim)


# for i in range(3):
#     print(all_matrix[i].shape)

# unstacking = np.unstack(all_matrix, axis=0)
# print(unstacking)

# trying = np.dsplit(all_matrix, 2)


# print(np.median(all_matrix, axis=0))

# matrix = np.array([
#     [[140, 58, 70],
#      [14, 13, 10]],

#     [[1,2,3],
#      [4,5,6]],

#     [[5,6,7],
#      [8,9,10]]
# ])
# print(matrix)
# print(np.median(matrix, axis=0))
# print(np.median(matrix, axis=1))
# print(np.median(matrix, axis=2))

## Trying tuples

# result = [A, B, C]
# tuple_res = (A, B, C)
# print(result)
# print(tuple_res)
# result_tuple = tuple(result)
# print(result_tuple)
# print(tuple_res == result_tuple)
# stacking = np.stack(result_tuple)
# print(np.median(stacking, axis=0))

# Trying median steps

# sarsa_steps = [[127, 97, 369,  47,  41,  31], 
#                [19,  39, 226, 57,  46,  63],
#                [ 207, 309,  1.5,  10,  12,  55]] # List with steps per episode for 2 runs
# np.save('sarsa_try.npy', sarsa_steps)
# sarsa_back = np.load('sarsa_try.npy')
# print(sarsa_back)
# # print(sarsa_back.mean(axis=0))
# print(np.median(sarsa_back, axis=0))


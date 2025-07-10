import numpy as np
import gymnasium as gym
import pygame
from Maze import Maze


from gym import Env
from gym import spaces

class MazeEnv(Env):
    def __init__(self, rows, cols, traps = [], fps=4000): # maze_size: tuple of the form (rows,columns)
         super(MazeEnv, self).__init__()

         self.rows = rows
         self.cols = cols
         self.fps = fps
         self.maze = Maze(rows, cols, traps, fps = fps)
         self.total_reward = 0
         self.wall_counter = 0 # Counting the number of times the agent hits a wall

         self.action_space = spaces.Discrete(4)

         # How states will be observed by the agent, in this case coordinates
         self.observation_space = spaces.Box(low=0, high=max(self.rows, self.cols), shape=(2,), dtype=np.int32)
    
    @property
    def get_states(self):
        states = []
        for row in range(self.rows):
            for col in range(self.cols):
                states.append((row, col))
        return states

    def reset(self):
        # When reseting for a new episode
        self.maze.reset()
        self.total_reward = 0
        self.wall_counter = 0
        return self.maze.current_position
    
    def step(self, action): # Taking an action
        self.maze.check_events()
        #current_position = self.maze.current_position
        done = self.maze.take_action(action)
        #updated_position = self.maze.current_position


        if done:
            reward = 0 # Get to the goal we obtain reward 0 since we are counting steps
        else:
            reward = -1 # Each step gives reward -1. In case of trap, extra step so same reward.

        # if done:
        #     reward = 0
        # elif self.maze.current_position in self.maze.traps: # If going into the trap!
        #     reward = -20
        #     done = True # End the episode
        # elif current_position == updated_position: # hitting a wall
        #     reward = -1
        #     self.wall_counter += 1
        # else:
        #     reward = 0
        #self.render()
        self.total_reward += reward
        #print(f"Current: {current_position}, Updated: {updated_position}, Reward: {reward}, Total reward: {self.total_reward}")
        return self.maze.current_position, reward, done

    def render(self, mode="human"):
        self.maze.render()

    def change_environment(self, new_traps):
        self.maze.change_maze(new_traps)

    def degree_of_action(self, action):
        ## Function for determining the rotation of the arrow.
        ## Arrow is pointing right from the beginning
        if action == 0: # Up
            degree = 90
        elif action == 3: # left
            degree = 180
        elif action == 1: # Down
            degree = 270
        else:
            degree = 0 # Right
        return degree
    
    def degree_list(self, optimal_action): ## Get the degrees of rotation for the arrow for the optimal action / ties.
        degree_list = []
        if optimal_action in [0, 1, 2, 3]: ## Optimal action is just a number, no ties list
            degree = self.degree_of_action(optimal_action)
            degree_list.append(degree)
        else: ## We have a list of ties
            for tie_action in optimal_action: # Loop throug all action in the ties list
                degree = self.degree_of_action(tie_action)
                degree_list.append(degree)
        return degree_list
    
    def position(self, degree, x, y): ## Obtain the right coordinates for printing the arrow on the grid
        if degree == 0: # Right
            position = (y * 100 + 35, x * 100 + 15)
        elif degree == 90: # Up
            position = (y * 100 + 14, x * 100 - 5)
        elif degree == 180: ## Left
            position = (y * 100 - 5, x * 100 + 16)
        else: ## Down, i.e. degree == 270
            position = (y * 100 + 15, x * 100 + 36)
        return position

    def print_optimal_policy(self, optimal_policy, title): # Policy doesnät have to be optimal!
        #start_ticks = pygame.time.get_ticks()
        self.maze.reset()
        self.maze.render()
        pygame.display.set_caption(title)
        arrow = pygame.image.load("arrow.png").convert_alpha()
        little_arrow = pygame.transform.scale_by(arrow, 0.12)
        loop = True
        while loop:
            x = 0 # State coordinate
            y = 0 # State coordinate
            if loop:
                for number in range(len(optimal_policy)):
                    optimal_action = optimal_policy[number]
                    if number in [5, 10, 15, 20]: # These are states (1,0), (2,0), (3,0) and (4,0)
                        x += 1
                        y = 0
                    degree_list = self.degree_list(optimal_action)
                    for degree in degree_list:
                        result_arrow = pygame.transform.rotate(little_arrow, degree)
                        if (x,y) != self.maze.goal_position and (x,y) not in self.maze.traps: # No arrows on traps nor goal position
                            if len(degree_list) == 1: # No ties
                                position = (y * 100 + 15, x * 100 + 15)
                            else: # List with ties
                                position = self.position(degree, x, y)
                            self.maze.window_surface.blit(result_arrow, position)
                        pygame.display.flip()
                    y += 1
            
            # elapsed_ms = pygame.time.get_ticks() - start_ticks
            # if elapsed_ms > 2000:   # 2000 ms = 2 s
            #     break
            for event in pygame.event.get():
                if event.type == pygame.QUIT: # Clicking exit in window
                    pygame.image.save(self.maze.window_surface, title + ".jpeg")
                    loop = False
        return


# env = MazeEnv(5, 5) # traps=[(2,2)]
# example_policy = [[2,3], [0,1], [0,2], [0,3], 0, 
#                   3, [1,2], [1,3], 0, 2,
#                   3, 2, [0, 1, 2, 3], 2, 2,
#                   3, 2, 2, 2, 2,
#                   1, 1, 1, 1, 1]
# env.print_optimal_policy(example_policy, "Example policy")

#print(env.degree_list([1,2, 3]))

# Hitting walls

# env.reset()


# current_state = env.maze.current_position
# while True:
#     print("Current state is: " + str(current_state))
#     action = example_policy[env.maze.state_tuple_to_number(current_state)]
#     print("Action taken is: " + str(action))
#     current_state, reward, done = env.step(action)
#     print("Reward is: " + str(reward))
#     print("Next state is: " + str(current_state))
#     if done:
#         print("Episode done")
#         break



# TRYING example policy going through trap

# env.reset()
# example_policy = [1, 2, 2, 1, 1, 
#                   1, 1, 1, 1, 1,
#                   2, 2, 2, 2, 1,
#                   2, 2, 2, 2, 1,
#                   0, 2, 2, 0, 1]
# env.print_optimal_policy(example_policy, "Example policy")



# current_state = env.maze.current_position
# while True:
#     print("Current state is: " + str(current_state))
#     action = example_policy[env.maze.state_tuple_to_number(current_state)]
#     print("Action taken is: " + str(action))
#     current_state, reward, done = env.step(action)
#     print("Reward is: " + str(reward))
#     print("Next state is: " + str(current_state))
#     if done:
#         print("Episode done")
#         break

# TRYING an episode. Seeing how each steps gets updated during a run. 
# Counting the reward, the total reward and the number of steps taken
# in the episode. This is then used in Sarsa and Q-learning.

# env = MazeEnv(5, 5, fps=1)
# env.reset()
# print("Episode starts!")
# print(f"Starting state: {env.maze.current_position}, Total Reward: {env.total_reward}")

# optimal_policy = [1, 2, 2, 1, 1, 
#                   1, 1, 1, 1, 1,
#                   1, 2, 2, 2, 1,
#                   1, 2, 2, 2, 1,
#                   2, 2, 2, 2, 1]

# total_steps = 0
# done = False
# while not done:
#     random_action = env.action_space.sample() # Takin a random action

#     old_position = env.maze.current_position
#     #optimal_action = optimal_policy[env.maze.state_tuple_to_number(old_position)]

#     new_position, reward, done = env.step(random_action)
#     total_steps += 1

#     if total_steps == 5:
#         print("Traps add!")
#         env.change_environment([(2,1), (2,2), (2,3), (2,4)])

#     print(f"Current state: {old_position}, Action: {random_action}, Reward: {reward}, Next state: {new_position}, Total Reward: {env.total_reward}, Total steps: {total_steps}")


# TRYING print_optimal_policy

# opt_policy_ex = [3, 1, 2, 1, 1, 2, 2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2]
# env.print_optimal_policy(opt_policy_ex)













##        # Representing the maze as a matrix with 0:s as free spaces and 1:s as traps.
##         self.maze = np.array([
##            [0, 0, 1, 0, 0],
##            [0, 1, 0, 0, 0],
##            [0, 1, 0, 1, 0], 
##            [0, 0, 0, 1, 0],
##            [1, 1, 0, 0, 0]  # Goal: (4,4)
##         ])
##
##         # The size of the maze
##         self.maze_size = self.maze.shape
##         
##         self.state = None # Initial state, see reset()
##         self.start = (0,0)
##         self.goal = (4,4)
##
##
##
##         # Actions: 0 is up, 1 is down, 2 is right, 3 is left
##         self.action_space = spaces.Discrete(4)
##
##         # Reward
##         self.reward_goal = 100
##         self.total_reward = 0

##    def render(self):
##        return
##
##
##    def step(self, action):
##        # Moving the agent after a certain action
##        
##        x_coordinate = self.state[0]
##        y_coordinate = self.state[1]
##
##        if action == 0 and x_coordinate != 0:
##            # Go up
##            x_coordinate -= 1
##        elif action == 1 and x_coordinate != self.maze_size[0]-1:
##            # Go down
##            x_coordinate += 1
##        elif action == 2 and y_coordinate != self.maze_size[1]-1:
##            # Go right
##            y_coordinate += 1
##        elif action == 3 and y_coordinate != 0:
##            # Go left
##            y_coordinate -= 1
##
##        self.state = (x_coordinate, y_coordinate) # Update the state
##
##        if self.state == self.goal:
##            done = True
##        else:
##            ?
##            
##        return
##
##    def close(self): # In case we want 
##        return

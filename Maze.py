import pygame
import sys
import numpy as np

class Maze():

    def __init__(self, rows=5, cols=5, traps=[],
                 start_position=(0,0), goal_position=(4,4), fps=1):
        self.rows = rows
        self.cols = cols
        self.traps = traps # list of tuples of coordinates of the traps

        self.fps = fps
        self.init_pygame()

        self.start_position = start_position
        self.goal_position = goal_position
        self.reset()
        self.current_position = start_position

    @property
    def size(self):
        return (self.rows, self.cols)

    def init_pygame(self):
        pygame.init()
        pygame.display.init()

        self.clock = pygame.time.Clock()

        self.cell_height = 100
        self.cell_width = 100 
        self.cell_size = (self.cell_width, self.cell_height) # the size of each square in the grid

        self.window_size = (self.cell_width * self.cols, self.cell_height * self.rows) # Make cells depending on the number of cols and rows
        self.window_surface = pygame.display.set_mode(self.window_size) 

        # No images for now

    def take_action(self, action):
        
        x_coordinate = self.current_position[0] #current row
        y_coordinate = self.current_position[1] # current column

        if action == 0 and x_coordinate > 0:
            x_coordinate -= 1 # Go up
        elif action == 1 and x_coordinate < self.rows-1:
            x_coordinate += 1 # Go down
        elif action == 2 and y_coordinate < self.cols - 1:
            y_coordinate += 1 # Go right
        elif action == 3 and y_coordinate > 0:
            y_coordinate -= 1 # Go left

        if (x_coordinate, y_coordinate) not in self.traps: # If the new state is a trap, we don't update the current position
            self.current_position = (x_coordinate, y_coordinate) 

        done = False
        if self.current_position == self.goal_position:
            done = True

        return done

    def reset(self):
        self.current_position = self.start_position

    def render(self):
        self.window_surface.fill((255, 255, 255))
        for col in range(self.cols):
            for row in range(self.rows):
                col_position = col * self.cell_width
                row_position = row * self.cell_height
                rectangel = (col_position, row_position, self.cell_width, self.cell_height)
                pygame.draw.rect(self.window_surface, (0, 0, 0), rectangel, 1) 

                if self.current_position == (row, col):
                    rectangel = (col_position, row_position, self.cell_width, self.cell_height)
                    pygame.draw.rect(self.window_surface, (255, 255, 0), rectangel) # Yellow = current position
                elif self.goal_position == (row, col):
                    rectangel = (col_position, row_position, self.cell_width, self.cell_height)
                    pygame.draw.rect(self.window_surface, (0, 255, 0), rectangel) # Green = goal
                elif (row, col) in self.traps:
                    rectangel = (col_position, row_position, self.cell_width, self.cell_height)
                    pygame.draw.rect(self.window_surface, (0, 0, 0), rectangel) # Black = trap

        pygame.display.flip()
        self.clock.tick(self.fps)

    def state_tuple_to_number(self,state_tuple):
        state_number = 0
        for row in range(self.rows):
            for col in range(self.cols):
                if state_tuple == (row, col):
                    return state_number
                state_number += 1   

    def check_events(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT: # Clicking exit in window
                pygame.quit()
                sys.exit()

            if(event.type == pygame.KEYDOWN):
                if(event.key == pygame.K_ESCAPE): # Clicking escape on keyboard
                    pygame.quit()
                    sys.exit()
    
    def change_maze(self, new_traps):
        self.traps = new_traps

    def print_trajectory(self, trajectory): ## DO WE NEED THIS?
        for state in trajectory:
            self.current_position = state
            self.render()
            self.check_events()

    





def pick_random_action():
    action = np.random.choice(4)
    return action

# maze = Maze(5,5, fps=10) #traps=[(2,1), (2,2), (2,3), (2,4)]

# traj = [(0, 0), (1, 0), (1, 1), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 0), (1, 0), (1, 0), (1, 1), (1, 1), (1, 2), (1, 2), (1, 1), (1, 2), (1, 2), (0, 2), (0, 1), (1, 1), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (0, 1), (0, 0), (0, 0), (0, 1), (1, 1), (0, 1), (0, 0), (0, 1), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 0), (1, 0), (0, 0), (0, 1), (1, 1), (1, 1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 1), (1, 1), (1, 1), (1, 2), (0, 2), (0, 2), (0, 3), (0, 4), (1, 4), (0, 4), (0, 4), (1, 4), (1, 3), (1, 3), (1, 4), (1, 3), (1, 2), (0, 2), (0, 1), (0, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (0, 1), (0, 1), (0, 2), (0, 3), (1, 3), (0, 3), (0, 3), (0, 4), (0, 3), (0, 2), (0, 3), (0, 4), (0, 4), (0, 4), (0, 4), (0, 3), (0, 3), (0, 4), (0, 4), (0, 4), (0, 3), (1, 3), (1, 4), (0, 4), (0, 4), (0, 3), (1, 3), (0, 3), (0, 4), (0, 3), (1, 3), (1, 3), (1, 4), (1, 4), (1, 3), (1, 4), (1, 4), (1, 4), (1, 4), (1, 3), (0, 3), (0, 2), (1, 2), (1, 3), (0, 3), (0, 3), (0, 3), (0, 2), (0, 3), (1, 3), (1, 3), (1, 2), (1, 1), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (0, 3), (0, 2), (1, 2), (0, 2), (0, 1), (0, 1), (0, 1), (1, 1), (1, 1), (1, 0), (1, 0), (1, 0), (1, 1), (1, 0), (0, 0), (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (1, 1), (1, 2), (1, 2), (1, 1), (1, 2), (1, 3), (1, 3), (0, 3), (0, 2), (1, 2), (1, 3), (1, 3), (0, 3), (1, 3), (1, 2), (1, 1), (1, 1), (0, 1), (1, 1), (1, 0), (1, 0), (1, 0), (1, 1), (1, 2), (1, 3), (1, 3), (0, 3), (0, 3), (0, 3), (1, 3), (1, 4), (1, 3), (1, 3), (1, 2), (1, 3), (1, 4), (0, 4), (0, 3), (0, 3), (0, 2), (0, 2), (0, 3), (1, 3), (1, 2), (1, 2), (0, 2), (0, 3), (0, 4), (0, 4), (1, 4), (0, 4), (0, 4), (0, 4), (1, 4), (1, 4), (0, 4), (1, 4), (1, 4), (1, 3), (1, 4), (1, 4), (1, 4), (2, 4), (1, 4), (2, 4), (2, 4), (3, 4), (2, 4), (2, 4), (3, 4), (3, 4)]
# maze.print_trajectory(traj)

#counter = 0
# while True:
#     maze.render()
#     counter += 1
#     if counter == 5:
#         break


# TRYING state_tuple to number_state
# for i in range(5):
#     for j in range(5):
#         print("State (" + str(i) + "," + str(j) + ") is equivalent to number " + 
#               str(maze.state_tuple_to_number((i, j))))


# maze = Maze(5,5, fps=1) #traps=[(2,1), (2,2), (2,3), (2,4)]
# counter = 0
# episode = 0
# actions = [1, 1, 1, 1, 2, 2, 2, 2]
# actions2 = [3, 1, 3, 1, 3, 1, 3, 1]

# while True:
#     maze.check_events()
#     #print("Current state is: " + str(maze.current_position))
#     if episode == 0:
#         action = actions[counter]
#     else:
#         action = actions2[counter]

#     #print("Action taken is: " + str(action))

#     print(f"Current state is: {maze.current_position}, Action taken is: {action}")

#     done = maze.take_action(action)
#     maze.render()
#     counter += 1

#     #Adding a trap after 10 steps.

#     # if counter == 10:
#         #maze.change_maze([(2,1), (2,2), (2,3), (2,4)])

    
#     if done and episode == 1:
#         break

#     if done:
#         maze.start_position = (0,4)
#         maze.current_position = maze.start_position
#         maze.goal_position = (4,0)
#         counter = 0
#         episode +=1


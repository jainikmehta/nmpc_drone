import numpy as np

# This should take map area, initial and final pos, obstacles.abs
# Here can use different trajectory generator
class map_2d:
    # creates 2d array with obstacle information
    # obstacle
    def __init__(self, height, width, num_obstacles, start, goal):
        self.height = height
        self.width = width
        self.num_obstacles = num_obstacles
        self.goal = goal
        self.start = start

    def generate_random_obstacle_map(self):
        map = np.array()
        
        return map
# Original code downloaded from https://gist.github.com/kfeeeeee/e81aeeff0516cfd7645c8e99cd4fa315#file-gridworld-py
# Modifications made by Fabian Ruehle
# Note that I completely change the game (rules, rewards, goal, colors)

import numpy as np
np.random.seed(4)
import itertools
import scipy.misc
import matplotlib as mpl
mpl.use('TkAgg')  # needed on mac with virtual environment
import matplotlib.pyplot as plt


# The gridworld environment
class GameEnv:
    def __init__(self):
        # initialization of the world
        self.sizeX = 5
        self.sizeY = 5
        self.num_pits=7
        self.state = ()  # A state in gridworld is just the (x,y) coordinate pair of the worker
        plt.ioff()  # there is currently a bug for Mac users which requires turning this off
        self.world_canvas = plt.figure("Maze")
        self.world_canvas.suptitle('Blue: Worker, Red: Pitfalls, Green: Exit')
        self.im = None
        plt.axis("off")
        self.objects = []
        self.initial_x = 0
        self.initial_y = 0
        self.gave_up = False
        self.fell = False

        # We want the worker to solve the maze as fast as possible without falling into the pits:
        # *)   -1 for each step (penalty to solve it quickly)
        # *)  -50 for each pitfall (penalty for falling into the pit)
        # *) +100 for finding the exit (reward for solving the maze)
        # *)   -2 for running into a wall / not moving at all
        self.step_penalty = -1.
        self.pitfall_penalty = -50.
        self.exit_reward = 100000.
        self.no_move_penalty = -2.

        # Actions in gridworld: move up, down, left, right
        self.action_space = [0, 1, 2, 3]  # up, down, left, right

        # keep track of the total number of steps and the steps that were taken in the game
        self.steps = 0
        self.steps_taken = []

        # maximal n umber of steps before we give up solving the maze
        self.max_steps = 1000

        # initialize and plot the world
        self.world = self.initialize_world()
        self.snapshot_world = self.initialize_world()

    # initialize a new random world
    def initialize_world(self):
        self.objects = []

        # 1.) The first parameter is the name of the object
        # 2.) The second parameter is the reward / penalty:
        # 3.) The third parameter is the position of the object in the world
        # 4.) Ignore the other parameters, they are just used for drawing the world (box sizes and color)

        # fix position of exit and worker
        # maze_exit = GameOb('exit', self.exit_reward, self.new_position(), 1, [0, 1, 0, 1])
        maze_exit = GameOb('exit', self.exit_reward, [4,4], 1, [0, 1, 0, 1])
        self.objects.append(maze_exit)
        # worker = GameOb('worker', None, self.new_position(), 1, [0, 0, 1, 1])
        worker = GameOb('worker', None, [0,0], 1, [0, 0, 1, 1])
        self.objects.append(worker)
        for i in range(self.num_pits):  # add pitfalls
            pitfall = GameOb('pitfall', self.pitfall_penalty, self.new_position(), 1, [1, 0, 0, 1])
            self.objects.append(pitfall)

        # store the initial (x,y) coordinates for a reset
        self.initial_x = worker.x
        self.initial_y = worker.y

        # show the world
        world = self.render_world()

        # initialize/ reset the variables
        self.reset()

        # plot the world
        plt.ioff()
        self.im = plt.imshow(world, interpolation="nearest")

        return world

    # reset the world to its initial configuration, ignore this
    def reset(self):
        self.steps = 0
        self.steps_taken = []
        self.gave_up = False
        self.fell = False
        self.state = (self.initial_x, self.initial_y)
        # np.random.seed(random.randint(0, 100000))
        for obj in self.objects:
            if obj.name == 'worker':
                obj.x = self.initial_x
                obj.y = self.initial_y
                break

    # move through the world
    # 0 - up, 1 - down, 2 - left, 3 - right
    def move_worker(self, direction):

        # identify the worker amongst the gridworld objects
        worker = None
        others = []
        for obj in self.objects:
            if obj.name == 'worker':
                worker = obj
            else:
                others.append(obj)

        worker_x = worker.x
        worker_y = worker.y

        # overall reward/penalty
        reward = self.step_penalty  # penalize each move

        # update the position of the worker in gridworld (move if possible)
        if direction == 0 and worker.y >= 1:
            worker.y -= 1
        if direction == 1 and worker.y <= self.sizeY - 2:
            worker.y += 1
        if direction == 2 and worker.x >= 1:
            worker.x -= 1
        if direction == 3 and worker.x <= self.sizeX - 2:
            worker.x += 1

        # move was illegal
        if worker.x == worker_x and worker.y == worker_y:
            reward = self.no_move_penalty

        # update to new position
        for i in range(len(self.objects)):
            if self.objects[i].name == 'worker':
                self.objects[i] = worker
                break

        # check whether new field is a special field (exit/pitfall) and compute reward/penalty
        is_maze_solved = False
        for other in others:
            if worker.x == other.x and worker.y == other.y:  # the worker ran into an object
                if other.name == "exit":  # the object was an exit
                    is_maze_solved = True
                    # print "I found the exit,yay!"
                    reward = other.reward
                    break  # we can exit the loop here since we can only run into one object
                elif other.name == "pitfall":  # the object was a pitfall
                    is_maze_solved = False
                    reward = other.reward
                    self.fell = True
                    break   # we can exit the loop here since we can only run into one object

        return reward, is_maze_solved

    # perform the step, collect the reward, check whether you have reached the exit
    def step(self, action, update_view=True):

        # collect the reward/punishment for the field the worker ends up in and check whether the exit was reached
        reward, done = self.move_worker(action)

        self.steps += 1
        self.steps_taken.append(action)

        # give up
        if self.steps >= self.max_steps and not done:
            done = True
            self.gave_up = True

        # fell into pit
        if self.fell:
            done = True

        # this just updates the graphic output of the world
        if update_view:
            # world = self.render_world()
            # plt.imshow(world, interpolation="nearest")
            self.im.set_array(self.render_world())
            plt.draw()

        # return the new state, the penalty/reward for the move and whether gridworld is solved/given up on
        return self.get_state(), reward, done

    # get the current state
    def get_state(self):
        for obj in self.objects:
            if obj.name == 'worker':
                return (obj.x, obj.y)

    # check whether an action is possible, i.e. whether a wall is blocking the way
    def is_possible_action(self, action):
        is_possible = False
        if action == 0 and self.state[1] >= 1:
            is_possible = True
        if action == 1 and self.state[1] <= self.sizeY - 2:
            is_possible = True
        if action == 2 and self.state[0] >= 1:
            is_possible = True
        if action == 3 and self.state[0] <= self.sizeX - 2:
            is_possible = True

        return is_possible



    ####################################################################################################################
    # ignore the code from here on, it just draws the world and represents the objects in the game.                    #
    ####################################################################################################################
    def close_world_display(self):
        plt.close("Gridworld")

    def new_position(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        current_position = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in current_position:
                current_position.append((objectA.x, objectA.y))
        for pos in current_position:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]

    def render_world(self):
        a = np.zeros([self.sizeY + 2, self.sizeX + 2, 4])
        a[0:, 0, 3] = 1  # left wall
        a[0, 0:, 3] = 1  # top wall
        a[0:, self.sizeX + 1, 3] = 1  # right wall
        a[self.sizeY + 1, 0:, 3] = 1  # bottom wall
        a[1:-1, 1:-1, :] = 1
        for item in self.objects:
            if a[item.y + 1, item.x + 1, 0] == 1 and a[item.y + 1, item.x + 1, 1] == 1 and a[item.y + 1, item.x + 1, 2] == 1:  # is completely white
                for i in range(len(item.channel)): a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, i] = item.channel[i]
            else:  # other object on the field, overlay worker with pitfalls / exit
                for i in range(len(item.channel)):
                    if a[item.y + 1, item.x + 1, i] == 0:
                        a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, i] += item.channel[i]
        a = scipy.misc.imresize(a[:, :], [84, 84, 4], interp='nearest', mode="RGBA")
        return a


# This represents an object in the game: worker, pitfall, exit
class GameOb:
    def __init__(self, name, reward, coordinates, size, RGBA):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.channel = RGBA
        self.reward = reward
        self.name = name

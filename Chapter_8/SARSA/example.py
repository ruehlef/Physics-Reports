# Based on a SARSA implementation downloaded from https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rl/sarsa.py
# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
# Modified and extended by Fabian Ruehle
from __future__ import print_function

# Explore Gridworld
import gridworld
import helperFunctions
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import copy
import numpy as np

# Gridworld:
# *) Worker:    Blue
# *) Pitalls:   Red
# *) Exit:      Green
agent = gridworld.GameEnv()


agent.step(0)
print("I moved up")
agent.step(1)
print("I moved down")
agent.step(2)
print("I moved left")
agent.step(3)
print("I moved right")

########################################################################################################################
#                                Part 1: Play the game once to see an untrained agent at work                          #
########################################################################################################################
agent.reset()
world = agent.render_world()
plt.ioff()
plt.tight_layout()
plt.axis("off")
plt.imshow(world, interpolation="nearest")
plt.draw()
plt.title("Maze layout", fontsize=24)
plt.tight_layout()
plt.show()
plt.close()

agent.reset()
agent.close_world_display()
print("Let the game begin...")

# generate all states
all_states = []
for x in range(agent.sizeX):
    for y in range(agent.sizeY):
        all_states.append((x, y))

# Q is a dictionary that contains the rewards for all four actions that can be performed in any given square of Gridworld.
# Initialize Q and keep track of how many times Q[s] has been updated
Q = {}
update_counts_sa = {}
for s in all_states:
    update_counts_sa[s] = {}
    Q[s] = {}
    for a in agent.action_space:
        update_counts_sa[s][a] = 1.0
        Q[s][a] = 0.0

gamma = 0.9  # discount factor
alpha_W = 0.1  # learning rate
t = 1.0  # count time

########################################################################################################################

# To start the algorithm we need any action, so we pick one randomly until we find a valid action which we perform
found_initial_move = False
current_action = None
current_state = agent.get_state()
while not found_initial_move:
    current_action = helperFunctions.random_action(None, agent.action_space, eps=1)
    found_initial_move = agent.is_possible_action(current_action)


# loop until done (i.e. solved the maze or gave up)
done = False
while not done:
    # perform current step and get the next state, the reward/penalty for the move, and whether the agent is done (solved or gave up)
    next_state, reward, done = agent.step(current_action, False)

    # get the best currently known action for the state we are in now
    next_action = helperFunctions.get_best_action(Q[current_state])[0]
    # randomize action to allow for exploration. As time progresses, make random actions less likely.
    next_action = helperFunctions.random_action(next_action, agent.action_space, eps=0.4/t)

    # Update Q
    alpha = alpha_W/update_counts_sa[current_state][current_action]
    update_counts_sa[current_state][current_action] += 0.005
    Q[current_state][current_action] = Q[current_state][current_action] + alpha*(reward + gamma*Q[next_state][next_action] - Q[current_state][current_action])

    # update current state, current action, and start over
    current_state = next_state
    current_action = next_action
    t += 0.001

########################################################################################################################
#                          Part 2: Show the exploration route taken by the untrained worker                            #
########################################################################################################################

# show exploration route
result = ""
if not agent.gave_up and not agent.fell:
    result = "I solved gridworld in " + str(agent.steps) + " steps."
elif not agent.gave_up and agent.fell:
    result = "I fell into a pit after " + str(agent.steps) + " steps."
else:
    result = "Sorry, I had to give up after " + str(agent.max_steps) + " steps."

# Animate the steps of the first game
print("Watch my exploration route... (close the plot window to contine)")
helperFunctions.animate_steps(agent, "Gridworld exploration untrained worker", result)


########################################################################################################################
#                        Part 3: Play the game 10 000 times to learn the best solution strategy                        #
########################################################################################################################

print("Now let me train for a while, I enjoyed the game so much!")

agent.reset()
plt.close('all')

# The code is essentially identical to the one used above, but now carried out 10 000 times
training_episodes = 6001
snapshot_interval = 1000
t = 1
q_snapshot = []
for i in range(training_episodes):
    if i % 1000 == 0:
        print("I'm playing game " + str(i) + " / " + str(training_episodes))
    if i % snapshot_interval == 0:
        q_snapshot.append(copy.deepcopy(Q))
    if i % 1000 == 0:
        t += 0.01
    agent.reset()
    found_initial_move = False
    current_action = None
    current_state = agent.get_state()
    while not found_initial_move:
        current_action = helperFunctions.random_action(None, agent.action_space, eps=1)
        found_initial_move = agent.is_possible_action(current_action)
    done = False

    # loop until done (i.e. solved the maze or gave up)
    while not done:
        # perform current step and get the next state, the reward/penalty for the move, and whether the agent is done (solved or gave up)
        next_state, reward, done = agent.step(current_action, False)

        # get the best currently known action for the state we are in now
        next_action = helperFunctions.get_best_action(Q[current_state])[0]
        # randomize action to allow for exploration. As time progresses, make random actions less likely.
        next_action = helperFunctions.random_action(next_action, agent.action_space, eps=0.4/t)

        # Update Q
        # alpha = alpha_W / update_counts_sa[current_state][current_action]
        alpha = 0.2
        update_counts_sa[current_state][current_action] += 0.005
        Q[current_state][current_action] = Q[current_state][current_action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[current_state][current_action])

        # update current state, current action, and start over
        current_state = next_state
        current_action = next_action

    # update one last time
    Q[current_state][current_action] = Q[current_state][current_action] + alpha * (reward + gamma * Q[next_state][next_action] - Q[current_state][current_action])

print("Ok, I am done practicing.")
agent.reset()
plt.close('all')

########################################################################################################################
#                           Show snapshots of what the agent has learned                                               #
########################################################################################################################

episode = 0
obj_pos = []
for o in agent.objects:
    if o.name == 'worker':
        continue
    obj_pos.append(tuple((o.x, o.y)))

for Q in q_snapshot:
    # plot the world
    world = np.zeros([5+2, 5+2])
    world = np.full_like(world, -1.)

    arrow_dict = {}
    # find the best action and map the colors according to certainty
    for pos, vals in Q.items():  # iterate over grid positions
        directions, q = list(vals.keys()), list(vals.values())
        best, worst = max(q), min(q)
        shift = -worst

        # find best direction for arrow, skip for objects
        if pos not in obj_pos:
            mpos = 0
            for mpos in range(4):
                if q[mpos] == best:
                    break
            arrow_dict[(pos[0]+1, pos[1]+1)] = directions[mpos]

        # scale everything to be in the interval [0,1]
        if (q[0]+q[1]+q[2]+q[3]+4*shift) != 0:
            color = (best + shift)/(q[0]+q[1]+q[2]+q[3]+4*shift)
        else:
            color = 0.00001

        world[1+pos[1]][1+pos[0]] = color

    for o in agent.objects:
        if o.name == 'worker':
            continue
        world[1+o.y][1+o.x] = 2

    plt.ioff()
    plt.axis("off")
    masked_cmap = plt.get_cmap('rainbow')
    masked_cmap.set_under(color='black')
    masked_cmap.set_over(color='white')
    im = plt.imshow(world, interpolation="nearest", cmap=masked_cmap, vmin=0, vmax=1)
    im.set_array(world)
    colbar = plt.colorbar()
    colbar.ax.tick_params(labelsize=24)

    # add arrows: [0, 1, 2, 3] = up, down, left, right
    for pos, arr_dir in arrow_dict.items():
        if Q[(pos[0]-1, pos[1]-1)].values() == [0, 0, 0, 0]:  # field never visited
            continue
        if arr_dir == 0:
            xstart, ystart = pos[0], pos[1] + 0.25
            deltax, deltay = 0, -0.5
        if arr_dir == 1:
            xstart, ystart = pos[0], pos[1] - 0.25
            deltax, deltay = 0, 0.5
        if arr_dir == 2:
            xstart, ystart = pos[0] + 0.25, pos[1]
            deltax, deltay = -0.5, 0
        if arr_dir == 3:
            xstart, ystart = pos[0] - 0.25, pos[1]
            deltax, deltay = 0.5, 0

        plt.arrow(xstart, ystart, deltax, deltay, shape='full', width=0.1, length_includes_head=True, head_width=.35, color='black')

    plt.draw()
    plt.title("After " + str(episode) + " episodes", fontsize=24)
    plt.tight_layout()
    plt.savefig("./" + str(episode) + ".pdf", bbox_inches='tight')
    episode += snapshot_interval
    plt.close()


########################################################################################################################
#                           Part 4: Show the exploration route taken by the trained worker                             #
########################################################################################################################

# Navigate the maze using the best steps as learned by the agent
current_state = agent.get_state()
done = False
while not done:
    current_action = helperFunctions.get_best_action(Q[current_state])[0]
    current_state, reward, done = agent.step(current_action, False)

result = ""
if not agent.gave_up:
    result = "I can now solve Gridworld in " + str(agent.steps) + " steps."
else:
    result = "I haven't learned solving Gridworld in " + str(agent.max_steps) + " steps."

print(result)
# Animate the steps of the trained worker
print("Watch my exploration route... (close the plot window to contine)")
helperFunctions.animate_steps(agent, "Gridworld exploration trained worker", result)


print("Thanks for playing! Bye.")

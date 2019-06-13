import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# get the best action. If several actions are equally good, choose a random one
def get_best_action(diction):
    best_actions = []
    max_key = None
    max_val = float('-inf')
    for k, v in diction.items():
        if v > max_val:
            max_val = v
            max_key = k
            best_actions = [[max_key, max_val]]
        elif v == max_val:
            best_actions.append([k, v])

    return best_actions[np.random.randint(0, len(best_actions))]


# randomize the action in 100*eps percent of the cases
def random_action(action, action_space, eps=0.3):
    p = np.random.random()
    if p < (1 - eps):
        return action
    else:
        return np.random.choice(action_space)


# Animate the steps taken, ignore this
step_counter = 0
explore_step = None


def animate_steps(agent, window_title, fig_title=""):
    global step_counter
    plt.ioff()
    fig = plt.figure(window_title)
    fig.suptitle(fig_title)
    mySteps = agent.steps_taken
    agent.reset()
    step_counter = 0
    explore_step = mySteps[step_counter]
    im = plt.imshow(agent.render_world(), animated=True)

    def update_fig(*args):
        global explore_step, step_counter
        if step_counter < len(mySteps):
            explore_step = mySteps[step_counter]
        else:
            step_counter = 0
            explore_step = mySteps[step_counter]
            agent.reset()
        agent.step(explore_step, False)
        step_counter += 1
        im.set_array(agent.render_world())
        plt.draw()
        return im,

    plt.axis("off")
    plt.savefig("./" + window_title + "_step_0.pdf", dpi=300, bbox_inches='tight')
    for f in range(len(mySteps)):
        plt.axis("off")
        update_fig()
        # uncomment to save each step individualy as a .pdf
        # plt.savefig("./" + window_title + "_step_" + str(f+1) + ".pdf", dpi=300, bbox_inches='tight')

    ani = animation.FuncAnimation(fig, update_fig, interval=150, blit=True, frames=len(mySteps)-1, repeat=True)
    plt.show(window_title)

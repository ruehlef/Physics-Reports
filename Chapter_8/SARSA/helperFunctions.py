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
    my_steps = agent.steps_taken
    agent.reset()
    step_counter = 0
    im = plt.imshow(agent.render_world(), animated=True)

    def update_fig(*args):
        global explore_step, step_counter
        if step_counter == 0:
            agent.reset()
        elif step_counter <= len(my_steps):
            explore_step = my_steps[step_counter-1]
            agent.step(explore_step, False)
        else:
            step_counter = -1
            agent.reset()
        im.set_array(agent.render_world())
        plt.draw()
        step_counter += 1
        return im

    plt.axis("off")
    plt.savefig("./" + window_title + "_step_0.pdf", dpi=300, bbox_inches='tight')
    # uncomment to save each step individually as a .pdf
    # for f in range(len(my_steps)):
    #     plt.axis("off")
    #     update_fig()
    #     plt.savefig("./" + window_title + "_step_" + str(f+1) + ".pdf", dpi=300, bbox_inches='tight')
    ani = animation.FuncAnimation(fig, update_fig, interval=300, blit=False, frames=len(my_steps)+1, repeat=True)
    plt.show()
    agent.close_world_display()
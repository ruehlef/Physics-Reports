import tkinter as tk
import tkinter.messagebox

import numpy as np

# import trained network
import chainer
from chainerrl.agents.dqn import DQN
from chainerrl import explorers, replay_buffer
from chainer import optimizers
from chainerrl import q_functions

from gym import spaces

# for GUI
board = tk.Tk()
board.title("Tic Tac Toe")
board.resizable(width=False, height=False)
frame = tk.Frame(board)
frame.pack()

# state of the board
state = np.array([[0. for _ in range(3)] for _ in range(3)], dtype=np.float32)  # 0: empty, -1: player, 1: AI
agent_path = "./"

# load trained DQN agent
obs_size = 9
n_actions = 9
q_func = q_functions.FCStateQFunctionWithDiscreteAction(obs_size, n_actions, n_hidden_channels=100, n_hidden_layers=2)
# Use epsilon-greedy for exploration
start_epsilon = 0.1
end_epsilon = 0.01
final_exploration_steps = 10 ** 4
action_space = spaces.Discrete(n_actions)
explorer = explorers.LinearDecayEpsilonGreedy(start_epsilon, end_epsilon, final_exploration_steps, action_space.sample)
opt = optimizers.Adam()
opt.setup(q_func)

rbuf_capacity = 5 * 10 ** 5
rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)
agent = DQN(q_function=q_func, optimizer=opt, replay_buffer=rbuf, gpu=-1, gamma=0.99, explorer=explorer)

agent.load(agent_path)


def check_game(field, action):
    global player_symbol, CPU_symbol, state

    # action is linearized to 0 to 8: break down into 3 x 3 array
    row, col = int(int(action) / 3), int(action) % 3

    if field["text"] == " ":  # user performed legal move
        field["text"] = player_symbol
        field["state"] = "disabled"
        state[row][col] = -1

    # player won
    if field1["text"] == player_symbol and field2["text"] == player_symbol and field3["text"] == player_symbol or field4["text"] == player_symbol and field5["text"] == player_symbol and field6["text"] == player_symbol or field7["text"] == player_symbol and field8["text"] == player_symbol and field9["text"] == player_symbol or field1["text"] == player_symbol and field4["text"] == player_symbol and field7["text"] == player_symbol or field2["text"] == player_symbol and field5["text"] == player_symbol and field8["text"] == player_symbol or field3["text"] == player_symbol and field6["text"] == player_symbol and field9["text"] == player_symbol or field1["text"] == player_symbol and field5["text"] == player_symbol and field9["text"] == player_symbol or field3["text"] == player_symbol and field5["text"] == player_symbol and field7["text"] == player_symbol:
        tkinter.messagebox.showinfo("Winner", "You won the game.")
        board.quit()
        return False

    # draw
    elif field1["text"] != " " and field2["text"] != " " and field3["text"] != " " and field4["text"] != " " and field5["text"] != " " and field6["text"] != " " and field7["text"] != " " and field8["text"] != " " and field9["text"] != " ":
        tkinter.messagebox.showinfo("Draw", "The game ended in a draw.")
        board.quit()

    # AI moves next
    field = AI_move()
    field["text"] = CPU_symbol
    field["state"] = "disabled"

    # CPU won
    if field1["text"] == CPU_symbol and field2["text"] == CPU_symbol and field3["text"] == CPU_symbol or field4["text"] == CPU_symbol and field5["text"] == CPU_symbol and field6["text"] == CPU_symbol or field7["text"] == CPU_symbol and field8["text"] == CPU_symbol and field9["text"] == CPU_symbol or field1["text"] == CPU_symbol and field4["text"] == CPU_symbol and field7["text"] == CPU_symbol or field2["text"] == CPU_symbol and field5["text"] == CPU_symbol and field8["text"] == CPU_symbol or field3["text"] == CPU_symbol and field6["text"] == CPU_symbol and field9["text"] == CPU_symbol or field1["text"] == CPU_symbol and field5["text"] == CPU_symbol and field9["text"] == CPU_symbol or field3["text"] == CPU_symbol and field5["text"] == CPU_symbol and field7["text"] == CPU_symbol:
        tkinter.messagebox.showinfo("Loser", "The AI won the game.")
        board.quit()
        return False
    # draw
    elif field1["text"] != " " and field2["text"] != " " and field3["text"] != " " and field4["text"] != " " and field5["text"] != " " and field6["text"] != " " and field7["text"] != " " and field8["text"] != " " and field9["text"] != " ":
        tkinter.messagebox.showinfo("Draw", "The game ended in a draw.")
        board.quit()
        return False


def AI_move():
    global agent, state, fields
    pout_probs = agent.model(np.array([state])).q_values.data[0]

    pout_top_action_probs = sorted(pout_probs, reverse=True)
    corresponding_actions = []
    actions_for_output = ""
    # find actions corresponding to the best probabilities
    for ap in pout_top_action_probs:
        position = np.where(pout_probs == ap)[0][0]
        row, col = int(int(position) / 3), int(position) % 3
        corresponding_actions.append(np.where(pout_probs == ap)[0])
        actions_for_output += "(" + str(row + 1) + "," + str(col + 1) + "): " + str(np.round(ap, 3)) + ", "
    # print actions
    my_probs.delete("1.0", tk.END)
    my_probs.insert(tk.END, "Moves I considered: " + actions_for_output[0:-2])

    # iterate over all actions, take the best legal one
    for action in corresponding_actions:
        row, col = int(int(action) / 3), int(action) % 3

        if state[row][col] == 0:  # legal move
            state[row][col] = 1
            return fields[action[0]]

    return None  # should never happen, one move needs to be legal

# intialize 9 fields, font="Helvetica 32 bold"
# fields = tk.StringVar()
field1 = tk.Button(frame, text=" ", bd=4, height=4, width=8, highlightbackground ="white", font="Helvetica 32 bold", command=lambda: check_game(field1, 0), name="field1")
field2 = tk.Button(frame, text=" ", bd=4, height=4, width=8, highlightbackground ="white", font="Helvetica 32 bold", command=lambda: check_game(field2, 1), name="field2")
field3 = tk.Button(frame, text=" ", bd=4, height=4, width=8, highlightbackground ="white", font="Helvetica 32 bold", command=lambda: check_game(field3, 2), name="field3")
field4 = tk.Button(frame, text=" ", bd=4, height=4, width=8, highlightbackground ="white", font="Helvetica 32 bold", command=lambda: check_game(field4, 3), name="field4")
field5 = tk.Button(frame, text=" ", bd=4, height=4, width=8, highlightbackground ="white", font="Helvetica 32 bold", command=lambda: check_game(field5, 4), name="field5")
field6 = tk.Button(frame, text=" ", bd=4, height=4, width=8, highlightbackground ="white", font="Helvetica 32 bold", command=lambda: check_game(field6, 5), name="field6")
field7 = tk.Button(frame, text=" ", bd=4, height=4, width=8, highlightbackground ="white", font="Helvetica 32 bold", command=lambda: check_game(field7, 6), name="field7")
field8 = tk.Button(frame, text=" ", bd=4, height=4, width=8, highlightbackground ="white", font="Helvetica 32 bold", command=lambda: check_game(field8, 7), name="field8")
field9 = tk.Button(frame, text=" ", bd=4, height=4, width=8, highlightbackground ="white", font="Helvetica 32 bold", command=lambda: check_game(field9, 8), name="field9")

# place in grid
field1.grid(row=0, column=0)
field2.grid(row=0, column=1)
field3.grid(row=0, column=2)
field4.grid(row=1, column=0)
field5.grid(row=1, column=1)
field6.grid(row=1, column=2)
field7.grid(row=2, column=0)
field8.grid(row=2, column=1)
field9.grid(row=2, column=2)
my_probs = tk.Text(board, height=2, width=150)
my_probs.pack()
my_probs.insert(tk.END, "Moves I considered: -")
board.update()

fields = [field1, field2, field3, field4, field5, field6, field7, field8, field9]

# decide who's X and who's O; X starts
if np.random.random() >= 0.5:
    player_symbol = "X"
    CPU_symbol = "O"
    tkinter.messagebox.showinfo("New game", "You have the X symbol and start the game!")
    board.update()
else:
    player_symbol = "O"
    CPU_symbol = "X"
    tkinter.messagebox.showinfo("New game", "You have the O symbol and the AI starts the game!")
    board.update()
    field = AI_move()
    field["text"] = CPU_symbol
    field["state"] = "disabled"

# start game
board.mainloop()

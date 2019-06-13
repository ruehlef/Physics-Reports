import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import chainer


class TicTacToeEnv(gym.Env):

    def __init__(self):
        # internal state
        self.start_config = np.array([[0. for _ in range(3)] for _ in range(3)], dtype=np.float32)  # 0: empty, 1:agent, -1: opponent (epsilon-greedy agent)
        self.state = np.copy(self.start_config)
        self.global_t = 0

        # rewards and punishments
        self.illegal_move_punishment = -20
        self.won_reward = 20
        self.lost_punishment = -20
        self.draw_reward = 19
        self.move_punishment = 1

        # states and actions
        self.num_actions = 9  # one for each field
        self.action_space = spaces.Discrete(self.num_actions)  # linearize action space into a number 0 to 8
        self.observation_space = spaces.Box(-1, 1, (3, 3), dtype=np.float32)

        # agent NN that suggests state and action values
        self.agent = None

    def step(self, action):
        # action is linearized to 0 to 8: break down into 3 x 3 array
        row, col = int(int(action)/3), int(action) % 3

        # illegal move, field not empty
        if self.state[row][col] != 0:
            my_rew, am_i_done = self.illegal_move_punishment, True
            return np.array(self.state), my_rew, am_i_done, {}

        # legal move, carry it out
        self.state[row][col] = 1
        my_rew, am_i_done = self.reward()

        # opponent's turn if game is not won/draw
        if my_rew != self.draw_reward and my_rew != self.won_reward:
            my_rew, am_i_done = self.opponent_move()  # board gets updated in self.perform_opponent_action()

        return np.array(self.state), my_rew, am_i_done, {}

    def reset(self):
        # create random start state
        self.state = self.get_random_board()
        return np.array(self.state)

    def get_random_board(self, start_into_a_game=False):
        # by default, X always starts. Who is X and who is O is decided randomly. To train the agent, we thus need to start with an empty board or with a board with one move

        # Simple case: start with an empty board or with a board with one move
        if not start_into_a_game:
            start_config = np.array([[0. for _ in range(3)] for _ in range(3)], dtype=np.float32)  # 0: empty, 1:agent, -1: opponent (epsilon-greedy agent)
            if np.random.random() < 0.5:
                row, col = np.random.randint(0, 3), np.random.randint(0, 3)
                start_config[row][col] = -1

        # this generates a board that has already been played for up to num_moves, i.e. the agent has to start from this configuration rather than an empty board
        else:
            num_moves = np.random.randint(0, 6)  # carry out up to 5 moves
            start_config = np.array([[0. for _ in range(3)] for _ in range(3)], dtype=np.float32)  # 0: empty, 1:agent, -1: opponent (epsilon-greedy agent)

            for tries in range(5):  # try 5 times to get a valid board; if this does not work, start from empty board
                current_player = -1
                successful_board = True

                for _ in range(num_moves):
                    row, col = np.random.randint(0, 3), np.random.randint(0, 3)
                    while start_config[row][col] != 0:
                        row, col = np.random.randint(0, 3), np.random.randint(0, 3)
                    start_config[row][col] = current_player
                    if current_player == 1:
                        current_player = -1
                    else:
                        current_player = 1

                    # check if game is already over
                    if self.has_won(1) or self.has_won(-1):
                        tries += 1
                        start_config = np.array([[0. for _ in range(3)] for _ in range(3)], dtype=np.float32)  # 0: empty, 1:agent, -1: opponent (epsilon-greedy agent)
                        successful_board = False
                        break  # start new try

                if successful_board:
                    return start_config

        return start_config

    # we don't use the seeding, but it's an abstract class method, so we get a warning if we don't implement it
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets checked as an int elsewhere, so we need to keep it below 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def has_won(self, player, state=None):
        # we pass state since we also use this function to see whether a simulated move would win the game
        if state is None:
            state = self.state
        return (state[0][0] == player and state[0][1] == player and state[0][2] == player) or \
               (state[1][0] == player and state[1][1] == player and state[1][2] == player) or \
               (state[2][0] == player and state[2][1] == player and state[2][2] == player) or \
               (state[0][0] == player and state[1][0] == player and state[2][0] == player) or \
               (state[0][1] == player and state[1][1] == player and state[2][1] == player) or \
               (state[0][2] == player and state[1][2] == player and state[2][2] == player) or \
               (state[0][0] == player and state[1][1] == player and state[2][2] == player) or \
               (state[0][2] == player and state[1][1] == player and state[2][0] == player)

    def is_draw(self):
        return self.state[0][0] != 0 and self.state[0][1] != 0 and self.state[0][2] != 0 and \
               self.state[1][0] != 0 and self.state[1][1] != 0 and self.state[1][2] != 0 and \
               self.state[2][0] != 0 and self.state[2][1] != 0 and self.state[2][2] != 0

    def reward(self):
        # agent won
        if self.has_won(1):
            return self.won_reward, True
        # draw
        if self.is_draw():
            return self.draw_reward, True

        # nothing special
        return self.move_punishment, False

    def opponent_move(self):
        # perform epsilon-greedy actions
        self.perform_opponent_action()

        # opponent won
        if self.has_won(-1):
            return self.lost_punishment, True
        # draw
        if self.is_draw():
            return self.draw_reward, True

        # nothing special, game goes on
        return self.move_punishment, False

    def set_agent(self, agent):
        if self.agent is None:
            self.agent = agent  # agent used to play against the AI that is being trained (either the agent that is being trained itself or a pretrained agent)

    def set_global_t(self, global_t):
        self.global_t = global_t

    def perform_opponent_action(self):
        # look ahead to see whether there exists a move such that the game is won
        for r in range(3):
            for c in range(3):
                test_state = np.copy(self.state)
                if test_state[r][c] == 0:
                    test_state[r][c] = -1
                    if self.has_won(-1, test_state):
                        self.state = np.copy(test_state)
                        return

        # if not act epsilon-greedily. Act more greedily over time:
        if np.random.random() < max(0.15, 0.4/np.log10(10 + 0.01*self.global_t)):
            # perform random action: generate all actions, shuffle, take first legal one
            actions = np.random.permutation(self.num_actions)
            for action in actions:
                # action is linearized to 0 to 8: break down into 3 x 3 array
                row, col = int(int(action) / 3), int(action) % 3

                if self.state[row][col] == 0:  # legal move
                    self.state[row][col] = -1
                    return

        else:
            # query NN for best actions, try successively best ones, take first legal one

            with chainer.using_config('train', False), chainer.no_backprop_mode():
                pout_probs = self.agent.model(np.array([self.state])).q_values.data[0]
                pout_top_action_probs = sorted(pout_probs, reverse=True)  # sort best to worst

            # iterate over all actions, take the best legal one
            for ap in pout_top_action_probs:
                # get index (i.e. the action) of the current probability
                # note that the prob array was sorted, so we have to find the action that corresponds to this probability
                action = np.where(pout_probs == ap)[0][0]
                # action is linearized to 0 to 8: break down into 3 x 3 array
                row, col = int(int(action) / 3), int(action) % 3

                if self.state[row][col] == 0:  # legal move
                    self.state[row][col] = -1
                    return

        return

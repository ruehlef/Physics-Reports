from gym.envs.registration import register

###### Tic Tac Toe environment
register(
    id='TTT-DQN-v0',
    entry_point='gym_dqn.TTT_env:TicTacToeEnv',
    max_episode_steps=100
)

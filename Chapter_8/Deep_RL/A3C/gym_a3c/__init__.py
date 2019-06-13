from gym.envs.registration import register

###### Tic Tac Toe environment
register(
    id='TTT-A3C-v0',
    entry_point='gym_a3c.TTT_env:TicTacToeEnv',
    max_episode_steps=100
)
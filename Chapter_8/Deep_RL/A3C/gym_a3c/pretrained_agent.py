i# import trained network
import chainer
from chainerrl.agents import a3c
from chainerrl import links
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies

# define the NN
class pretrained_NN(chainer.ChainList, a3c.A3CModel):
    def __init__(self, ndim_obs, n_actions, hidden_sizes=(100, 100)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


# pretrained agent
def get_pretrained_agent(agent_path="./"):
    model = pretrained_NN(ndim_obs=9, n_actions=9)
    opt = rmsprop_async.RMSpropAsync()
    opt.setup(model)
    agent = a3c.A3C(model, opt, t_max=5, gamma=0.99, beta=1e-2)
    agent.load(agent_path)
    return agent

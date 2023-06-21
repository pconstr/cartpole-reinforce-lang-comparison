import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SoftMaxMultiArmedBanditPolicy(nn.Module):
    """
    Singular policy not conditional on inputs or state
    For plain Multi-armed bandits
    """
    def __init__(self, n_arms):
        super(SoftMaxMultiArmedBanditPolicy, self).__init__()
        self.pref = nn.Linear(1, n_arms, bias=False)
        self.device = None

    def forward(self, x):
        # ones expected as input in x
        x = self.pref(x)
        return F.softmax(x, dim=1)

    def act(self):
        # There really is no state but we fabricate a constant (singular) state
        state = torch.ones(1,1)#.to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def policy(self):
        return F.softmax(self.pref.weight, dim=0)

    def to(self, device):
        return super(SoftMaxMultiArmedBanditPolicy, self).to(device)

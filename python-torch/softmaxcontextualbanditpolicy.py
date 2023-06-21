import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

class SoftMaxContextualBanditPolicy(nn.Module):
    """                                                                                                                                                                                   
    Conditional on context but not modelling state                                                                                                                                    
    For contextual bandits                                                                                                                                                         
    """
    def __init__(self, n_arms, n_context, n_hidden):
        super(SoftMaxContextualBanditPolicy, self).__init__()
        
        self.l1 = nn.Linear(n_context, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_arms)
        self.device = None

    def forward(self, x):
        h = F.relu(self.l1(x))
        pref = self.l2(h)                
        return F.softmax(pref, dim=1)

    def act(self, context, return_prob=False, hard=False, device=None):
        # unsqueeze a batch of 1
        context = torch.from_numpy(context).float().unsqueeze(0)
        if device is not None:
            context = context.to(device)
        probs = self.forward(context).cpu()
        m = Categorical(probs)
        if hard:
            action = probs.argmax()
        else:
            action = m.sample()
        if return_prob:
            return action.item(), m.log_prob(action), probs
        else:
            return action.item(), m.log_prob(action)

    def to(self, device):
        return super(SoftMaxContextualBanditPolicy, self).to(device)


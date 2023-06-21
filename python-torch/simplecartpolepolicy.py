import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

def flatten_to_choose0(i, d0, device=None):
    n = i.shape[0]
    return d0 * torch.arange(n, dtype=i.dtype).to(device) + i

def choose_along_axis0(t, i, device=None):
    idx = flatten_to_choose0(i, t.shape[1], device=device)
    return t.take(idx)


class SimpleCartPolePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 2)
        self.device = None
        
    def forward(self, x):
        h = F.relu(self.l1(x))
        pref = self.l2(h)
        return F.log_softmax(pref, dim=1)
    
    def old_act(self, context, return_prob=False, device=None):
        # unsqueeze a batch of 1
        context = torch.from_numpy(context).float().unsqueeze(0)
        if device is not None:
            context = context.to(device)
        probs = self.forward(context).cpu()
        m = Categorical(probs)
        action = m.sample()
        if return_prob:
            return action.item(), m.log_prob(action), probs
        else:
            return action.item(), m.log_prob(action)

    def act(self, context, device=None):
        # unsqueeze a batch of 1
        context = torch.from_numpy(context).float().unsqueeze(0)
        if device is not None:
            context = context.to(device)
        probs = torch.exp(self.forward(context).cpu())
        m = Categorical(probs)
        action = m.sample()
        return action

    def recap(self, context, action, device=None):
        context = torch.from_numpy(context).float()
        if device is not None:
            context = context.to(device)
        probs = self.forward(context)#.cpu()
        pred = choose_along_axis0(probs, action, device=device)
        return pred

    def to(self, device):
        return super().to(device)

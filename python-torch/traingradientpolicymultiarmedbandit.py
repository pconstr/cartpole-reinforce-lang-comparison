import numpy
from collections import deque
import torch
import matplotlib.pyplot as plt

def evaluate_gradient_policy_multi_armed_bandit(policy, bandit, n_evaluation_episodes, hard=False):

    is_contextual = hasattr(bandit, 'observe')
    
    episode_rewards = []
    episode_regrets = []

    with torch.no_grad():
        for episode in range(n_evaluation_episodes):

            if is_contextual:
                context = bandit.observe()
                arm, log_prob = policy.act(context, hard=hard)
            else:
                arm, log_prob = policy.act()

            reward = bandit.pull(arm)
            regret = bandit._latest_regret

            episode_rewards.append(reward)
            episode_regrets.append(regret)

    return numpy.array(episode_rewards), numpy.array(episode_regrets)

def train_gradient_policy_multi_armed_bandit(policy, bandit, optimizer, n_training_episodes, batch_size=16, print_every=100):

    is_contextual = hasattr(bandit, 'observe')

    batched_returns = []
    batched_log_probs = []
  
    regrets = []

    eps = numpy.finfo(numpy.float32).eps.item()
    returns_deque = deque(maxlen=100)

    for i_episode in range(1, n_training_episodes + 1):

        if is_contextual:
            context = bandit.observe()
            arm, log_prob = policy.act(context)
        else:
            arm, log_prob = policy.act()

        reward = bandit.pull(arm)

        returns_deque.append(reward)
        regrets.append(bandit._regret())

        batched_log_probs.append(log_prob)
        batched_returns.append(reward)

        if len(batched_log_probs) == batch_size:

            returns = torch.tensor(batched_returns)

            if len(returns_deque) < 5:
                # subtract baseline.
                # the baseline is the mean over previous episodes
                rt = torch.tensor(returns_deque)
                b = rt.mean().item()
                returns = returns - b
            elif True:
                # standardise
                rt = torch.tensor(returns_deque)
                m = rt.mean().item()
                sd = rt.std().item()
                returns = (returns - m) / (sd + eps)
            else:
                pass

            log_probs = torch.cat(batched_log_probs)
            loss_terms = - log_probs * returns
            policy_loss = loss_terms.sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            batched_log_probs.clear()
            batched_returns.clear()


        if i_episode % print_every == 0:
            print("Episode {}\tAverage reward: {:.2f}".format(i_episode, numpy.mean(returns_deque)))

    plt.scatter(x=numpy.arange(len(regrets)), y=regrets)

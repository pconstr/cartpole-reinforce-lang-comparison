import numpy
from collections import deque
import torch
import matplotlib.pyplot as plt


def sum_discounted(rewards, gamma):
    n = rewards.shape[0]
    m = torch.pow(gamma, torch.arange(n))
    return torch.flip(torch.flip(m * rewards, dims=(0,)).cumsum(0), dims=(0,))


def reinforce(
        policy,
        env,
        gamma,
        optimizer,
        n_training_episodes,
        batch_size=16,
        device=None
        ):

    batched_returns = []
    batched_observations = []
    batched_actions = []
    #batched_log_probs = []
    batched_episodes = 0
    batch_counter = 0

    regrets = []

    returns_deque = deque(maxlen=100)
    durations_deque = deque(maxlen=100)
    eps = numpy.finfo(numpy.float32).eps.item()


    def update_network():
        returns = torch.cat(batched_returns).to(device)

        m = returns.mean().item()
        sd = returns.std().item()
        returns = (returns - m) / (sd + eps)

        observations = numpy.stack(batched_observations)
        actions = torch.cat(batched_actions).to(device)
        log_probs = policy.recap(observations, actions, device=device)
        loss_terms = - log_probs * returns
        policy_loss = loss_terms.sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    def report():
        mean_return = numpy.mean(returns_deque)
        mean_duration = numpy.mean(durations_deque)
        print (f'{episode_i} {mean_return} {mean_duration}')

    for episode_i in range(1, n_training_episodes + 1):

        episode_rewards = []

        observation, _ = env.reset()

        # experimence entire episode according to current policy
        while True:
            with torch.no_grad():
                action = policy.act(observation, device=device)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_rewards.append(reward)
                batched_observations.append(observation)
                batched_actions.append(action)

            if terminated or truncated:
                break

        episode_rewards = torch.tensor(episode_rewards)

        returns_deque.append(episode_rewards.sum())
        durations_deque.append(episode_rewards.shape[0])
        discounted = sum_discounted(episode_rewards, gamma)

        batched_returns.append(discounted)
        batched_episodes = batched_episodes + 1

        if batched_episodes == batch_size:
            update_network()

            batched_returns.clear()
            batched_actions.clear()
            #batched_log_probs.clear()
            batched_observations.clear()
            batched_episodes = 0            
            batch_counter = batch_counter + 1

            if batch_counter % 10 == 0:
                report()

    if batched_episodes:
        update_network()
        report()

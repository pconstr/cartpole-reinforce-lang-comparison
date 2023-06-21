import matplotlib.pyplot as plt
import numpy

def simulate(bandit, player, n_steps):
    regrets = []
    for i in range(n_steps):
        arm = player.act()
        reward = bandit.pull(arm)
        player.experience(arm , reward)
        regrets.append(bandit._regret())
    
    plt.scatter(x=numpy.arange(len(regrets)), y=regrets)

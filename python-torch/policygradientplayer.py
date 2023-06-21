from banditplayer import BanditPlayer
import numpy
from scipy.special import softmax

class PolicyGradientPlayer(BanditPlayer):
    def __init__(self, n_arms, alpha):
        super().__init__(n_arms)
        
        self.alpha = alpha
        self.theta = numpy.zeros(n_arms)
        self.total_reward = 0.0
        self.count = 0
    
    def experience(self, arm, reward):
        self.total_reward = self.total_reward + reward
        self.count = self.count + 1
        
        p = softmax(self.theta)
        n = p.shape[0]

        ai = numpy.zeros(n)
        ai[arm] = 1.0
        
        b = self.total_reward / self.count
        
        self.theta = self.theta + self.alpha * (reward - b) * (ai - p)
    
    def act(self):
        p = softmax(self.theta)
        n = p.shape[0]
        # draw from distribution
        arm = numpy.random.choice(n, p=p)
        return arm

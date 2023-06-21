from banditplayer import BanditPlayer
import numpy

class EpsilonGreedyPlayer(BanditPlayer):
    def __init__(self, n_arms, epsilon):
        super().__init__(n_arms)
        
        self.epsilon = epsilon
        self.all_arms = numpy.arange(n_arms)
        self.counts = numpy.zeros(n_arms, dtype='int64')
        self.totals = numpy.zeros(n_arms, dtype='float64')
        self.means = numpy.zeros(n_arms, dtype='float64')
        
    def experience(self, arm, reward):
        numpy.add.at(self.counts, [arm], 1)
        numpy.add.at(self.totals, [arm], reward)
        self.means[arm] = self.totals[arm] / self.counts[arm]
        
    def act(self):
        m = numpy.max(self.means)
        if isinstance(self.epsilon, float):
            e = self.epsilon
        else:
            i = self.counts.sum() + 1
            e = self.epsilon(i + 1)
        if numpy.random.uniform() < e:
            arms = self.all_arms
        else:
            arms = numpy.where(self.means == m)[0]
        n = arms.shape[0]
        assert n > 0
        if n == 1:
            return arms[0]
        else:
            i = numpy.random.choice(n)
            return arms[i]

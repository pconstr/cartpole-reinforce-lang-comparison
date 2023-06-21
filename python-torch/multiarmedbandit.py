import numpy

class MultiArmedBandit(object):
    def __init__(self, d_rewards):
        self._d_rewards = d_rewards
        self.n_arms = len(d_rewards)
        self._counts = numpy.zeros(self.n_arms, dtype='int64')

        self._means = [d.mean() for d in self._d_rewards]
        self._best_mean = max(self._means)
                
    def pull(self, arm):
        numpy.add.at(self._counts, [arm], 1)
        reward = self._d_rewards[arm].rvs()
        return reward

    def _regret(self):
        return numpy.dot(self._counts,
                         self._best_mean - self._means)

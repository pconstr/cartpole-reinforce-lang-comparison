from normalgamma import NormalGamma
import pandas
import seaborn
import numpy

from banditplayer import BanditPlayer
class ThompsonNormalPlayer(BanditPlayer):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        
        self.counts = numpy.zeros(n_arms, dtype='int64')
        self.reward_distributions = [NormalGamma(0,0,0,0) for i in range(n_arms)]
        
    def experience(self, arm, reward):
        numpy.add.at(self.counts, [arm], 1)
        x = numpy.array([reward])
        self.reward_distributions[arm] = self.reward_distributions[arm].updated(x)

    def act(self):
        unpulled = self.counts < 2
        unpulled = numpy.where(unpulled)[0]
        n_unpulled = unpulled.shape[0]
        if n_unpulled > 0:
            i = numpy.random.choice(n_unpulled)
            return unpulled[i]
        else:
            rewards = numpy.array([
                r.rvs(size=1) for r in self.reward_distributions
            ])
            best = numpy.argmax(rewards)
            return best

    def plot_reward_distributions(self):
        df = pandas.concat([
            pandas.DataFrame({
                'r': d.rvs(5000),
                'arm': i
            })
            for i, d in enumerate(self.reward_distributions)
        ])
        return seaborn.displot(df, x='r', hue='arm', kind='kde')

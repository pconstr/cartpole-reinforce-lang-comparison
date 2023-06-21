from banditplayer import BanditPlayer
import numpy

class BanditUCBPlayer(BanditPlayer):
    def __init__(self, n_arms, c=numpy.sqrt(2)):
        super().__init__(n_arms)
        self.c = c
        self.counts = numpy.zeros(n_arms, dtype='int64')
        self.totals = numpy.zeros(n_arms, dtype='float32')
        self.means = numpy.zeros(n_arms, dtype='float32')
        
    def act(self):
        # if any arm still hasn't been pulled pull from one of them at random
        # this does a bit of initial guaranteed exploration a avoids division by 0
        unpulled = self.counts == 0
        unpulled = numpy.where(unpulled)[0]
        n_unpulled = unpulled.shape[0]
        if n_unpulled > 0:
            i = numpy.random.choice(n_unpulled)
            return unpulled[i]
        else:
            t = self.counts.sum()
            u = self.c * numpy.sqrt(numpy.log(t) / self.counts)
            z = self.means + u
            return numpy.argmax(z)
        
    def experience(self, arm, reward):
        numpy.add.at(self.counts, [arm], 1)
        numpy.add.at(self.totals, [arm], reward)
        self.means[arm] = self.totals[arm] / self.counts[arm]

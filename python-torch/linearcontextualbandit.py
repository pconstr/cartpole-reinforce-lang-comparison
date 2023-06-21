import numpy

class LinearContextualBandit(object):
    def __init__(self,
                 d_context,
                 w_rewards,
                 b_rewards,
                 d_rewards_noise
                 ):
        self.d_context = d_context
        self.w_rewards = w_rewards
        self.b_rewards = b_rewards
        self.n_arms = w_rewards.shape[0]
        self.n_context = d_context.dim
        self.total_regret = 0
        self.d_rewards_noise = d_rewards_noise
        self.context = None
        self._latest_rewards = None
        self._latest_arm = None
        self._latest_regret = None
        self._total_regret = 0

    def observe(self):
        if self.context is None:
            self.context = self.d_context.rvs()
        return self.context
    
    def pull(self, arm):
        if self.context is None:
            raise RuntimeError("blind pull")
        rewards = self.w_rewards @ self.context + self.b_rewards
        observed_rewards = rewards + self.d_rewards_noise.rvs()
        self.context = None
        self._latest_rewards = rewards
        self._latest_arm = arm
        best = numpy.argmax(rewards)
        self._latest_regret = self._latest_rewards.max() - self._latest_rewards[self._latest_arm]
        self._total_regret += self._latest_regret

        return observed_rewards[arm]
            
    def _regret(self):
        "cummulative, compared to mean not observed (not taking noise)"
        return self._total_regret

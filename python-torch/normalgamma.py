from typing import NamedTuple
import numpy
from scipy import stats

class NormalGamma(object):
    """
    NormalGamma distribution, with bayesian updates
    from https://en.wikipedia.org/wiki/Normal-gamma_distribution
    """
    def __init__(self, p_mu, p_lambda, p_alpha, p_beta):
        self.p_mu = p_mu
        self.p_lambda = p_lambda
        self.p_alpha = p_alpha
        self.p_beta = p_beta
        
    def __repr__(self):
        return f"NormalGamma({self.p_mu}, {self.p_lambda}, {self.p_alpha}, {self.p_beta})"
    
    def rvs(self, size):
        tau = stats.gamma(a=self.p_alpha,
                          scale=1.0/self.p_beta).rvs(size=size)
        prec = self.p_lambda * tau
        var = 1 / prec
        std = numpy.sqrt(var)
        return stats.norm(loc=self.p_mu, scale=std).rvs(size=size)
        
    def updated(self, x):
        x_mean = x.mean()
        x_var = x.var()
        n = x.shape[0]
        lpn = self.p_lambda + n
        u_mu = (self.p_lambda * self.p_mu + n * x_mean) / lpn
        u_lambda = lpn
        u_alpha = self.p_alpha + n / 2.0
        u_beta = self.p_beta + 0.5 * (n * x_var + (self.p_lambda * n * (x_mean - self.p_mu)**2) / lpn)
        return NormalGamma(u_mu, u_lambda, u_alpha, u_beta)

import numpy as np
import pandas as pd


# Steps

# reward of picking a sample is modelled by a distribution, using likelihood fn Pr(r|mu)
# Posterior is given by Pr(r|mu).Pr(mu). This is a multivariate gaussian defined as:
# N(mu(t+1), v^2.B(t+1)^-1), where B(t) = I_d + sum_(t-1){b_c(tau).b_c(tau).T}
# mu = sum_(t-1){b_c(tau).b_c(tau).T} from R^d
# b_c = [dist_samples, var_samples, n_samples, prop_labeled, ratio_classes] will be a matrix of c x d
# y(t) = r_t.D(t), where D(x) = alpha.exp(-beta.x)
# y(t) = 2.cos-1((H_t-1.Ht)/||H_t-1||.||H_t||)/pi
#r_t = norm_hinge/D(t)

class thompson_sampling:
    __init__(self, estimator, X, y, epochs):

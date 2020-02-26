import numpy as np
import pandas as pd



class gridsearch:
    def __init__(self, est, params, cv):

        # define parameter search grid
        param_values = []
        for key in params.keys():
            param_values.append(params[key])

        self.params = np.array(np.meshgrid(a,b,c)).T.reshape(-1,3)

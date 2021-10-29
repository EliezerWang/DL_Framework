from sklearn.feature_selection import SelectFromModel
import numpy as np
from data_gen.var_lib import *


class Features_optimizer_split(object):

    def __init__(self, data, split_spot):
        self.data = data
        self.split_spot = split_spot
        self.train_data = self.data.loc[:self.split_spot].iloc[:, :-1]
        self.train_label = self.data.loc[:self.split_spot].iloc[:, -1]

    def select_features(self):
        sfm = SelectFromModel(model, max_features=aft_sel_num, threshold=-np.inf).fit(self.train_data,
                                                                                      self.train_label)
        self.data = self.data.iloc[:, np.append(sfm.get_support(), True)]

    def main(self):
        self.select_features()
        return self.data

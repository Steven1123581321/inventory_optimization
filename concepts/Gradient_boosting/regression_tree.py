import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import load_boston

class regression_tree():
    def __init__(self):
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load(self, data_set = 'boston'):
        if data_set is None:
            raise Exception ('data_set is not given')
        if data_set == 'boston':
            self.dataset = load_boston()

    def prepare_dataset(self):
        X = self.dataset['data']
        y = self.dataset['target']
        X, y = shuffle(X, y)
        data = np.column_stack([X,y])
        train, test = train_test_split(data, test_size=0.1)
        self.X_train = pd.DataFrame(train[:, :-1], columns=self.dataset['feature_names'])
        self.X_test = pd.DataFrame(test[:, :-1], columns=self.dataset['feature_names'])
        self.y_train = pd.Series(train[:, -1], name='House Price')
        self.y_test = pd.Series(test[:, -1], name='House Price')

    # def rss(self, y_left, y_right):
    #     def squared_residual_sum(y):
    #         return np.sum((y - np.mean(y)) ** 2)
    #     return squared_residual_sum(y_left) + squared_residual_sum(y_right)

    # def find_best_rule(self, input_data=None, output_data=None):
    #     best_feature, best_threshold, min_rss = None, None, np.inf
    #     for feature in input_data.columns:
    #         thresholds = input_data[feature].unique().tolist()
    #         thresholds.sort()
    #         thresholds = thresholds[1:]
    #         for t in thresholds:
    #             y_left_ix = input_data[feature] < t
    #             y_left, y_right = output_data[y_left_ix], output_data[~y_left_ix]
    #             t_rss = self.rss(y_left, y_right)
    #             if t_rss < min_rss:
    #                 min_rss = t_rss
    #                 best_threshold = t
    #                 best_feature = feature
    #     return {'feature': best_feature, 'threshold': best_threshold}

    # def split(self, input_data=None, output_data=None, depth=0, max_depth=5):
    #     rule = self.find_best_rule(input_data, output_data)
    #     left_ix = input_data[rule['feature']] < rule['threshold']
    #     input_data_left = input_data[left_ix]
    #     input_data_right = input_data[~left_ix]
    #     if len(input_data_left) < 2:
    #         input_data['label']
    #     return rule
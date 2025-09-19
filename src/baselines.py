import pandas as pd
import numpy as np 

class MajorityClassBaseline:
    def __init__(self):
        self.majority_probs = None 


    def fit(self, y_train):
        class_counts = pd.Series(y_train).value_counts()
        most_common_class = class_counts.index[0]

        self.majority_probs = np.zeros(3)
        self.majority_probs[most_common_class] = 1.0
        
    def predict_proba(self, X_test):
       n_samples = len(X_test)
       return np.tile(self.majority_probs, (n_samples, 1))

class UniformRandomBaseline:
    def predict_proba(self, X_test):
        n_samples = len(X_test)
        return np.full((n_samples, 3), 1/3)

class BookmakerBaseline:
    def predict_proba(self, odds_data):
        prob_cols = ['prob_home', 'prob_draw', 'prob_away'] 
        return odds_data[prob_cols].values


import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as MSE


class XGBobj:
    def __init__(self, max_depth=6, objective="reg:squarederror", gamma = 0, learning_rate = 0.01,
                    n_estimators = 1000, subsample = 0.5):
        self.param = {'max_depth': max_depth,
                    'objective': objective,
                    'gamma': gamma,
                    'learning_rate': learning_rate,
                    'n_estimators': n_estimators,
                    'subsample': subsample}

    def train(self, train_X, train_Y, test_X, test_Y, epochs):
        dtrain = xgb.DMatrix(train_X, label=train_Y)
        dtest = xgb.DMatrix(test_X, label=test_Y)

        self.bst = xgb.train(self.param, dtrain, epochs, [(dtrain, 'train'), (dtest, 'eval')], early_stopping_rounds = 100, verbose_eval = False)

    def valTest(self, val_X, val_Y):
        dtest = xgb.DMatrix(val_X, label=val_Y)
        pred = self.bst.predict(dtest)
        return MSE(val_Y, pred)

    def test(self, test_X, test_Y, test):
        dtest = xgb.DMatrix(test_X, label=test_Y)
        pred = self.bst.predict(dtest)
        return test[1](test_Y, pred)

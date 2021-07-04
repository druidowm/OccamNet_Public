import pickle
from pmlb import fetch_data, regression_dataset_names
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

data = []
i=0
for dataset in regression_dataset_names:
    if i<15:
        X, y = fetch_data(dataset, return_X_y=True)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = np.expand_dims(y, axis = 1)
        scaler = StandardScaler()
        y = scaler.fit_transform(y).squeeze()
        train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = .4, random_state = 42)
        val_X, test_X, val_Y, test_Y = train_test_split(test_X, test_Y, test_size = .5, random_state = 24)

        if (train_X.shape[0]<=1000):
            data.append((dataset,[train_X,train_Y,val_X,val_Y,test_X,test_Y]))
            i+=1

print(data)
file = open('pmlb.dat', 'wb')
pickle.dump(data, file)

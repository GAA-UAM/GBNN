
#  This is an example to plot the training time plot for different neural network models.

import time
import warnings
import numpy as np
import pandas as pd
import sklearn.datasets as dt
import matplotlib.pyplot as plt
from gbnn import GNEGNERegressor
from sklearn.preprocessing import LabelEncoder
from matplotlib.ticker import AutoMinorLocator
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

warnings.simplefilter("ignore")
random_state = 1

T = 200
reps = 100
step = 20


time_nn = np.zeros((T,))
time_deep = np.zeros((T,))
time_gbnn = []

X, y = dt.make_regression()

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state)


for i in range(reps):
    gbnn = GNEGNERegressor(total_nn=T,
                           num_nn_step=1,
                           subsample=1,
                           eta=0.025,
                           random_state=random_state)

    gbnn.fit(x_train, y_train)
    time_gbnn.append(gbnn._training_time)

    print('*', end='')

gbnn_ti = np.mean(time_gbnn, axis=0)

for t in range(1, T + 1, step):
    t0 = time.time()
    for i in range(reps):
        m = MLPRegressor(hidden_layer_sizes=(t,),
                         solver="adam",
                         #    tol=0.0, n_iter_no_change=200,
                         random_state=random_state)
        m.fit(X, y)
    time_nn[t-1] = (time.time()-t0)/reps
    
    t0 = time.time()
    for i in range(reps):
        n = MLPRegressor(hidden_layer_sizes=(t, t, t),
                         solver="adam",
                         #    tol=0.0, n_iter_no_change=200,
                         random_state=random_state)
        n.fit(X, y)
    time_deep[t-1] = (time.time()-t0)/reps

    print('*', end='')


fig, ax1 = plt.subplots()
ax1.set_xlabel("Number of neurons")
ax1.set_ylabel("Time (in seconds)")
ax1.plot(range(1, T + 1, 64),
         time_nn[0::step], color='tab:orange', label="NN")
ax1.plot(g, gbnn_ti, color='tab:blue', label="GBNN")
ax1.plot(range(1, T + 1, 64),
         time_deep[0::step], color='tab:green', label="Deep-NN")

ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.legend()
ax1.grid(True, linewidth=1, color='gainsboro', linestyle='-')
plt.title("Average training time")
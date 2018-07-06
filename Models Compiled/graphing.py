import matplotlib.pyplot as plt
import random
from matplotlib import style
import Regression_using_Machine_learning
import numpy as np
import csv
import lstm
import time
import tensorflow as tf
from sklearn.svm import SVR
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk


# here i was setting up my graphs that i want to plot on
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

# setting arrays for dates and prices
dates = []
prices = []


# used to read the csv file containing the data
def read_stock_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split(' ')[0]))
            prices.append(float(row[1]))
        return


# method used to create the SVM regression prediction
def predict_future(dates, prices, x):
    # using numpy to format our list to n * 1 matrix
    dates = np.reshape(dates, (len(dates), 1))

    # defining the svr model paramenters
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    # training the models with our dates and prices data
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    # plotting the graph using matplotlib
    ax1.scatter(dates, prices, color='black', label='Data')
    ax1.plot(dates, svr_rbf.predict(dates), color='navy', label='RBF model')
    ax1.plot(dates, svr_lin.predict(dates), color='c', label='Linear model')
    ax1.plot(dates, svr_poly.predict(dates), color='cornflowerblue', label='Polynomial model')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.set_title('SVM and Neural Networks For Bitcoin Stocks')
    ax1.legend()


# used to plot lstm prediction results on the second graph
def plot_results(predicted_data, true_data):
    ax2.plot(true_data, label='True Data')
    ax2.plot(predicted_data, label='Prediction')
    ax2.set_xtitle('Prices')
    ax2.set_ytitle('Dates')
    ax2.legend()


# used to plot the predicted results against real data for lsm graph2
def plot_results_multiple(predicted_data, true_data, prediction_len):
    # please not ever other color except blue is a prediction
    ax2.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        ax2.plot(padding + data)
        ax2.legend()


def tensor_neural():
    # Import data
    data = pd.read_csv('data_stocks.csv')

    # Drop date variable
    data = data.drop(['DATE'], 1)

    # Dimensions of dataset
    n = data.shape[0]
    p = data.shape[1]

    # Make data a np.array
    data = data.values

    # Training and test data
    train_start = 0
    train_end = int(np.floor(0.8 * n))
    test_start = train_end + 1
    test_end = n
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]

    # Scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)

    # Build X and y
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]

    # Number of stocks in training data
    n_stocks = X_train.shape[1]

    # Neurons
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128

    # Session
    net = tf.InteractiveSession()

    # Placeholder
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])

    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Hidden weights
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

    # Output weights
    W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
    bias_out = tf.Variable(bias_initializer([1]))

    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    # Output layer (transpose!)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))

    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # Init
    net.run(tf.global_variables_initializer())

    # Setup plot
    line1, = ax3.plot(y_test, color="purple")
    line2, = ax3.plot(y_test * 0.9, color="black")

    # Fit neural net
    batch_size = 256
    mse_train = []
    mse_test = []

    # Run
    epochs = 10
    for e in range(epochs):
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]


        # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

    mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
    print(mse_final)


# main section used to call methods we need to use
read_stock_data('BTC formatted.csv')
predict_future(dates, prices, 26)
tensor_neural()

# settings for the lsm network by playing with
# epoch and sequence length and batch size we can make a good model
global_start_time = time.time()
epochs = 10
seq_len = 50
print('> Loading data... ')
X_train, y_train, X_test, y_test = lstm.load_data('sp500.csv', seq_len, True)
print('> Data Loaded. Compiling...')
model = lstm.build_model([1, 50, 100, 1])
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=epochs,
    validation_split=0.05)
predictions = lstm.predict_sequences_multiple(model, X_test, seq_len, 50)
print('Training duration (s) : ', time.time() - global_start_time)
plot_results_multiple(predictions, y_test, 50)
plt.show()



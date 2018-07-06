import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.svm import SVR

#arrays for storing details
dates = []
prices = []

#function to read the data from a csv file and format it before saving to the arrays
def read_stock_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split(' ')[0]))
            prices.append(float(row[1]))
        return

#function to predict trends using SVR models (linear, polynomial and RBF)
def predict_future(dates,prices,x):
    #using numpy to format our list to n * 1 matrix
    dates = np.reshape(dates,(len(dates),1))

    #defining the svr model paramenters
    svr_lin = SVR(kernel='linear',C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

    #training the models with our dates and prices data
    svr_lin.fit(dates,prices)
    svr_poly.fit(dates,prices)
    svr_rbf.fit(dates,prices)

    #plotting the graph using matplotlib
    plt.scatter(dates,prices,color='black' , label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='navy', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='c', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='cornflowerblue' , label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression For Bitcoin Stocks')
    plt.legend()
    plt.show()

    #We then return each model predicted figures
    return svr_rbf.predict(x)(0), svr_lin.predict(x)[0], svr_poly.predict(x)[0]

#calling the read stock data function with the file as parameter
#read_stock_data('BTC formatted.csv')
#calling predict function to predict a trend
#predicted_price = predict_future(dates,prices,26)



# about the SVM model

#kernel
#Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’,
#‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a
# callable is given it is used to precompute the kernel matrix

#Degree
#Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.

#C
#Penalty parameter C of the error term. We set it to 1e3 which is 1000 shorthand style

#Gamma
#Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.





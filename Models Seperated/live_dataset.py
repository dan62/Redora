from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import sys
import requests, json
from time import sleep
import time;

#arrays to store data
price_array=[]
time_stamp=[]

#x for looping purposes
x=1

#function using alpha_vantage to pull stock prices in real time
def stockchart(symbol):
    plt.gcf().clear()
    ts = TimeSeries(key='S6CZE51TY1Y43KOE', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol,interval='1min', outputsize='full')

   #displaying the data both on the console and on a matplot graph
    print(data)
    data['close'].plot()
    plt.title('Stock chart')
    getBitcoinPrice()
    plt.show()

#functions gettingcurrent  bitcoin price in real time from bitstamp
def getBitcoinPrice():
    URL = 'https://www.bitstamp.net/api/ticker/'

    #try catch block using json to pull latest bitcoin price
    try:
        r = requests.get(URL)
        priceFloat = float(json.loads(r.text)['last'])
        return priceFloat
    except requests.ConnectionError:
        print ("Error querying Bitstamp API")

#while it gets live bitcoin price it assigns it to local computer time and stores both into arrays
while True:
    price = (str(getBitcoinPrice()))

    #printing the live bitcoin price in the console
    print (price)

    #incrementing x as index in arrays
    x=x+1
    localtime_Thailand = time.asctime(time.localtime(time.time()))
    price_array.insert(x,[price])
    time_stamp.insert(x, localtime_Thailand)
    print(time_stamp+price_array)

    #delay between getting the data to avoid overloads
    sleep(5)
    symbol = 'BTC'

    #calling stockchart function with the symbol identified so that bitstamp knows what data we want
    stockchart(symbol)




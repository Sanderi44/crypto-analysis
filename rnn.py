import sys
from ccxt_api import ccxt_api
from featureCalculator import FeatureCalculator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

def get_candles(exchange, base, market, interval='1m'):
	api = ccxt_api(load_markets=True, exchanges=[exchange])	
	candle_sets = api.get_candles(base, market, interval)
	return candle_sets[0]


class Parser:
	def __init__(self):
		self.exchanges = []
		self.bases = []
		self.markets = []
		self.intervals = []
		self.ohlcvs = []
		self.percent = 0.67
	
	def parse_key(self, key, value):
		if key == "exchange":
			self.exchanges.append(value)
		if key == "base":
			self.bases.append(value)
		if key == "market":
			self.markets.append(value)
		if key == "interval":
			self.intervals.append(value)
		if key == "ohlcv":
			self.ohlcvs.append(value)
		if key == "percent":
			self.percent = float(value)

		# print key + " is " + value
	
	def print_info(self):
		print "Exchanges: " + str(self.exchanges)
		print "Bases: " + str(self.bases)
		print "Markets: " + str(self.markets)
		print "Intervals: " + str(self.intervals)
		print "Ohlcvs: " + str(self.ohlcvs)
		print "Percent: " + str(self.percent)

	def get_query(self, query_num):
		if query_num < len(self.exchanges) and query_num < len(self.bases) and query_num < len(self.markets) and query_num < len(self.intervals) and query_num < len(self.ohlcvs):
			return self.exchanges[query_num], self.bases[query_num], self.markets[query_num], self.intervals[query_num], self.ohlcvs[query_num], self.percent
		else:
			self.print_info()
			raise Exception("Please specify an exchange, a base, a market and an inverval for each plot")

	def get_number_of_plots(self):
		return len(self.markets)

def ohlcvToNum(ohlcv):
	ohlcv_num = 1
	if ohlcv == "open":
		ohlcv_num = 1
	if ohlcv == "high":
		ohlcv_num = 2
	if ohlcv == "low":
		ohlcv_num = 3
	if ohlcv == "close":
		ohlcv_num = 4
	if ohlcv == "volume":
		ohlcv_num = 5
	return ohlcv_num

def plot(candles, exchange, base, market, interval, ohlcv="open"):
	ohlcv_num = ohlcvToNum(ohlcv)
	plt.figure()
	ts = [datetime.datetime.fromtimestamp(t/1000) for t in candles[:,0]]
	dates = date2num(ts)
	plt.plot_date(dates, candles[:,ohlcv_num], '-')
	plt.gcf().autofmt_xdate()
	plt.title("{0}/{1} market {2} from {3} for interval {4}".format(base, market, ohlcv, exchange, interval))
	plt.xlabel("Date")
	plt.ylabel("Value ({0})".format(base))


def feature_extract(candles, lookback=1):
	dataX, dataY = [], []
	for i in range(len(candles)-lookback-1):
		a = candles[i:(i+lookback), 0]
		dataX.append(a)
		dataY.append(candles[i + lookback, 0])
	return np.array(dataX), np.array(dataY)






def main(argv):
	i = 1
	parser = Parser()
	while i < len(argv):
		eq = argv[i].find("=")
		if eq > -1:
			key = argv[i][0:eq]
			value = argv[i][eq+1:]
			parser.parse_key(key, value)

		i += 1
	parser.print_info()

	lookback=6

	for i in range(parser.get_number_of_plots()):
		exchange, base, market, interval, ohlcv, percent = parser.get_query(i)
		candles = np.array(get_candles(exchange, base, market, interval))

		ohlcv_num = ohlcvToNum(ohlcv)
		data_untransformed = candles[:, ohlcv_num].reshape(-1, 1)
		# normalize the dataset
		scaler = MinMaxScaler(feature_range=(0.0, 0.9))
		dataset = scaler.fit_transform(data_untransformed)

		train_size = int(len(dataset) * percent)
		test_size = len(dataset) - train_size
		train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]




		train_x, train_y = feature_extract(train, lookback=lookback)
		test_x, test_y = feature_extract(test, lookback=lookback)
		

		# reshape input to be [samples, time steps, features]
		trainX = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
		testX = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))


		# create and fit the LSTM network
		model = Sequential()
		# LSTM neurons = 4, input_shape = (timesteps, features)
		model.add(LSTM(4, input_shape=(lookback, 1)))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error', optimizer='adam')
		model.fit(trainX, train_y, epochs=100, batch_size=1, verbose=2)



		# make predictions
		trainPredict = model.predict(trainX)
		testPredict = model.predict(testX)
		# invert predictions
		trainPredict = scaler.inverse_transform(trainPredict)
		trainY = scaler.inverse_transform([train_y])
		testPredict = scaler.inverse_transform(testPredict)
		testY = scaler.inverse_transform([test_y])
		# calculate root mean squared error
		trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
		print('Train Score: %.2f RMSE' % (trainScore))
		testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
		print('Test Score: %.2f RMSE' % (testScore))
		# shift train predictions for plotting
		# trainPredictPlot = np.empty_like(dataset)
		# trainPredictPlot[:, :] = np.nan
		# trainPredictPlot[lookback:len(trainPredict)+lookback, :] = trainPredict
		# shift test predictions for plotting
		# testPredictPlot = np.empty_like(dataset)
		# testPredictPlot[:, :] = np.nan
		# testPredictPlot[len(trainPredict)+(lookback*2)+1:len(dataset)-1, :] = testPredict
		
		# print trainPredict, testPredict


		# plot baseline and predictions
		plt.figure()
		ts = [datetime.datetime.fromtimestamp(t/1000) for t in candles[:,0]]
		dates = date2num(ts)
		plt.plot_date(dates, data_untransformed, '-', label="All data")
		plt.plot_date(dates[lookback:train_size-1], trainPredict, '-', label="Trained Prediction")
		plt.plot_date(dates[train_size + lookback:len(dataset)-1], testPredict, '-', label="Untrained Prediction")
		plt.gcf().autofmt_xdate()
		plt.title("{0}/{1} market {2} from {3} for interval {4}".format(base, market, ohlcv, exchange, interval))
		plt.xlabel("Date")
		plt.ylabel("Value ({0})".format(base))
		plt.legend()











		# plot(candles, exchange, base, market, interval, ohlcv)




		# plt.figure()
		# plt.plot_date(dates, np.log(candles[:,4]), '-')
		# plt.gcf().autofmt_xdate()

	plt.show()

if __name__ == "__main__":
	main(sys.argv)

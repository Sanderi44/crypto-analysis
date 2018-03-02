import sys
from ccxt_api import ccxt_api
from featureCalculator import FeatureCalculator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, concatenate, add, average
from keras.callbacks import TensorBoard
from keras.initializers import RandomUniform
from keras import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import math
import csv
import random, string


def load_google_data(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		data = list(reader)
	return np.array(data)

def get_candles(exchange, base, market, interval='1m'):
	api = ccxt_api(load_markets=True, exchanges=[exchange])	
	candle_sets = api.get_candles(base, market, interval)
	return candle_sets[0]

def load_file(filename):
	with open(filename, "r") as f:
		reader = csv.reader(f, delimiter=',')
		data = list(reader)
	return np.array(data).astype(float)			
	

def filter_data_by_start_and_end_date(data, start_date, end_date):
	new_data = []
	# print start_date, end_date
	for row in data:
		# print row
		numeric_date = row[0]
		# print numeric_date, start_date, end_date
		if numeric_date >= start_date and numeric_date <= end_date:
			new_data.append(row)

	return np.array(new_data)

def linear_interpolation_days(data):
	new_data = []
	for i in range(1, len(data)):
		spread = data[i,0] - data[i-1,0]
		new_data.append(data[i-1,:])
		if spread > 1:
			# print i, spread, data[i-1,0], data[i,0]
			for k in range(1, int(spread)):
				new_row = []
				new_date = data[i-1,0] + float(k)
				new_row.append(new_date)
				# print new_date
				for j in range(1, len(data[i,:])):
					new_val = ((data[i-1,j] * (data[i,0] - new_date)) + (data[i,j] * (new_date - data[i-1,0])))/(data[i,0] - data[i-1,0])
					new_row.append(new_val)
			new_data.append(new_row)
	return np.array(new_data)


def remove_repeating_days(data):
	new_data = []
	for i in range(1, len(data)):
		if data[i,0] != data[i-1,0]:
			new_data.append(data[i-1,:])

		
	return np.array(new_data)

def create_batches(data, time, separation, label_length):
	batches = []
	times = []
	for i in range(len(data)):
		count = 0
		batch_set = []
		for j in range(data[i].shape[0]):
			if count == separation + label_length:
				count = 0
			if count == separation:
				batch_set.append(data[i][j,:,:])
			count += 1


		batches.append(np.array(batch_set))

	count = 0
	for i in range(len(time)):
		if count == separation + label_length:
			count = 0
			times.append(time[i])

		# if count == separation:

		count += 1
	
	# print len(batches[0]), len(times)
	return batches, times


class Parser:
	def __init__(self):
		self.filename = ""
		self.percent = 0.9
		self.epochs = 20
		self.batch_size = 1
		self.name = ""
	
	def parse_key(self, key, value):
		if key == "filename":
			self.filename = value
		if key == "percent":
			self.percent = float(value)
		if key == "epochs":
			self.epochs = int(value)
		if key == "batch":
			self.batch_size = int(value)
		if key == "name":
			self.name = value

		# print key + " is " + value
	
	def print_info(self):
		print "Training Filename: " + self.filename
		print "Training Name: " + self.name
		print "Training Percentage: " + str(self.percent)
		print "Training Epochs: " + str(self.epochs)
		print "Training Batch Size: " + str(self.batch_size)

	def get_query(self):
		if self.filename != "":
			return self.filename, self.percent, self.epochs, self.batch_size, self.name
		else:
			self.print_info()
			raise Exception("Please specify an exchange, a base, a market and an inverval for each plot")
	


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
	fig = plt.figure()
	# ts = [datetime.datetime.fromtimestamp(t/1000) for t in candles[:,0]]
	# dates = date2num(ts)
	dates = candles[:,0]
	plt.plot_date(dates, candles[:,ohlcv_num], '-')
	plt.gcf().autofmt_xdate()
	plt.title("{0}/{1} market {2} from {3} for interval {4}".format(base, market, ohlcv, exchange, interval))
	plt.xlabel("Date")
	plt.ylabel("Value ({0})".format(base))
	return fig

def plot_all(candles, exchange, base, market, interval):
	fig = plt.figure()
	# ts = [datetime.datetime.fromtimestamp(t/1000) for t in candles[:,0]]
	dates = candles[:,0]
	plt.plot_date(dates, candles[:,1], '-', label="open")
	plt.plot_date(dates, candles[:,2], '-', label="high")
	plt.plot_date(dates, candles[:,3], '-', label="low")
	plt.plot_date(dates, candles[:,4], '-', label="close")
	plt.gcf().autofmt_xdate()
	plt.title("{0}/{1} market from {2} for interval {3}".format(base, market, exchange, interval))
	plt.xlabel("Date")
	plt.ylabel("Value ({0})".format(base))
	plt.legend()
	return fig



def feature_extract(data, dataLength, labelLength):
	dataX, dataY = [], []
	for i in range(dataLength, len(data)-labelLength-1):
		a = data[i-dataLength:i]
		b = data[i+1:(i+1+labelLength)]
		dataX.append(np.array([a]).T)
		dataY.append(np.array(b).T)
	return np.array(dataX), np.array(dataY)


def feature_extract_slope(data, dataLength):
	dataX, dataY = [], []
	for i in range(len(data)-dataLength-2):
		a = data[i:(i+dataLength)]
		b = data[i+1] - data[i]
		dataX.append(np.array([a]).T)
		dataY.append(np.array(b).T)
	return np.array(dataX), np.array(dataY)


def buildModel(dataLength, labelLength, neurons=20, optimizer='adam'):
	# _open = Input(shape=(dataLength, 1), name="Open")
	_high = Input(shape=(dataLength, 1), name="High")
	_low = Input(shape=(dataLength, 1), name="Low")
	_close = Input(shape=(dataLength, 1), name="Close")
	_volume = Input(shape=(dataLength, 1), name="Volume")
	_google = Input(shape=(dataLength, 1), name="Google")
	# _bias = Input(tensor=tf.constant([1.]))

	# openLayer = LSTM(10, return_sequences=False, recurrent_dropout=0.1)(_open)
	highLayer = LSTM(neurons, return_sequences=False)(_high)
	lowLayer = LSTM(neurons, return_sequences=False)(_low)
	closeLayer = LSTM(neurons, return_sequences=False)(_close)
	volumeLayer = LSTM(neurons, return_sequences=False)(_volume)
	googleLayer = LSTM(neurons, return_sequences=False)(_google)

	output = concatenate(
		[
			# openLayer,
			highLayer,
			lowLayer,
			closeLayer,
			volumeLayer,
			googleLayer,
			# _bias,
		]
	)

	# output = Dense(4, activation="linear", name="combiner_layer")(output)


	# print output

	output = Dense(labelLength, activation="linear", name="weighted_average_ouput", bias_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=None))(output)

	model = Model(inputs=[_high, _low, _close, _volume, _google], outputs=[output])
	model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
	return model

def main(argv):
	# Take in data from terminal arguments
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


	## Old Method
	# lookback=6
	# create and fit the LSTM network
	# model = Sequential()
	# # LSTM neurons = 4, input_shape = (timesteps, features)
	# batch_size = 1
	# model.add(LSTM(4, input_shape=(lookback, 1)))
	# model.add(Dense(1))
	# model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
	# model.reset_states()


	# This determines the length of the input data and output data
	# TODO: Need to fix the feature_extract function so that it works with multi-output 
	dataLength = 4
	labelLength = 1


	# Loads daily google search data for BTC that was downloaded
	google_data = load_google_data("BTC_google_data/BTC_search_01_2016-02_2018.csv")
	for i in range(len(google_data)):
		google_data[i,0] = date2num(datetime.datetime.fromtimestamp(float(google_data[i,0])))
		google_data[i,1] = float(google_data[i,1])
	google_data = google_data.astype(float)


	# Loads file with OHLCV data	
	filename, percent, epochs, batch_size, name = parser.get_query()
	datas_untransformed = load_file(filename)
	for i in range(len(datas_untransformed[:,0])):
		datas_untransformed[i,0] = float(math.floor(date2num(datetime.datetime.fromtimestamp(datas_untransformed[i, 0]/1000))))



	# print start_date, end_date
	datas_untransformed = linear_interpolation_days(datas_untransformed)
	datas_untransformed = linear_interpolation_days(datas_untransformed)

	# Filter the data between start and stop dates
	start_date = max(google_data[0,0], datas_untransformed[0, 0])
	end_date = min(google_data[-1,0], datas_untransformed[-1, 0])


	print google_data.shape, datas_untransformed.shape
	google_data = filter_data_by_start_and_end_date(google_data, start_date, end_date)	

	datas_untransformed = filter_data_by_start_and_end_date(datas_untransformed, start_date, end_date)	
	print google_data.shape, datas_untransformed.shape


	# Plots the price and volumes
	fig1 = plot_all(datas_untransformed, "bittrex", "USD", "BTC", "1d")
	plot(datas_untransformed, "bittrex", "USD", "BTC", "1d", "volume")
	
	# additionally plots the Google Search Data scaled to match the scale of the price data
	max_price = np.max(datas_untransformed[:,4])
	ax = fig1.gca()
	ax.plot_date(google_data[:,0], max_price*google_data[:,1].astype(float), "-", label="google data")
	ylbl = ax.yaxis.get_label().get_text()
	ylbl = ylbl + "\nScaled Google Search For Bitcoin"
	ax.set_ylabel(ylbl)


	# Scales the data for input into the Deep Learning
	scalar = 25000
	ohlc = datas_untransformed[:, 1:-1]/scalar
	v = datas_untransformed[:, -1]
	volume_scalar = np.max(v)
	v /= volume_scalar


	# Creates Features and labels
	times = datas_untransformed[:, 0]
	open_input, open_output = feature_extract(ohlc[:, 0], dataLength, labelLength)
	high_input, high_output = feature_extract(ohlc[:, 1], dataLength, labelLength)
	low_input, low_output = feature_extract(ohlc[:, 2], dataLength, labelLength)
	close_input, close_output = feature_extract(ohlc[:, 3], dataLength, labelLength)
	volume_input, volume_output = feature_extract(v, dataLength, labelLength)
	google_input, google_output = feature_extract(google_data[:,1], dataLength, labelLength)
	

	# Split the data into training and testing portions
	train_length = int(percent*ohlc.shape[0])

	open_train_x = open_input[:train_length, :]
	high_train_x = high_input[:train_length, :]
	low_train_x = low_input[:train_length, :]
	close_train_x = close_input[:train_length, :]
	volume_train_x = volume_input[:train_length, :]
	google_train_x = google_input[:train_length, :]
	
	close_train_y = close_output[:train_length, :]

	open_test_x = open_input[train_length:, :]
	high_test_x = high_input[train_length:, :]
	low_test_x = low_input[train_length:, :]
	close_test_x = close_input[train_length:, :]
	volume_test_x = volume_input[train_length:, :]
	google_test_x = google_input[train_length:, :]

	close_test_y = close_output[train_length:, :]



	# Build the deep learning model
	# TODO: add dynamic number of lstms and layer widths and which features
	model = buildModel(dataLength, labelLength, neurons=20, optimizer='adam')

	# Fits the new model to the data with 20% validation
	history = model.fit([high_train_x, low_train_x, close_train_x, volume_train_x, google_train_x], [close_train_y], epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[TensorBoard("/home/sander/tensorboard/"+name+"_"+filename.split(".")[0]+"_epochs"+str(epochs)+"_"+"batch"+str(batch_size))])

	# Predict on both the training and testing data
	# TODO: do batches to show multi-output results 
	training_set = [high_train_x, low_train_x, close_train_x, volume_train_x, google_train_x]
	testing_set = [high_test_x, low_test_x, close_test_x, volume_test_x, google_test_x]

	# ts = [datetime.datetime.fromtimestamp(t/1000) for t in datas_untransformed[:, 0]]
	# dates = date2num(ts)
	dates = datas_untransformed[:, 0]


	# if labelLength == 1:
	# 	trainPredict = model.predict(training_set, batch_size=batch_size)
	# 	testPredict = model.predict(testing_set, batch_size=batch_size)
	# 	# calculate root mean squared error
	# 	trainScore = math.sqrt(mean_squared_error(close_train_y[:,0]*scalar, trainPredict*scalar))/len(trainPredict)
	# 	print('Train Score: %.4f RMSE' % (trainScore))
	# 	testScore = math.sqrt(mean_squared_error(close_test_y[:,0]*scalar, testPredict*scalar))/len(testPredict)
	# 	print('Test Score: %.4f RMSE' % (testScore))

	# 	# plot baseline and predictions
	# 	plt.figure()


	# 	plt.plot_date(dates, datas_untransformed[:, 4], '-', label="All data")
	# 	print len(dates[dataLength:len(trainPredict)+dataLength]), len(trainPredict)
	# 	print len(dates[len(trainPredict)+dataLength+1:-labelLength]), len(testPredict)
	# 	plt.plot_date(dates[dataLength:len(trainPredict)+dataLength], trainPredict*scalar, '-', label="Trained Prediction {0}".format(i))
	# 	plt.plot_date(dates[len(trainPredict)+dataLength+1:-labelLength], testPredict*scalar, '-', label="Test Prediction {0}".format(i))
	# 	plt.gcf().autofmt_xdate()
	# 	plt.title("{0}/{1} market from {2} for interval {3}".format("bittrex", "USD", "BTC", "1d"))
	# 	plt.xlabel("Date")
	# 	plt.ylabel("Value ({0})".format("USD"))
	# 	# plt.legend()


	# else:
	# Batching groups
	training_batches, training_times = create_batches(training_set, dates[:train_length], dataLength, labelLength)
	testing_batches, testing_times = create_batches(testing_set, dates[train_length:], dataLength, labelLength)


	trainPredict = model.predict(training_batches, batch_size=batch_size)
	testPredict = model.predict(testing_batches, batch_size=batch_size)
	

	# plot baseline and predictions
	plt.figure()

	# print training_batches[2][0,:]
	plt.plot_date(dates, datas_untransformed[:, 4], 'k--', label="All data")
	for i in range(len(training_times)):
		# print [training_times[i]-dataLength+1+k for k in range(dataLength)]
		t_in = [training_times[i]+k-labelLength for k in range(dataLength)]
		data_in = training_batches[2][i,:]*scalar
		t_out = [training_times[i]+dataLength+k-labelLength for k in range(labelLength)]
		data_out = trainPredict[i,:]*scalar
		plt.plot_date(t_in, data_in, 'g-', label="Training Input {0}".format(i))
		plt.plot_date(t_out, data_out, 'r-x', label="Trained Prediction {0}".format(i))
		plt.plot_date([t_in[-1], t_out[0]], [data_in[-1], data_out[0]], 'r--', label="Trained Prediction {0}".format(i))
		if i != len(training_times)-1:
			plt.plot_date([t_out[-1], training_times[i+1]-labelLength], [data_out[-1], (training_batches[2][i+1,0]*scalar)], 'r--', label="Trained Prediction {0}".format(i))
	for i in range(len(testing_times)-1):
		t_in = [testing_times[i]+k-labelLength for k in range(dataLength)]
		data_in = testing_batches[2][i,:]*scalar
		t_out = [testing_times[i]+dataLength+k-labelLength for k in range(labelLength)]
		data_out = testPredict[i,:]*scalar
		plt.plot_date(t_in, data_in, 'g-', label="Testing Input {0}".format(i))
		plt.plot_date(t_out, data_out, 'b-x', label="Testing Prediction {0}".format(i))
		plt.plot_date([t_in[-1], t_out[0]], [data_in[-1], data_out[0]], 'b--')
		if i != len(testing_times)-2:
			plt.plot_date([t_out[-1], testing_times[i+1]-labelLength], [data_out[-1], (testing_batches[2][i+1,0]*scalar)], 'b--')

	plt.gcf().autofmt_xdate()
	plt.title("{0}/{1} market from {2} for interval {3}".format("bittrex", "USD", "BTC", "1d"))
	plt.xlabel("Date")
	plt.ylabel("Value ({0})".format("USD"))
	# plt.legend()



	

	plt.show()

if __name__ == "__main__":
	main(sys.argv)

from ccxt_api import ccxt_api
from featureCalculator import FeatureCalculator
import time
import cPickle as pickle
from matplotlib.finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# class CandlePlot:
# 	def __init__(self, base, market , axis, increments="day"):
# 		self.axis = axis
# 		self.data = []
# 		self.market=market
# 		self.base=base
# 		self.increments=increments


# 	def bittrex_load(self, candles):
# 		for candle in candles:
# 			t = datetime.strptime(candle['T'], '%Y-%m-%dT%H:%M:%S')
# 			t = mdates.date2num(t)
# 			append = t, candle['O'], candle['H'], candle['L'], candle['C'], candle['V']
# 			self.data.append(append)
# 		print len(self.data)	
# 		# print self.data


# 	def plot(self):
# 		# print self.close - self.open
# 		candlestick_ohlc(self.axis, self.data,width=0.01, colorup='#77d879', colordown='#db3f3f')
# 		if self.increments == "day":
# 			self.axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# 			self.axis.xaxis.set_major_locator(mticker.MaxNLocator(10))
# 		elif self.increments == "min":
# 			self.axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
# 			self.axis.xaxis.set_major_locator(mticker.MaxNLocator(10))


# 		for label in self.axis.xaxis.get_ticklabels():
# 			label.set_rotation(45)
# 		title = "{0}-{1}".format(self.market.upper(),self.base.upper())
# 		plt.title(title)
# 		plt.xlabel('Date')
# 		plt.ylabel('Price ({0})'.format(self.base))
# 		plt.legend()
# 		plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)





# buff = MathCircularBuffer(capacity=3)
# buff.append(1.)
# print buff.variance()
# buff.append(2.)
# print buff.variance()
# buff.append(3.)
# print buff.variance()
# values = [1,2,3,3,4,4,5,6,10,20]
# capacity = len(values)
# x = MathCircularBuffer(capacity=capacity)
# for i in range(capacity):
# 	x.append(values[i])
# print x.data, x.standardDeviation()

markets=["ETH", "NEO", "OMG", "BTC", "XRP", "XVG", "ADA", "ETC", "BCC", "BTG", "NXT", "LTC", "XMR", "ZEC", "DASH"]
# markets=["ETH", "NEO"]
base=["USD"]

exchanges = ["bittrex", "binance", "gdax", "okex", "bitfinex"]
# exchanges = ["bittrex"]
api = ccxt_api(load_markets=False, exchanges=exchanges)

featureCalculator = FeatureCalculator(api, base=base, markets=markets, increments="1m", label_cutoff=0.005)
# markets=["OMG"]
# featureCalculator2 = FeatureCalculator(api, base=base, markets=markets, increments="1m", label_cutoff=0.005)

# featureCalculator.loadTrainingDataFromWeb()
# featureCalculator.saveCandlesToFile("latest_candles.pickle")
# featureCalculator.loadCandlesFromFile("latest_candles.pickle")
# featureCalculator.loadLSTMFeatures()
# featureCalculator.saveTrainingDataToFile("latest_features.csv")

# featureCalculator2.loadCandlesFromFile("latest_candles.pickle")

# featureCalculator2.loadFeatures()

# featureCalculator.loadFeatures()



featureCalculator.loadTrainingDataFromFile("latest_features.csv")

# featureCalculator2.loadTrainingDataFromWeb()
# featureCalculator2.saveTrainingDataToFile("latest_data1m_NXT.csv")
# featureCalculator2.loadTrainingDataFromFile("latest_data1m_NXT.csv")

features_all = featureCalculator.getFeatures()
labels_all = featureCalculator.getLabels()
classifications_all = featureCalculator.getClassifications()

features, features2, classifications, classifications2 = train_test_split(features_all, classifications_all, test_size=0.2, random_state=0)

# features2 = featureCalculator2.getFeatures()
# labels2 = featureCalculator2.getLabels()
# classifications2 = featureCalculator2.getClassifications()
# print features.shape, features2.shape



# model = Sequential()
# model.add(LSTM(4, input_shape=(1, look_back)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)



# print "DECISION TREE"
# clf = tree.DecisionTreeClassifier().fit(features, classifications)
# print clf.score(features, classifications)*100
# print clf.score(features2, classifications2)*100
# y_pred1 = clf.predict(features)
# y_pred = clf.predict(features2)
# print confusion_matrix(classifications2, y_pred)
# print(classification_report(classifications, y_pred1))
# print(classification_report(classifications2, y_pred))

# print "Gaussian Naive Bayes"
# gnb = RandomForestClassifier(n_estimators=20).fit(features, classifications)
# print gnb.score(features, classifications)*100
# print gnb.score(features2, classifications2)*100
# y_pred1 = clf.predict(features)
# y_pred = gnb.predict(features2)
# print confusion_matrix(classifications2, y_pred)
# print(classification_report(classifications, y_pred1))
# print(classification_report(classifications2, y_pred))




# folds = 10
# for i in range(folds):
# 	X_train, X_test, y_train, y_test = train_test_split(features, classifications, test_size=0.2, random_state=0)
# 	clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
# 	print "Fold {0} with {1} trained and {2} tested".format(i+1, len(X_train), len(X_test))
# 	print "Trained Data Score: {0}".format(clf.score(X_train, y_train)*100)
# 	print "Test Data Score: {0}".format(clf.score(X_test, y_test)*100)
# 	y_pred = clf.predict(X_test)
# 	print confusion_matrix(y_test, y_pred)
	# print clf.score(X_test, y_test)



# n, bins, patches = plt.hist(classifications_all, 20)



featureCalculator.printFeatureVsTime(0, title="Feature 0")
# featureCalculator.printFeatureVsTime(30, title="Feature 30")
# featureCalculator.printFeatureVsTime(31, title="Feature 31")
# featureCalculator.printFeatureVsTime(32, title="Feature 32")
# featureCalculator.printFeatureVsTime(33, title="Feature 33")
# featureCalculator.printRelationshipWithFeature(0, 2, title="Feature 0 vs. 2")
# featureCalculator.printFeatureVsTime(0, title="HL 1")
# featureCalculator.printFeatureVsTime(4, title="4")
# featureCalculator.printFeatureVsTime(5, title="5")
# featureCalculator.printFeatureVsTime(32, title="32")
# featureCalculator.printFeatureVsTime(33, title="33")



# featureCalculator.printRelationshipWithLabels(4, title="OC 4")
# featureCalculator.printRelationshipWithLabels(5, title="Vol 4")
# featureCalculator.printRelationshipWithLabels(6, title="HL 8")
# featureCalculator.printRelationshipWithLabels(7, title="OC 8")
# featureCalculator.printRelationshipWithLabels(8, title="Vol 8")
# featureCalculator.printRelationshipWithLabels(9, title="HL 16")
# featureCalculator.printRelationshipWithLabels(10, title="OC 16")
# featureCalculator.printRelationshipWithLabels(11, title="Vol 16")
# print featureCalculator.features

plt.show()


# fig = plt.figure()
# ax1 = plt.subplot2grid((1,1), (0,0))
# bittrex = bittrex_api()
# candles = bittrex.get_candles("{0}-{1}".format(base, market), 'hour')
# plot = CandlePlot(base, market, ax1, increments="day")
# plot.bittrex_load(candles)
# plot.plot()
# plt.show()

# print bittrex.get_balances()
# print bittrex.get_balance("OMG")
# time.sleep(1)
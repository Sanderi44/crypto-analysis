from bittrex_api import bittrex_api
from ccxt_api import ccxt_api
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


class CandlePlot:
	def __init__(self, base, market , axis, increments="day"):
		self.axis = axis
		self.data = []
		self.market=market
		self.base=base
		self.increments=increments


	def bittrex_load(self, candles):
		for candle in candles:
			t = datetime.strptime(candle['T'], '%Y-%m-%dT%H:%M:%S')
			t = mdates.date2num(t)
			append = t, candle['O'], candle['H'], candle['L'], candle['C'], candle['V']
			self.data.append(append)
		print len(self.data)	
		# print self.data


	def plot(self):
		# print self.close - self.open
		candlestick_ohlc(self.axis, self.data,width=0.01, colorup='#77d879', colordown='#db3f3f')
		if self.increments == "day":
			self.axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
			self.axis.xaxis.set_major_locator(mticker.MaxNLocator(10))
		elif self.increments == "min":
			self.axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
			self.axis.xaxis.set_major_locator(mticker.MaxNLocator(10))


		for label in self.axis.xaxis.get_ticklabels():
			label.set_rotation(45)
		title = "{0}-{1}".format(self.market.upper(),self.base.upper())
		plt.title(title)
		plt.xlabel('Date')
		plt.ylabel('Price ({0})'.format(self.base))
		plt.legend()
		plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)



class MathCircularBuffer:
	def __init__(self, capacity=1):
		self.setCapacity(capacity)


	def setCapacity(self, capacity=1):
		self.capacity = capacity
		self.clear()

	def clear(self):
		self.data = []
		self.sum = 0
		self.sum_squared = 0

	def append(self, val):
		if len(self.data) == self.capacity:
			last = self.data.pop(0)
			self.sum -= last
			self.sum_squared -= last*last
		self.data.append(val)

		self.sum += val
		self.sum_squared += val*val

	def full(self):
		return len(self.data) == self.capacity

	def mean(self):
		return self.sum/len(self.data)

	def variance(self):
		if len(self.data) < 2:
			return 0.
		variance = ((self.sum_squared/float(len(self.data))) - (self.sum/float(len(self.data))*(self.sum/float(len(self.data)))))
		# variance = np.var(self.data)
		if variance < 0.0:
			return 0.0
		# else:
		return variance

	def standardDeviation(self):
		if len(self.data) < 2:
			return 0.
		var1 = self.variance()		
		res = np.sqrt(var1)
		return res


	def max(self):
		return np.max(self.data)

	def min(self):
		return np.min(self.data)

	def sum(self):
		return self.sum

	def sumSquared(self):
		return self.sum_squared



class CandleSeries:
	def __init__(self, capacity=1, base="USDT", market="BTC"):
		self.setCapacity(capacity)
		self.base=base
		self.market=market

	def setCapacity(self, capacity=1):
		self.capacity=capacity
		self.times=MathCircularBuffer(capacity=capacity)
		self.highs=MathCircularBuffer(capacity=capacity)
		self.lows=MathCircularBuffer(capacity=capacity)
		self.opens=MathCircularBuffer(capacity=capacity)
		self.closes=MathCircularBuffer(capacity=capacity)
		self.mids_time=MathCircularBuffer(capacity=capacity)
		self.mids_HL=MathCircularBuffer(capacity=capacity)
		self.volumes=MathCircularBuffer(capacity=capacity)

	def clear(self):
		self.times.clear()
		self.highs.clear()
		self.lows.clear()
		self.opens.clear()
		self.closes.clear()
		self.mids_time.clear()
		self.mids_HL.clear()
		self.volumes.clear()

	def full(self):
		return self.times.full()


	def addCandle(self, candle):
		# t = datetime.strptime(candle['T'], '%Y-%m-%dT%H:%M:%S')
		# t = mdates.date2num(t)
		# print candle
		# t = datetime.fromtimestamp(candle[0]/1000)
		# if len(self.times.data) == 0:
		self.t0 = candle[0]
		self.times.append(candle[0])
		self.opens.append(candle[1])
		self.highs.append(candle[2])
		self.lows.append(candle[3])
		self.closes.append(candle[4])
		self.mids_time.append( (candle[4] + candle[1])/2. )
		self.mids_HL.append( (candle[3] + candle[2])/2. )
		self.closes.append(candle[4])
		self.volumes.append(candle[5])

	def calculateHLRangeNormalized(self):
		min_ = self.lows.min()
		max_ = self.highs.max()
		res = (max_ - min_)/min_
		# print max_, min_, res
		return res

	def calculateOCNormalized(self):
		open_ = self.opens.data[0]
		res = (self.closes.data[-1] - open_)/open_
		# print self.opens.data, self.closes.data, res
		return res

	def calculateVolumePercentChange(self):
		if self.volumes.data[0] > 0:
			res = (self.volumes.data[-1] - self.volumes.data[0])/self.volumes.data[0]
			return res
		return 0.

	def calculateCenterTimeSlope(self):
		sum_xx = self.mids_time.sumSquared()
		sum_xy = sum([self.mids_time.data[i]*i for i in range(len(self.mids_time.data))])
		return sum_xx/sum_xy

	def calculateCenterHLSlope(self):
		sum_xx = self.mids_HL.sumSquared()
		sum_xy = sum([self.mids_HL.data[i]*i for i in range(len(self.mids_HL.data))])
		return sum_xx/sum_xy

	def calculateCenterTimeSlopePercentage(self):
		m = self.calculateCenterTimeSlope()
		y = self.mids_time.mean()
		x = sum(range(len(self.mids_time.data)))/len(self.mids_time.data)
		b = y - m*x
		return m/b

	def calculateCenterHLSlopePercentage(self):
		m = self.calculateCenterHLSlope()
		y = self.mids_HL.mean()
		x = sum(range(len(self.mids_time.data)))/len(self.mids_time.data)
		b = y - m*x
		return m/b


	def calculatePositiveMomentum(self):
		return (self.highs.max() - self.closes.data[-1])/self.closes.data[-1]

	def calculateNegativeMomentum(self):
		return (self.lows.max() - self.closes.data[-1])/self.closes.data[-1]

class FeatureCalculator:
	def __init__(self, api, base="USDT", markets=["BTC"], increments="day", label_cutoff=0.05):
		print "Initializing Feature Calculator"
		self.increments = increments
		self.base = base
		self.markets = markets
		self.features = []
		self.labels = []
		self.classifications = []
		self.api = api
		self.candleSets = []
		self.times = []
		self.label_cutoff = label_cutoff


	def loadTrainingDataFromWeb(self):
		# self.candleSet = []
		for base in self.base:
			for market in self.markets:
				candle_sets = self.api.get_candles(base, market, self.increments)
				data = {
						"base": base,
						"market": market,
						"increments": self.increments,
						"set": candle_sets
						}

				self.candleSets.append(data)
				# candles = self.bittrex.get_candles("{0}-{1}".format(self.base, market), self.increments)


	def loadLSTMFeatures(self, length=3):
		print len(self.candleSets)
		for candle_set in self.candleSets:
			
			for candles in candle_set["set"]:
				# print len(candles)
				base = candle_set["base"]
				market = candle_set["market"]


				if candles is None or len(candles) < length:
					print "Got No Data for {0}-{1} pair".format(base, market)
					continue
				




				# candleSeries1 = CandleSeries(capacity=1, base=base, market=market)
				# candleSeries2 = CandleSeries(capacity=2, base=base, market=market)
				# candleSeries4 = CandleSeries(capacity=4, base=base, market=market)
				# candleSeries8 = CandleSeries(capacity=8, base=base, market=market)
				# candleSeries16 = CandleSeries(capacity=16, base=base, market=market)
				# print "Adding {0}-{1} data with {2} candles".format(base, market, len(candles))
				candleSeries = CandleSeries(capacity=length, base=base, market=market)

				for i in range(length-1, len(candles)-1):
					try:
						candleSeries.addCandle(np.array(np.array(candles)[i,:]))
						# o = np.array(candles)[i-length:i,1]
						# h = np.array(candles)[i-length:i,2]
						# l = np.array(candles)[i-length:i,3]
						# c = np.array(candles)[i-length:i,4]
						# v = np.array(candles)[i-length:i,5]


						if candleSeries.full():


							mean_o = candleSeries.opens.mean()
							mean_h = candleSeries.highs.mean()
							mean_l = candleSeries.lows.mean()
							mean_c = candleSeries.closes.mean()
							mean_v = candleSeries.volumes.mean()

							std_o = candleSeries.opens.standardDeviation()
							std_h = candleSeries.highs.standardDeviation()
							std_l = candleSeries.lows.standardDeviation()
							std_c = candleSeries.closes.standardDeviation()
							std_v = candleSeries.volumes.standardDeviation()
							# print std_o, std_h, std_l, std_c, std_v
							if np.isnan(std_o):
								# print np.array(candles)[i-length:i,1]
								continue
							if std_o == 0. or std_h == 0. or std_l == 0. or std_c == 0. or std_v == 0.:
								# print np.array(candles)[i-length:i,1]
								continue

							f = []
							for j in range(length):
								# print candles

								f.append((candleSeries.opens.data[j] - mean_o)/std_o)
								f.append((candleSeries.highs.data[j] - mean_h)/std_h)
								f.append((candleSeries.lows.data[j] - mean_l)/std_l)
								f.append((candleSeries.closes.data[j] - mean_c)/std_c)
								f.append((candleSeries.volumes.data[j] - mean_v)/std_v)

							# print len(f)
							self.features.append(f)
							self.labels.append((candles[i+1][4] - mean_c)/std_c)
							self.classifications.append(candles[i+1][4] > candles[i+1][1])
							self.times.append(candles[i][0])


					except Exception as e:
						print e, np.array(candles)[i-length:i,1]
						pass



	def loadFeatures(self):
		for candle_set in self.candleSets:
			# print "Candle Sets: " + str(len(candle_set))
			for candles in candle_set["set"]:
				base = candle_set["base"]
				market = candle_set["market"]
				if candles is None or len(candles) < 63:
					print "Got No Data for {0}-{1} pair".format(base, market)
					continue
				candleSeries1 = CandleSeries(capacity=1, base=base, market=market)
				candleSeries2 = CandleSeries(capacity=2, base=base, market=market)
				candleSeries4 = CandleSeries(capacity=4, base=base, market=market)
				candleSeries8 = CandleSeries(capacity=8, base=base, market=market)
				candleSeries16 = CandleSeries(capacity=16, base=base, market=market)
				print "Adding {0}-{1} data with {2} candles".format(base, market, len(candles))
				for candle, i in zip(candles, range(len(candles))):
					try:
						candleSeries1.addCandle(candle)
						candleSeries2.addCandle(candle)
						candleSeries4.addCandle(candle)
						candleSeries8.addCandle(candle)
						candleSeries16.addCandle(candle)
						# print i
						if candleSeries16.full() and i != len(candles)-1:
							f = []
							f.append((candles[i][4] - candles[i][1])/candles[i][1]) #0
							f.append((candles[i][2] - candles[i][3])/candles[i][3]) #1
							f.append(candleSeries1.calculateHLRangeNormalized()) #2
							f.append(candleSeries1.calculateOCNormalized()) #3
							f.append(candleSeries1.calculatePositiveMomentum())
							f.append(candleSeries1.calculateNegativeMomentum())
							f.append(candleSeries2.calculateHLRangeNormalized())
							f.append(candleSeries2.calculateOCNormalized())
							f.append(candleSeries2.calculateVolumePercentChange())
							f.append(candleSeries2.calculateCenterTimeSlopePercentage())
							f.append(candleSeries2.calculateCenterHLSlopePercentage())
							f.append(candleSeries2.calculatePositiveMomentum())
							f.append(candleSeries2.calculateNegativeMomentum())
							f.append(candleSeries4.calculateHLRangeNormalized())
							f.append(candleSeries4.calculateOCNormalized())
							f.append(candleSeries4.calculateVolumePercentChange())
							f.append(candleSeries4.calculateCenterTimeSlopePercentage())
							f.append(candleSeries4.calculateCenterHLSlopePercentage())
							f.append(candleSeries4.calculatePositiveMomentum())
							f.append(candleSeries4.calculateNegativeMomentum())
							f.append(candleSeries8.calculateHLRangeNormalized())
							f.append(candleSeries8.calculateOCNormalized())
							f.append(candleSeries8.calculateVolumePercentChange())
							f.append(candleSeries8.calculateCenterTimeSlopePercentage())
							f.append(candleSeries8.calculateCenterHLSlopePercentage())
							f.append(candleSeries8.calculatePositiveMomentum())
							f.append(candleSeries8.calculateNegativeMomentum())
							f.append(candleSeries16.calculateHLRangeNormalized())
							f.append(candleSeries16.calculateOCNormalized())
							f.append(candleSeries16.calculateVolumePercentChange())
							f.append(candleSeries16.calculateCenterTimeSlopePercentage())
							f.append(candleSeries16.calculateCenterHLSlopePercentage())
							f.append(candleSeries16.calculatePositiveMomentum())
							f.append(candleSeries16.calculateNegativeMomentum())
							# if len(f) > 34:
							# 	print market

							# l = (candles[i+1]['C'] - candles[i+1]['O'])/candles[i+1]['O']
							m = (candles[i+1][4] - candles[i+1][1])/candles[i+1][1]
							if m < -self.label_cutoff:
								l = 0
							elif m >= -self.label_cutoff and m < 0.:
								l = 0
							elif m >= 0. and m < self.label_cutoff:
								l = 1
							else:
								l = 1
							# print l
							self.features.append(f)
							self.labels.append(m)
							self.classifications.append(l)
							self.times.append(candleSeries1.times.data[0])
					except:
						break

	def saveCandlesToFile(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self.candleSets, f, pickle.HIGHEST_PROTOCOL)


	def loadCandlesFromFile(self, filename):
		with open(filename, 'rb') as f:
			self.candleSets = pickle.load(f)

	def saveTrainingDataToFile(self, filename):
		import csv
		with open(filename, 'wb') as csvfile:
			csvwrite = csv.writer(csvfile, delimiter=',')
			for i in range(len(self.features)):
				row = self.features[i][:]
				row.append(self.labels[i])
				row.append(self.times[i])
				csvwrite.writerow(row)

	def loadTrainingDataFromFile(self, filename):
		import csv
		with open(filename, 'rb') as csvfile:
			csvread = csv.reader(csvfile, delimiter=',')
			print "Loading datas from {0}".format(filename)
			for row in csvread:
				row = [float(x) for x in row]
				self.features.append(row[:-2])
				if row[-2] < -self.label_cutoff:
					l = 0
				elif row[-2] >= -self.label_cutoff and row[-2] < 0.:
					l = 0
				elif row[-2] >= 0. and row[-2] < self.label_cutoff:
					l = 1
				else:
					l = 1
				self.labels.append(row[-2])
				self.classifications.append(l)
				self.times.append(row[-1])
			print "File has {0} datas".format(len(self.times))



	def printRelationshipWithLabels(self, feature_num, title="title"):
		plt.figure()
		plt.plot(np.array(self.labels), np.array(self.features)[:,feature_num], ".")
		plt.title(title)
		plt.ylabel("Feature {0}".format(feature_num))
		plt.xlabel("Label")

	def printRelationshipWithFeature(self, feature_num1, feature_num2, title="title"):
		plt.figure()
		plt.plot(np.array(self.features)[:,feature_num1], np.array(self.features)[:,feature_num2], ".")
		plt.title(title)
		plt.xlabel("Feature {0}".format(feature_num1))
		plt.ylabel("Feature {0}".format(feature_num2))

	def printFeatureVsTime(self, feature_num, title="title"):
		plt.figure()
		plt.plot(self.times, np.array(self.features)[:,feature_num])
		plt.title(title)
		plt.xlabel("Time")
		plt.ylabel("Feature {0}".format(feature_num))
	# def calculateFeatures(self):
	# 	print self.candleSet.calculateOCNormalized()

	def getFeatures(self):
		return np.array(self.features)

	def getLabels(self):
		return np.array(self.labels)

	def getTimes(self):
		return np.array(self.times)

	def getClassifications(self):
		return np.array(self.classifications)





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



model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)



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
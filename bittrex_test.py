from bittrex_api import bittrex_api
import time
from matplotlib.finance import candlestick_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix




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
		return (self.sum_squared - (self.sum*self.sum/len(self.data)))/(len(self.data))

	def standardDeviation(self):
		return np.sqrt(self.variance())

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
		t = datetime.strptime(candle['T'], '%Y-%m-%dT%H:%M:%S')
		t = mdates.date2num(t)
		self.times.append(t)
		self.opens.append(candle['O'])
		self.highs.append(candle['H'])
		self.lows.append(candle['L'])
		self.closes.append(candle['C'])
		self.mids_time.append( (candle['C'] + candle['O'])/2. )
		self.mids_HL.append( (candle['H'] + candle['L'])/2. )
		self.closes.append(candle['C'])
		self.volumes.append(candle['V'])

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
		return (self.volumes.data[-1] - self.volumes.data[0])/self.volumes.data[0]

	def calculateCenterTimeSlope(self):
		sum_xx = self.mids_time.sumSquared()
		sum_xy = sum([self.mids_time.data[i]*self.times.data[i] for i in range(len(self.mids_time.data))])
		return sum_xx/sum_xy

	def calculateCenterHLSlope(self):
		sum_xx = self.mids_HL.sumSquared()
		sum_xy = sum([self.mids_HL.data[i]*self.times.data[i] for i in range(len(self.mids_HL.data))])
		return sum_xx/sum_xy

	def calculatePositiveMomentum(self):
		return (self.highs.max() - self.closes.data[-1])/self.closes.data[-1]

	def calculateNegativeMomentum(self):
		return (self.lows.max() - self.closes.data[-1])/self.closes.data[-1]

class FeatureCalculator:
	def __init__(self, base="USDT", markets=["BTC"], increments="day", label_cutoff=0.075):
		print "Initializing Feature Calculator"
		self.increments = increments
		self.base = base
		self.markets = markets
		self.features = []
		self.labels = []
		self.classifications = []
		self.bittrex = bittrex_api()
		self.candleSet = []
		self.times = []
		self.label_cutoff = label_cutoff


	def loadTrainingDataFromWeb(self):
		# self.candleSet = []
		for market in self.markets:
			candles = self.bittrex.get_candles("{0}-{1}".format(self.base, market), self.increments)
			if candles is None:
				print "Got No Data for {0}-{1} pair".format(self.base, market)
				continue
			candleSeries1 = CandleSeries(capacity=1, base=self.base, market=market)
			candleSeries2 = CandleSeries(capacity=8, base=self.base, market=market)
			candleSeries4 = CandleSeries(capacity=16, base=self.base, market=market)
			candleSeries8 = CandleSeries(capacity=32, base=self.base, market=market)
			candleSeries16 = CandleSeries(capacity=64, base=self.base, market=market)
			print "Gathering {0}-{1} data with {2} candles".format(self.base, market, len(candles))
			for candle, i in zip(candles, range(len(candles))):
				candleSeries1.addCandle(candle)
				candleSeries2.addCandle(candle)
				candleSeries4.addCandle(candle)
				candleSeries8.addCandle(candle)
				candleSeries16.addCandle(candle)
				# print i
				if candleSeries16.full() and i != len(candles)-1:
					f = []
					f.append((candles[i]['C'] - candles[i]['O'])/candles[i]['O']) #0
					f.append((candles[i]['H'] - candles[i]['L'])/candles[i]['L']) #1
					f.append(candleSeries1.calculateHLRangeNormalized()) #2
					f.append(candleSeries1.calculateOCNormalized()) #3
					f.append(candleSeries1.calculatePositiveMomentum())
					f.append(candleSeries1.calculateNegativeMomentum())
					f.append(candleSeries2.calculateHLRangeNormalized())
					f.append(candleSeries2.calculateOCNormalized())
					f.append(candleSeries2.calculateVolumePercentChange())
					f.append(candleSeries2.calculateCenterTimeSlope())
					f.append(candleSeries2.calculateCenterHLSlope())
					f.append(candleSeries2.calculatePositiveMomentum())
					f.append(candleSeries2.calculateNegativeMomentum())
					f.append(candleSeries4.calculateHLRangeNormalized())
					f.append(candleSeries4.calculateOCNormalized())
					f.append(candleSeries4.calculateVolumePercentChange())
					f.append(candleSeries4.calculateCenterTimeSlope())
					f.append(candleSeries4.calculateCenterHLSlope())
					f.append(candleSeries4.calculatePositiveMomentum())
					f.append(candleSeries4.calculateNegativeMomentum())
					f.append(candleSeries8.calculateHLRangeNormalized())
					f.append(candleSeries8.calculateOCNormalized())
					f.append(candleSeries8.calculateVolumePercentChange())
					f.append(candleSeries8.calculateCenterTimeSlope())
					f.append(candleSeries8.calculateCenterHLSlope())
					f.append(candleSeries8.calculatePositiveMomentum())
					f.append(candleSeries8.calculateNegativeMomentum())
					f.append(candleSeries16.calculateHLRangeNormalized())
					f.append(candleSeries16.calculateOCNormalized())
					f.append(candleSeries16.calculateVolumePercentChange())
					f.append(candleSeries16.calculateCenterTimeSlope())
					f.append(candleSeries16.calculateCenterHLSlope())
					f.append(candleSeries16.calculatePositiveMomentum())
					f.append(candleSeries16.calculateNegativeMomentum())
					# if len(f) > 34:
					# 	print market

					# l = (candles[i+1]['C'] - candles[i+1]['O'])/candles[i+1]['O']
					m = (candles[i+1]['C'] - candles[i+1]['O'])/candles[i+1]['O']
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
				# t = datetime.strptime(candle['T'], '%Y-%m-%dT%H:%M:%S')
				# t = mdates.date2num(t)
				# append = t, candle['O'], candle['H'], candle['L'], candle['C'], candle['V'], "{0}-{1}".format(self.base, market)
				# self.raw_data.append(append)
			# self.candleSet.append(candleSeries)

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
		plt.plot(np.array(self.features)[:,feature_num], np.array(self.classifications), ".")
		plt.title(title)
		plt.xlabel("Feature {0}".format(feature_num))
		plt.ylabel("Classification (up/down)")

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

markets=["OMG", "NEO", "XRP", "XMR", "ADA", "LTC", "ETC", "ZEC", "BCC", "XLM", "QTUM", "POWR", "STRAT", "BAT", "XEM"]
base="ETH"



# buff = MathCircularBuffer(capacity=3)
# buff.append(1.)
# print buff.variance()
# buff.append(2.)
# print buff.variance()
# buff.append(3.)
# print buff.variance()






featureCalculator = FeatureCalculator(base=base, markets=markets, increments="oneMin", label_cutoff=0.005)
markets=["SC", "FUN", "SNT", "DASH"]
featureCalculator2 = FeatureCalculator(base=base, markets=markets, increments="oneMin", label_cutoff=0.005)

# featureCalculator.loadTrainingDataFromWeb()
# featureCalculator.saveTrainingDataToFile("latest_data1m.csv")
featureCalculator.loadTrainingDataFromFile("latest_data1m.csv")

featureCalculator2.loadTrainingDataFromWeb()
# featureCalculator2.saveTrainingDataToFile("latest_data1m_NXT.csv")

features = featureCalculator.getFeatures()
labels = featureCalculator.getLabels()
classifications = featureCalculator.getClassifications()
features2 = featureCalculator2.getFeatures()
labels2 = featureCalculator2.getLabels()
classifications2 = featureCalculator2.getClassifications()
print features.shape, features2.shape

clf = tree.DecisionTreeClassifier().fit(features, classifications)
print clf.score(features, classifications)*100
print clf.score(features2, classifications2)*100
y_pred = clf.predict(features2)
print confusion_matrix(classifications2, y_pred)


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



n, bins, patches = plt.hist(labels, 20)



featureCalculator.printRelationshipWithLabels(11, title="Feature 4")
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
import numpy as np
import matplotlib.pyplot as plt
from candleSeries import CandleSeries
import cPickle as pickle


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
		# print len(self.candleSets)
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
				# candleSeries = CandleSeries(capacity=length, base=base, market=market)

				for i in range(length-1, len(candles)-1):
					try:
						# candleSeries.addCandle(np.array(np.array(candles)[i,:]))
						# o = np.array(candles)[i-length:i,1]
						# h = np.array(candles)[i-length:i,2]
						# l = np.array(candles)[i-length:i,3]
						c = np.array(candles)[i-length:i,4]
						# v = np.array(candles)[i-length:i,5]

						if len(c) == length:
							f = np.copy(c)


							# mean_o = candleSeries.opens.mean()
							# mean_h = candleSeries.highs.mean()
							# mean_l = candleSeries.lows.mean()
							# mean_c = candleSeries.closes.mean()
							# mean_v = candleSeries.volumes.mean()

							# std_o = candleSeries.opens.standardDeviation()
							# std_h = candleSeries.highs.standardDeviation()
							# std_l = candleSeries.lows.standardDeviation()
							# std_c = candleSeries.closes.standardDeviation()
							# std_v = candleSeries.volumes.standardDeviation()
							# print std_o, std_h, std_l, std_c, std_v
							# if np.isnan(std_o):
							# 	# print np.array(candles)[i-length:i,1]
							# 	continue
							# if std_o == 0. or std_h == 0. or std_l == 0. or std_c == 0. or std_v == 0.:
							# 	# print np.array(candles)[i-length:i,1]
							# 	continue

							# f = []
							# for j in range(length):
							# 	# print candles

							# 	# f.append((candleSeries.opens.data[j] - mean_o)/std_o)
							# 	# f.append((candleSeries.highs.data[j] - mean_h)/std_h)
							# 	# f.append((candleSeries.lows.data[j] - mean_l)/std_l)
							# 	f.append(c[j])
							# 	# f.append((candleSeries.volumes.data[j] - mean_v)/std_v)

							# print len(f)
							self.features.append(f)
							self.labels.append(candles[i+1][4])
							self.classifications.append(candles[i+1][4] > candles[i][4])
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

	def getCandles(self):
		return self.candleSets
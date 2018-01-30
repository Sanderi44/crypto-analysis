import numpy as np

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

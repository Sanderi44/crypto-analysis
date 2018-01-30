import sys
from ccxt_api import ccxt_api
from featureCalculator import FeatureCalculator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime


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

		print key + " is " + value
	
	def print_info(self):
		print "Exchanges: " + str(self.exchanges)
		print "Bases: " + str(self.bases)
		print "Markets: " + str(self.markets)
		print "Intervals: " + str(self.intervals)
		print "Ohlcvs: " + str(self.ohlcvs)

	def get_query(self, query_num):
		if query_num < len(self.exchanges) and query_num < len(self.bases) and query_num < len(self.markets) and query_num < len(self.intervals) and query_num < len(self.ohlcvs):
			return self.exchanges[query_num], self.bases[query_num], self.markets[query_num], self.intervals[query_num], self.ohlcvs[query_num]
		else:
			self.print_info()
			raise Exception("Please specify an exchange, a base, a market and an inverval for each plot")

	def get_number_of_plots(self):
		return len(self.markets)



def plot(candles, exchange, base, market, interval, ohlcv="open"):
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
	plt.figure()
	ts = [datetime.datetime.fromtimestamp(t/1000) for t in candles[:,0]]
	dates = date2num(ts)
	plt.plot_date(dates, candles[:,ohlcv_num], '-')
	plt.gcf().autofmt_xdate()
	plt.title("{0}/{1} market {2} from {3} for interval {4}".format(base, market, ohlcv, exchange, interval))
	plt.xlabel("Date")
	plt.ylabel("Value ({0})".format(base))



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

	for i in range(parser.get_number_of_plots()):
		exchange, base, market, interval, ohlcv = parser.get_query(i)
		candles = np.array(get_candles(exchange, base, market, interval))
		plot(candles, exchange, base, market, interval, ohlcv)
		# plt.figure()
		# plt.plot_date(dates, np.log(candles[:,4]), '-')
		# plt.gcf().autofmt_xdate()

	plt.show()

if __name__ == "__main__":
	main(sys.argv)
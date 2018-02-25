import csv
import sys
import numpy as np
from ccxt_api import ccxt_api


class Parser:
	def __init__(self):
		self.exchanges = []
		self.bases = []
		self.markets = []
		self.intervals = []
	
	def parse_key(self, key, value):
		if key == "exchange":
			self.exchanges.append(value)
		if key == "base":
			self.bases.append(value)
		if key == "market":
			self.markets.append(value)
		if key == "interval":
			self.intervals.append(value)

		# print key + " is " + value
	
	def print_info(self):
		print "Exchanges: " + str(self.exchanges)
		print "Bases: " + str(self.bases)
		print "Markets: " + str(self.markets)
		print "Intervals: " + str(self.intervals)

	def get_query(self, query_num):
		if query_num < len(self.exchanges) and query_num < len(self.bases) and query_num < len(self.markets) and query_num < len(self.intervals):
			return self.exchanges[query_num], self.bases[query_num], self.markets[query_num], self.intervals[query_num]
		else:
			self.print_info()
			raise Exception("Please specify an exchange, a base, a market and an inverval for each plot")

	def get_number_of_plots(self):
		return len(self.markets)


def get_candles(exchange, base, market, interval='1m'):
	api = ccxt_api(load_markets=True, exchanges=[exchange])	
	candle_sets = api.get_candles(base, market, interval)
	return candle_sets[0]

def save_candles(filename, candles):
	with open(filename, 'w') as f:
		writer = csv.writer(f)
		for candle in candles:
			writer.writerow(candle)				 

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
		exchange, base, market, interval = parser.get_query(i)
		candles = np.array(get_candles(exchange, base, market, interval))
		filename = market+"_"+base+"_"+str(interval)+"_"+exchange+".csv"
		print filename
		save_candles(filename, candles)


if __name__ == "__main__":
	main(sys.argv)
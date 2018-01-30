import ccxt
import time
import sys
from types import UnicodeType

class ccxt_api:
	def __init__(self, load_markets=False, exchanges=[]):
		self.ccxt = ccxt
		self.exchanges = {}
		for exchange in ccxt.exchanges:
			if len(exchanges) == 0 or exchange in exchanges:
				e = eval('ccxt.%s ()' % exchange)
				if e.has['fetchOHLCV']:
					self.exchanges[exchange] = e
		if load_markets:
			self.load_markets()

	def load_markets(self):
		for exchange in self.exchanges:
			try:
				print "Loading Exchange: " + exchange
				m = self.exchanges[exchange].load_markets()
				print "--> Found " + str(len(m)) + " markets"
				# print "Rate Limit " + str(self.exchanges[exchange].rateLimit)
				time.sleep (4 * self.exchanges[exchange].rateLimit / 1000) # time.sleep wants seconds
			except (KeyboardInterrupt, SystemExit):
				print "\nExiting"
				sys.exit()
			except:
				pass

	def get_candles(self, base, market, tick_interval):
		candle_sets = []
		for exchange in self.exchanges:
			try:
				markets = self.exchanges[exchange].markets
				# print markets
				if markets is not None:
					# print "Checking " + exchange + " for " + base + "/" + market
					try:
						for symbol in markets:
							# print markets[symbol]['base'], markets[symbol]['quote']
							if (markets[symbol]['base'].find(base) > -1 and markets[symbol]['quote'].find(market) > -1) or (markets[symbol]['base'].find(market) > -1 and markets[symbol]['quote'].find(base) > -1):
								# print e.rateLimit

								time.sleep (4 * self.exchanges[exchange].rateLimit / 1000) # time.sleep wants seconds
								res = self.exchanges[exchange].fetch_ohlcv(symbol, tick_interval)
								# print exchange, res
								if type(res) is list and type(res[0]) is list:
									for v in range(len(res)):
										# print res[v]
										for q in range(len(res[v])):
											if type(res[v][q]) is UnicodeType:
												res[v][q] = float(res[v][q]) 
									candle_sets.append(res)
									print "--> " +  exchange + " has " + str(len(res)) + " samples of " + base + "/" + market + " with symbol " + symbol
								else:
									pass
									# print res, " is not a list" 
					except Exception as e:
						print e
						pass
						# print exchange + " does not have " + base + "/" + market

			except Exception as e:
				print e
				pass

		return candle_sets
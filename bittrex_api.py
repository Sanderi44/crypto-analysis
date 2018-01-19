from bittrex.bittrex import Bittrex, API_V2_0
import json



class bittrex_api:
	def __init__(self):
		key, secret = self.getSecret()
		self.bittrex = Bittrex(key, secret, api_version=API_V2_0)

	def getSecret(self):
		try:
			with open("secrets.json", 'r') as secrets_file:
				secrets = json.load(secrets_file)
				secrets_file.close()
			return secrets["key"], secrets["secret"]
		except:
			return None, None



	def print_balances(self):
		my_balances = self.bittrex.get_balances()
		# print my_balances["result"]
		if my_balances['success'] == True:
			for balance in my_balances["result"]:
				if balance["Balance"]['Available'] > 0:
					print balance["Balance"]['Currency'], balance["Balance"]['Available']


	def get_balances(self):
		my_balances = self.bittrex.get_balances()
		# print my_balances["result"]
		balances = {}
		if my_balances['success'] == True:
			for balance in my_balances["result"]:
				if balance["Balance"]['Available'] > 0:
					balances[balance["Balance"]['Currency']] = balance["Balance"]['Available']
		return balances

	def get_balance(self, coin):
		res = self.bittrex.get_balance(coin)
		if res['success'] == True:
			return res['result']['Available']

	def get_candles(self, market, tick_interval):
		res = self.bittrex.get_candles(market, tick_interval)
		if res['success'] == True:
			return res['result']
import sys
from ccxt_api import ccxt_api
from featureCalculator import FeatureCalculator



def test_ccxt_api():
	markets=["ETH", "NEO"]
	base=["USD"]
	exchanges = ["bittrex", "okex"]
	api = ccxt_api(load_markets=True, exchanges=exchanges)
	assert len(api.get_exchanges()) > 0
	return api


def test_feature_calculator(api):
	markets=["ETH", "NEO"]
	base=["USD"]
	for market in markets:
		candle_sets = api.get_candles(base[0], market, '1m')
		# print candle_sets
		assert len(candle_sets) > 0
	featureCalculator = FeatureCalculator(api, base=base, markets=markets, increments="1m", label_cutoff=0.005)
	featureCalculator.loadCandlesFromFile("latest_candles.pickle")
	assert len(featureCalculator.getCandles()) > 0
	featureCalculator.loadLSTMFeatures(length=3)
	assert len(featureCalculator.getFeatures()) > 0
	# featureCalculator.loadTrainingDataFromFile("latest_features.csv")



def main(args):
	api = test_ccxt_api()
	test_feature_calculator(api)

if __name__ == "__main__":
	main(sys.argv)
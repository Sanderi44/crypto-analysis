from ccxt_api import ccxt_api


ccxt = ccxt_api()
ccxt.get_candles("BTC", "USD", "1m")
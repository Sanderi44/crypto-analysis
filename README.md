# crypto-analysis
This is an analysis api for cryptocurrency prices.  It queries historical data using ccxt.  It is currently in development.  The goal is to be able to query large amounts of data and then run some machine learning on that data in order to gather insights into cryptocurrency prices.

Here are some examples: 
To run a plot of the highs and lows of BTC/USD Bittrex data for 1d intervals run this command:

~~~~
python plot_candles.py exchange=bittrex base=USD market=BTC interval=1d ohlcv=high exchange=bittrex base=USD market=BTC interval=1d ohlcv=low
~~~~


This runs a testing script for checking if the api works.  You can change the input base, market, exchange, and intervals to test the functionality of the api.

~~~~
python test.py 
~~~~


To run the Recurrent Neural Network that I built for XRP/USD daily close data from bittrex, with 80% of the data used as training data, run this: 

~~~~
python rnn.py filename=BTC_USD_1d_bittrex.csv batch=50 epochs=10000 name=name
~~~~

If you have questions, please send me an email via github.

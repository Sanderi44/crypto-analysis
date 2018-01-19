# crypto-analysis
This is my first attempt at building a library for cryptocurrency analysis.  The idea is to be able to gather as much data as possible from as many sources as possible so that I can do some machine learning and analysis on the data.  The classifier that I am looking at right now decides whether the price will go up or down in the next minute/hour/day based on a number of features.  With the limited data from only bittrex (at this time), I am getting a 60% classification rate on untrained data (which I think is pretty good).

As I continue to build this system, I will try to make it into an api for others to use as well.  The api's that I am using right now are bittrex and ccxt which is another open source library for pulling data from many sources.  

Eventually, I will also try to turn this into a trading bot, depending on the accuracy of the machine learning.

 
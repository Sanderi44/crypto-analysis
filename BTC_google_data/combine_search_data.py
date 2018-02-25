import csv
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt

def import_csv(filename):
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		data = list(reader)
		return data

def write_csv(filename, data):
	with open(filename, 'w') as f:
		writer = csv.writer(f, delimiter=',')
		for row in data:
			writer.writerow(row)



def main():
	file_list = ["BTC_search_01_2016-02_2016.csv",
			 "BTC_search_02_2016-05_2016.csv",
			 "BTC_search_05_2016-08_2016.csv",
			 "BTC_search_08_2016-11_2016.csv",
			 "BTC_search_11_2016-02_2017.csv",
			 "BTC_search_02_2017-05_2017.csv",
			 "BTC_search_05_2017-08_2017.csv",
			 "BTC_search_08_2017-11_2017.csv",
			 "BTC_search_11_2017-02_2018.csv",
			 ]
	x = 1
	all_vals = np.array([])
	all_dates = np.array([])
	first_val = 1
	last_val = 1
	for f in file_list:
		data = import_csv(f)
		data = data[3:]
		max_val = 0
		min_val = 100
		vals = []
		dates = []
		for item in data:
			d = datetime.datetime.strptime(item[0], '%Y-%m-%d')
			val = int(item[1])
			dates.append(d)
			vals.append(val)
				# if (val > max_val):
				# 	max_val = val
				# if (val < min_val):
				# 	min_val = val
			# print val
			# print d
		# print data
		if len(all_vals) >0:
			x = all_vals[-1]/vals[0]

			all_vals = np.concatenate((all_vals, np.array(vals[1:])*x), axis=0)
			all_dates = np.concatenate((all_dates, np.array(dates[1:])), axis=0)
		else:
			all_vals = np.concatenate((all_vals, np.array(vals)*x), axis=0)
			all_dates = np.concatenate((all_dates, np.array(dates)), axis=0)

		# if len(all_vals) > 0:
		# 	x = first_val/last_val
		# last_val = vals[-1]


		# print max_val, min_val, vals[0], vals[-1] 

	# print all_vals
	all_timestamps = [time.mktime(dt.timetuple()) for dt in all_dates]

	data = np.array([all_timestamps, (all_vals/np.max(all_vals)).tolist()]).T
	write_csv("BTC_search_01_2016-02_2018.csv", data)
	# print data
	plt.figure()
	plt.plot_date(all_dates, all_vals/np.max(all_vals), '-')
	plt.gcf().autofmt_xdate()
	# plt.title("{0}/{1} market {2} from {3} for interval {4}".format(base, market, ohlcv, exchange, interval))
	plt.xlabel("Date")
	plt.ylabel("Value ")
	plt.show()



if __name__ == "__main__":
	main()

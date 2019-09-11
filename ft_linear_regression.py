#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  train.py                                                                    #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Tuesday Sep 2019 8:45:38 pm                                       #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import argparse
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

def parse():
	parser = argparse.ArgumentParser(description='Simple linear regression model')
	parser.add_argument('-t', '--train', action='store_true')
	parser.add_argument('input')
	return parser.parse_args()

def get_data(input):
	try:
		data = pd.read_csv(input)
	except FileNotFoundError as e:
		print('ft_linear_regression: ', e)
		exit(1)
	return data

def step_gradient(X, Y, params, learning_rate):
	Y_hat = params[0] * X + params[1]
	grad_0 = -2 * sum(X * (Y - Y_hat)) / len(X)
	grad_1 = -2 * sum(Y - Y_hat) / len(X)
	# print(X)
	print(params)
	# print(Y_hat)
	# print(Y - Y_hat)
	# print((Y - Y_hat) / len(X))
	# print(X * (Y - Y_hat) / len(X))
	# print(grad_0, grad_1)
	# # print(params + (np.array([grad_0, grad_1]) * learning_rate))
	# print(np.array([grad_0, grad_1]) * learning_rate)
	print('\n')
	return params - (np.array([grad_0, grad_1]) * learning_rate)

def normalize(X, Y):
	X_norm = (X - np.min(X)) / (np.max(X) - np.min(X))
	# Y_norm = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
	# return [X_norm, Y_norm]
	return [X_norm, Y]

def normalize_point(point, width, min_val):
	return (point - min_val) / (width)

def denormalize_point(point, width, min_val):
	return point * width + min_val

def train(input):
	data = get_data(input)
	learning_rate = .1
	num_iter = 5000
	params = np.array([-2 , 0], dtype='float')
	# data.plot.scatter(x='km', y='price')
	# plt.show()
	X, Y = normalize(data['km'], data['price'])
	for i in range(num_iter):
		params = step_gradient(X, Y, params, learning_rate)
		# data.plot.scatter(x='km', y='price')
		# plt.plot(data['km'], params[0] * data['km'] + params[1])
		# plt.show()
	print(X[0])
	print(denormalize_point(X[0], np.max(data['km']) - np.min(data['km']), np.min(data['km'])))

if __name__ == '__main__':
	args = parse()
	print(args)
	if args.train:
		train(args.input)
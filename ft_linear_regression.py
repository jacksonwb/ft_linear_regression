#! /usr/bin/env python3
# ---------------------------------------------------------------------------- #
#  train.py                                                                    #
#                                                                              #
#  By - jacksonwb                                                              #
#  Created: Wednesday December 1969 4:00:00 pm                                 #
#  Modified: Wednesday Sep 2019 2:15:46 pm                                     #
#  Modified By: jacksonwb                                                      #
# ---------------------------------------------------------------------------- #

import argparse
from linreg import LinReg
import pandas  as pd

def parse():
	parser = argparse.ArgumentParser(description='Simple linear regression model')
	parser.add_argument('-t', '--train', action='store_true')
	parser.add_argument('input')
	return parser.parse_args()

def get_data(input):
	try:
		data = pd.read_csv(input)
	except FileNotFoundError as e:
		print('ft_linear_regression:', e)
		exit(1)
	return data

if __name__ == '__main__':
	args = parse()
	lin_reg_model = LinReg()
	if args.train:
		data = get_data(args.input)
		lin_reg_model.train(data['km'], data['price'],.1, 1000, plot=True)
		lin_reg_model.save_model()
	else:
		lin_reg_model.load_model()
		try:
			print(lin_reg_model.solve_for(int(args.input)))
		except Exception as e:
			print('ft_linear_regression:', e)
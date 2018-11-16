import numpy as np


class Info(object):

	def __init__(self,**args):
		self.args = args
		self.args['fisher'] = None
		self.args['data'] = []
		self.args['label'] = []

	def get_value(self,key):
		v = None
		try:
			v = self.args[key]
		except KeyError:
			pass
		return v

	def set_value(self,key,value):
		self.args[key] = value

	def add(self,key,value):
		self.args[key].append(value)

	def delete(self,key):
		del self.args[key]

	def add_data(self,data,label):
		self.args['data'].append(data)
		self.args['label'].append(label)

	def add_fisher(self,value):
		self.args['fisher'].append(value)

	def update_fisher(self,value):
		fisher = self.args['fisher']
		if fisher is None:
			self.args['fisher'] = value
		else:
			for i in len(fisher):
				indi = np.where(fisher[i] < value[i])
				fisher[indi] = value[indi]
				self.args['fisher'] = fisher



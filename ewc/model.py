import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout
from information import Info
from keras import backend as K
from utils import *
import tensorflow as tf


class Model(object):

	def __init__(self):
		self.dim = 64
		self.num_classes = 10
		self.history = None
		self.epoch = 10
		
		self.info = Info()
		self.count = 0

		
		self.model = Sequential()
		self.model.add(Dense(50,input_shape=(784,),activation='relu'))
		self.model.add(Dense(self.num_classes,activation='softmax'))

		self.y = self.model.output
		self.var_list = self.model.trainable_weights


	def val_data(self,X_val,y_val):
		self.X_val = X_val
		self.y_val = y_val


	def compute_fisher(self,X,y):
		fisher = []
		g = get_weight_grad(self.model,X[:200],y[:200])
		'''
		for i in g:
			temp = np.square(i)
			temp = temp / np.max(temp)
			fisher.append(temp/np.max(temp))
		self.info.update_fisher(fisher)
		self.FISHER = self.info.get_value('fisher')
		'''
		self.FISHER = np.square(g)
		print('--'*20,'fisher','--'*20)
		print(self.FISHER)

	def ewc_fisher(self, data_img, data_lbl, num_samples=200, eq_distrib=True):
		# initialize Fisher information for most recent task
		self.F_accum = []
		self.y = self.model.output
		sess = K.get_session()
		self.var_list = self.model.trainable_weights
		for v in range(len(self.var_list)):
			self.F_accum.append(np.zeros(self.var_list[v].shape))

        # sampling a random class from softmax
		probs = self.y
		class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

		classes = np.unique(data_lbl)
		if eq_distrib:
			# equally distributed among classes samples
			indx = []
			for cl in range(len(classes)):
				tmp = np.where(data_lbl == classes[cl])[0]
				np.random.shuffle(tmp)
				indx = np.hstack((indx, tmp[0:min(num_samples, len(tmp))]))
				indx = np.asarray(indx).astype(int)
		else:
			# random non-repeating selected images
			indx = random.sample(xrange(0, data_img.shape[0]), num_samples * len(classes))
        
		for i in range(len(indx)):
			# select random input image
			im_ind = indx[i]

			# compute first-order derivatives
			tmp = tf.gradients(tf.log(probs[0,class_ind]), self.var_list)
			for v in range(len(tmp)):
				if tmp[v] == None:
					tmp[v] = tf.zeros([1]) 
			ders = sess.run( tmp , feed_dict={ self.model.input: data_img[im_ind:im_ind+1]})

			# square the derivatives and add to total
			for v in range(len(self.F_accum)):
				self.F_accum[v] += np.square(ders[v])

		# divide totals by number of samples
		for v in range(len(self.F_accum)):
			self.F_accum[v] /= len(indx)

		print(self.F_accum[0]==self.FISHER[0])


	def star(self):
		self.star_vars = []
		for v in range(len(self.var_list)):
			self.star_vars.append(K.eval(self.var_list[v]))

	def restore(self):
		if hasattr(self,'star_vars'):
			self.model.set_weights(self.star_vars)

	def fit(self,X_train,y_train):
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		self.model.fit(X_train,y_train,epochs = self.epoch, batch_size=128, verbose = True,validation_data=(self.X_val,self.y_val))

	def ewc_loss(self,lam=15):
		print('*'*20,self.count,'*'*20)
		def loss(y_true,y_pred):
			ewc = K.categorical_crossentropy(y_true,y_pred)
			for v in range(len(self.var_list)):
				ewc += ( lam / 2 ) * tf.reduce_sum(tf.multiply(self.FISHER[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
			return ewc
		return loss


	def transfer(self,X_train,y_train):
		self.model.compile(optimizer='sgd',loss=self.ewc_loss(),metrics=['accuracy'])
		self.model.fit(X_train,y_train,epochs = self.epoch, batch_size=128, verbose = True,validation_data=(self.X_val,self.y_val))
		self.count += 1








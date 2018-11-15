from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Conv3D,MaxPooling3D,Dropout,core,Reshape,Input
from keras.callbacks import Callback,TensorBoard
from keras import backend as K
from keras.engine import Layer
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from information import Info


class cnn_model(object):
	'''
	 - Initialize the model
  	 - important parameters:
  	 	- history: it will record all the accuracy on training dataset and validation dataset
	'''
	def __init__(self):
		self.dim = 64
		self.num_classes = 10
		self.history = None
		self.epoch = 10
		
		self.info = Info()

		
		self.model = Sequential()
		self.model.add(Dense(50,input_shape=(784,),activation='relu'))
		self.model.add(Dense(self.num_classes,activation='softmax'))

		self.tbCallBack = TensorBoard(log_dir='./logs/mnist_drift/kal/',  
		histogram_freq=0,  
		write_graph=True,  
		write_grads=True, 
		write_images=True,
		embeddings_freq=0, 
		embeddings_layer_names=None, 
		embeddings_metadata=None)


	#What data is used for validation
	def val_data(self,X_test,y_test):
		self.X_test = X_test
		self.y_test = y_test

	#train the model by normal gradient descent algorithm
	def fit(self,X_train,y_train):
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		if self.history is None:
			self.history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,validation_data=(self.X_test,self.y_test))
		else:
			history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,validation_data=(self.X_test,self.y_test))
			self.history.history['acc'].extend(history.history['acc'])
			self.history.history['val_acc'].extend(history.history['val_acc'])

		self.info.add_data(X_train,y_train)

	'''
	 - This function is used for 'kal' algorithm.
	 - The model will calculate the gradient on D1[batch0], and never access to D1 again
	'''
	def transfer(self,X_train,y_train,num=2):
		self.info.add_data(X_train,y_train)
		opcallback = op_pre_callback(info=self.info,use_pre=False)
		#self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,callbacks=[opcallback],validation_data=(self.X_test,self.y_test))
		self.history.history['acc'].extend(history.history['acc'])
		self.history.history['val_acc'].extend(history.history['val_acc'])

	'''
	 - This function is used for 'kal_pre' algorithm
	 - The model will access to D1 to calculate the gradient during all training process
	'''
	def use_pre(self,X_train,y_train):
		opcallback = op_pre_callback(use_pre=True)
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,callbacks=[opcallback],validation_data=(self.X_test,self.y_test))
		self.history.history['acc'].extend(history.history['acc'])
		self.history.history['val_acc'].extend(history.history['val_acc'])

	'''
	 - This funciton is used for 'kal_cur' algorithm
	 - The model won't access to D1. All the gradients are calculated on current training data
	'''
	def use_cur(self,X_train,y_train):
		self.info.set_value('X_train',X_train)
		self.info.set_value('y_train',y_train)
		opcallback = op_pre_callback(use_pre=False)
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,callbacks=[opcallback],validation_data=(self.X_test,self.y_test))
		self.history.history['acc'].extend(history.history['acc'])
		self.history.history['val_acc'].extend(history.history['val_acc'])

	'''
	 - A simple transer learning function
	 - It will pop the last two layers, and add another classifier to the end of the model
	'''
	def nor_trans(self,X_train,y_train):
		self.model.pop()
		self.model.pop()
		
		for layer in self.model.layers:
			layer.trainable = False
		self.model.add(Dense(32,activation='relu'))
		self.model.add(Dense(self.num_classes,activation='softmax'))
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,validation_data=(self.X_test,self.y_test))
		self.history.history['acc'].extend(history.history['acc'])
		self.history.history['val_acc'].extend(history.history['val_acc'])


	def save(self,name):
		import json
		with open('./logs/{}.txt'.format(name),'w') as f:
			json.dump(self.history.history,f)
		self.model.save('./models/{}.h5'.format(name))

	def evaluate(self,X_test,y_test):
		
		score=self.model.evaluate(X_test,y_test,batch_size=128)
		print("Convolutional neural network test loss:",score[0])
		print('Convolutional neural network test accuracy:',score[1])

		return score[1]

	def get_history(self):
		return self.history


	'''
	Plot the history of accuracy
	'''
	def plot(self,name,model,shift=2):
		plt.subplot(211)
		plt.title('accuracy on current training data')
		for i in range(shift):
			plt.vlines(self.epoch*(i+1),0,1,color='r',linestyles='dashed')
		
		plt.plot(self.history.history['acc'],label='{}'.format(model))
		plt.ylabel('acc')
		plt.xlabel('training time')
		plt.legend(loc='upper right')
		plt.subplot(212)
		plt.title('validation accuracy on original data')
		plt.plot(self.history.history['val_acc'],label='{}'.format(model))
		plt.ylabel('acc')
		plt.xlabel('training time')
		for i in range(shift):
			plt.vlines(self.epoch*(i+1),0,1,color='r',linestyles='dashed')
		plt.legend(loc='upper right')
		plt.subplots_adjust(wspace=1,hspace=1)
		plt.savefig('./images/{}.png'.format(name))
		





'''
Kalman Callback.
	- use_pre: if is true, the model will calculate the gradients on X_tarin,y_train
		|- in 'kal_pre' algorithm, the X_train,y_train will be D1
		|- in 'kal_cur' altorithm, the X_train,y_train will be current training data
	- X_train, y_train will be
		|- 'kal', 'kal_cur' : The training data (D2,or D3)
		|- 'kal_pre': Always D1

'''
class op_pre_callback(Callback):
	"""docstring for op_batch_callback"""
	def __init__(self,info=None,use_pre=False):
		super(op_pre_callback, self).__init__()
		self.use_pre = use_pre
		self.info = info

		
		self.pre_x = self.info.get_value('data')[-1]
		self.pre_y = self.info.get_value('label')[-1]

		





	#Calculate the Kalman Gain based on current gradients and previous gradients
	def Kal_gain(self,cur_grad,pre_grad):
		res = []
		for i in range(len(pre_grad)):
			temp = np.absolute(pre_grad[i]) / ( np.absolute(cur_grad[i])  * self.FISHER[i] + np.absolute(pre_grad[i]) )
			temp[np.isnan(temp)] = 1
			res.append(temp)
		return res

	#Calculate the gradients of model on D1[ batch_0 ]
	#It will be used in 'kal'. 
	#	|- For other algorithm, the self.pre_g will be updated. 
	#	|- For 'kal', self.pre_g will not be updated
	def on_train_begin(self,logs={}):
		G = self.info.get_value('G')
		X = self.info.get_value('data')[-2]
		y = self.info.get_value('label')[-2]
		if G is None:
			print('G is None')
			self.pre_g = get_weight_grad(self.model,X,y)
		else:
			print('G is not None!!~')
			self.pre_g = G

		self.info.update_fisher(self.fisher(get_weight_grad(self.model,X[:512],y[:512])))
		self.FISHER = self.info.get_value('fisher')
		#self.pre_w = get_weights(self.model)


	def on_epoch_begin(self,epoch,logs={}):
		self.epoch = epoch
		
	#At the begining of each batch, get the weights
	#if use previous knowledge, update previous gradients(self.pre_g)
	def on_batch_begin(self,batch,logs={}):
		self.pre_w = get_weights(self.model)
		if self.use_pre:
			self.pre_g = get_weight_grad(self.model,self.pre_x[batch*128:(batch+1)*128],self.pre_y[batch*128:(batch+1)*128])


		

	#At the end of the batch:
	# |- Get the current weights
	# |- Calculate the gradients of model on X_train,y_train
	# |- Calculate the Kalman Gain
	# |- Kalman Filter to calculate the new weights and set_weights
	# |- Update error(self.pre_g): This will be used in 'kal' only. Other algorithm will update at the batch begin

	def fisher(self,g):
		fisher = []
		for i in g:
			temp = np.square(i)
			temp = temp / np.max(temp)
			fisher.append(temp/np.max(temp))
		return fisher
#

	def on_batch_end(self,batch,logs={}):
		
		
		self.cur_w = get_weights(self.model)

		self.cur_g = get_weight_grad(self.model,self.pre_x[batch*128:(batch+1)*128],self.pre_y[batch*128:(batch+1)*128])

		
		Kalman_gain = self.Kal_gain(self.cur_g,self.pre_g)
		new_w = []


		for P,Z,E,F in zip(self.pre_w,self.cur_w,Kalman_gain,self.FISHER):
			new_w.append(P + (Z-P) * E  )
			#new_w.append(Z + (1-E) * (P - Z) * F)

		
		self.model.set_weights(new_w)
		
		new_g = []
		for kal,g in zip(Kalman_gain,self.pre_g):
			new_g.append((1- kal) * g )

		self.pre_g = new_g

	def on_train_end(self,logs={}):
		self.info.set_value('G',self.pre_g)







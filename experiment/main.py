from utils import create_permuted_mnist_task
from model import cnn_model
import matplotlib.pyplot as plt


#igore the warning messages. Cause the kal drawback is slower than normal process, keras 
#will print some warning messages.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


#Plot the accuracy of test data
#Parameters:
# - name: the name of the model. It will be used in label
# - acc: list of accuracy
# - data_num: which data is plotted(D1,D2 or D3)
def acc_plot(name,acc,data_num):
	plt.figure(1)
	sub = '31'+str(data_num)
	plt.subplot(sub)
	plt.title('test accuracy on {}th dataset'.format(data_num))
	plt.plot(acc,label=name)
	plt.ylabel('acc')
	plt.xlabel('training time')
	for i in range(len(acc)-1):
		plt.vlines((i+1),0,1,color='r',linestyles='dashed')
	plt.legend(loc='upper right')
	plt.subplots_adjust(wspace=1,hspace=1)
	plt.savefig('./images/permuted.png'.format(name))


#Load the drift data
#task = create_permuted_mnist_task(3)
train_set,test_set,vali_set = create_disjoint_mnist_task()


#record the test accuracy. 
#Parameters:
# - model: the instance of model
# - acc_test_d1: record the accuracy of model on D1
# - acc_test_d2: record the accuracy of model on D2
# - acc_test_d3: record the accuracy of model on D3
def test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3):
	acc_test_d1.append(model.evaluate(test_set[0][0],test_set[0][1]))
	acc_test_d2.append(model.evaluate(test_set[1][0],test_set[1][1]))
	acc_test_d3.append(model.evaluate(test_set[2][0],test_set[2][1]))


# - Train the model.
# - The validation dataset is alwasy D1_test: model.val_data(X_test[:TEST_NUM],y_test[:TEST_NUM])
# - After the model is trained on each dataset, it will record the accuracy of model on test 
#	data of D1,D2 and D3 by using test_acc()
# - Return: this function will return the accuracy of D1_test,D2_test,D3_test after being trained
#	on each dataset.
global model
first_model = None
def train(name):

    acc_test_d1 = []
    acc_test_d2 = []
    acc_test_d3 = []
    t1_x,t1_y = train_set[0][0],train_set[0][1]
    t2_x,t2_y = train_set[1][0],train_set[1][1]
    t3_x,t3_y = train_set[2][0],train_set[2][1]
    global first_model
    if first_model is None:
        first_model = cnn_model()
        first_model.val_data(vali_set[0][0],vali_set[0][1])
        first_model.fit(t1_x,t1_y)

    from copy import deepcopy
    model = deepcopy(first_model)

    test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
	
    if name == 'kal':
        model.transfer(t2_x,t2_y)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
        model.transfer(t3_x,t3_y)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
    if name == 'kal_pre':
        model.use_pre(t2_x,t2_y)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
        model.use_pre(t3_x,t3_y)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
    if name == 'kal_cur':
        model.use_cur(t2_x,t2_y)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
        model.use_cur(t3_x,t3_y)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
    if name == 'nor':
        model.fit(t2_x,t2_y)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
        model.fit(t3_x,t3_y)
        test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)


    model.save(name)
    model.plot('res',name)
	
    return acc_test_d1,acc_test_d2,acc_test_d3

def save_acc(name,d):
	import json
	for i in range(3):
		path = './logs/permuted/acc{}/{}.txt'.format(str(i+1),name)
		with open(path,'w') as f:
			json.dump(d[i],f)

if __name__ == '__main__':
	#Train the model.
    print('--'*10,'kal','--'*10)
    kal_d1,kal_d2,kal_d3 = train('kal')
    save_acc('kal',[kal_d1,kal_d2,kal_d3])

    print('--' * 10, 'nor', '--' * 10)
    nor_d1, nor_d2, nor_d3 = train('nor')
    save_acc('nor', [nor_d1, nor_d2, nor_d3])
    


	#print('--'*10,'kal_cur','--'*10)
	#kal_cur_d1, kal_cur_d2, kal_cur_d3 = train('kal_cur')
	#save_acc('kal_cur',[kal_cur_d1, kal_cur_d2, kal_cur_d3])


	#print('--'*10,'kal_pre','--'*10)
	#kal_pre_d1, kal_pre_d2, kal_pre_d3 = train('kal_pre')
	#save_acc('kal_pre',[kal_pre_d1, kal_pre_d2, kal_pre_d3])

	#Plot the accuracy on test data.

    acc_plot('nor',nor_d1,1)
    acc_plot('nor',nor_d2,2)
    acc_plot('nor',nor_d3,3)
    acc_plot('kal',kal_d1,1)
    acc_plot('kal',kal_d2,2)
    acc_plot('kal',kal_d3,3)
	#acc_plot('kal_pre',kal_pre_d1,1)
	#acc_plot('kal_pre',kal_pre_d2,2)
	#acc_plot('kal_pre',kal_pre_d3,3)
	#acc_plot('kal_cur',kal_cur_d1,1)
	#acc_plot('kal_cur',kal_cur_d2,2)
	#acc_plot('kal_cur',kal_cur_d3,3)


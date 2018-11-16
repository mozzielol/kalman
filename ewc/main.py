from model import Model
from utils import create_permuted_mnist_task


task_list = create_permuted_mnist_task(10)


model = Model()
model.val_data(task_list[0].validation.images,task_list[0].validation.labels)
model.fit(task_list[0].train.images,task_list[0].train.labels)
model.compute_fisher(task_list[0].validation.images,task_list[0].validation.labels)
model.ewc_fisher(task_list[0].validation.images,task_list[0].validation.labels)

for t in task_list[1:]:
	#model.star()
	model.fit(t.train.images,t.train.labels)
	model.compute_fisher(t.validation.images,t.validation.labels)
	model.ewc_fisher(t.validation.images,t.validation.labels)
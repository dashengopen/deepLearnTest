import numpy as np
from mnist import load_mnist
from simple_convnet import *
from util import *
#import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True, one_hot_label=True)

train_size = x_train.shape[0]

network = SimpleConvNet()

train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_num = 600
batch_size = 100
learning_rate = 0.01

optimizer = SGD(learning_rate)

#max_epochs = 20
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)
    #grad = gradient(x_batch, t_batch)

    optimizer.update(network.params, grad)

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train, batch_size)
        test_acc = network.accuracy(x_test, t_test, batch_size)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        
# 保存参数
network.save_params("params.pkl")
print("saved net params!")

'''
# 绘制图形
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()


'''

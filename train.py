import torch
import torchvision
import torchvision.transforms as transforms
from model import CNN_NET
import torch.optim as optim
from utils import *
import torch.nn as nn

# 定义超参数
BATCH_SIZE = 64
EPOCH = 2

# torchvision模块载入CIFAR10数据集，并且通过transform归一化到[0,1]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='data/', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='data/', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print(len(testloader))  # 157
# exit()
net = CNN_NET()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 优化器
loss_func = torch.nn.CrossEntropyLoss()  # 预测值和真实值的误差计算公式 (交叉熵)

Loss_list = []
Accuracy_list = []

for epoch in range(EPOCH):
    running_loss = 0.0
    for step, (b_x, b_y) in enumerate(trainloader):
        outputs = net(b_x)  # 喂给 net 训练数据 x, 输出预测值
        loss = loss_func(outputs, b_y)  # 计算两者的误差
        accuracy = torch.max(outputs, 1)[1].numpy() == b_y.numpy()
        # print(type(accuracy))
        # exec()

        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值

        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        # 打印状态信息
        running_loss += loss.item()
        if step % 50 == 0:  # 每2000批次打印一次
            print('[epoch=%d, step=%5d] loss: %.3f acc: %.3f' %
                  (epoch + 1, step + 1, running_loss / 50, accuracy.mean()))
            Loss_list.append(loss.item())
            Accuracy_list.append(accuracy.mean())
            running_loss = 0.0
        if step == 100:
            break

print('Finished Training')
plot_loss(Loss_list)
plot_acc(Accuracy_list)
# plot_roc()

########测试集精度#######
correct = 0
total = 0
# predicted = []
# labels = []
# outputs_list = np.array([])
with torch.no_grad():
    # 不计算梯度，节省时间
    for i, (images, labels) in enumerate(testloader):
        outputs = net(images)
        if i == 0:
            outputs_list = outputs.numpy().copy()
            labels_list = labels.numpy().copy()
        else:
            outputs_list = np.concatenate((outputs_list, outputs.numpy()), axis=0)
            labels_list = np.concatenate((labels_list, labels.numpy()), axis=0)
        # print(outputs.numpy().shape)
        # exit()
        numbers, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    plot_roc(y_true=labels_list, y_scores=outputs_list)

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

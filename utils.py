import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils.multiclass import type_of_target
import numpy as np
from sklearn.preprocessing import label_binarize


def plot_loss(loss_list):
    plt.figure('PyTorch_CNN_Loss')
    plt.plot(loss_list, label='Loss')
    plt.legend()
    plt.show()


def plot_acc(acc_list):
    plt.figure('PyTorch_CNN_Acc')
    plt.plot(acc_list, label='Acc')
    plt.legend()
    plt.show()


def plot_roc(y_true, y_scores):
    # y_true = y_true.numpy()
    # y_scores = y_scores.numpy()

    y_one_hot = label_binarize(y_true, np.arange(10))

    # print(type(predicted))
    # y_scores = np.amax(y_true, axis=1)
    # print(type_of_target((y_true)))
    # print(type_of_target(y_scores))
    # exit()

    print(y_true.shape)  # (157,)
    print(y_scores.shape)  # (157, 10)
    print(y_one_hot.shape)  # (157, 7)
    # exit()

    fpr, tpr, threshold = roc_curve(y_one_hot.ravel(), y_scores.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.show()

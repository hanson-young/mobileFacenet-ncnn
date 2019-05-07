#encoding=utf-8
from sklearn.metrics import roc_curve, auc
# from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import KFold
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 08:57:13 2015
@author: shifeng
"""
print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
#int8 [lfw]Accuracy: 0.99032+-0.00643
#fp32 [lfw]Accuracy: 0.99123+-0.00576
with open("/home/hanson/Documents/heils-git/code/mobileFacenet-ncnn/fp32.txt", 'r') as f:
    lines = f.readlines()
    print(lines)

result = []
for idx, line in enumerate(lines):
    enum = line.strip().split('\t')
    result.append([int(enum[0]),float(enum[1])])
print(result)
result = np.array(result)

result[np.where(result[:,0] < 0),0] = 0
# print(result)
#
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100000)
all_tpr = []

# print((result[:,0], result[:,1]))
fpr, tpr, thresholds = roc_curve(result[:,0], result[:,1])
mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
mean_tpr[0] = 0.0  # 初始处为0
roc_auc = auc(fpr, tpr)
# 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (0, roc_auc))

# 画对角线
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= 1  # 在mean_fpr100个点，每个点处插值插值多次取平均
mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）
mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
# 画平均ROC曲线
# print mean_fpr,len(mean_fpr)
print (mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


thresholds = np.arange(0, 1.0, 0.001)

#求解当前阈值时的准确率
def eval_acc(threshold, result):
    y_true = []
    y_predict = []
    for r in result:
        same = 1 if float(r[1]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(r[0]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0*np.count_nonzero(y_true==y_predict)/len(y_true)
    return accuracy

#eval_acc和find_best_threshold共同工作，来求试图找到最佳阈值，
#
def find_best_threshold(thresholds, predicts):
    #threshould 阈值
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold,best_acc

print(find_best_threshold(thresholds,result))


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.greater(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

acc_train = np.zeros((len(thresholds)))
for threshold_idx, threshold in enumerate(thresholds):
    _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, result[:,1], result[:,0])
best_threshold_index = np.argmax(acc_train)

print(thresholds[best_threshold_index],acc_train[best_threshold_index])


def find_best_acc(labels, scores):
    best_thresh = None
    best_acc = 0
    for score in scores:
        preds = np.greater_equal(scores, score).astype(np.int32)
        acc = accuracy_score(labels, preds, normalize=True)
        if acc > best_acc:
            best_thresh = score
            best_acc = acc
    return best_acc, best_thresh

def cal_acc(fold_num=10):

    scores = result[:,1]
    labels = result[:,0]
    indices = np.arange(scores.shape[0])
    k_fold = KFold(n_splits=fold_num)

    acc_list = []
    for train_set, test_set in k_fold.split(indices):
        _, best_thresh = find_best_acc(labels[train_set], scores[train_set])
        test_preds = np.greater_equal(scores[test_set], best_thresh).astype(np.int32)
        acc_list.append(accuracy_score(labels[test_set], test_preds, normalize=True))
    return acc_list

acc_list = cal_acc()
print(acc_list)
acc, std = np.mean(acc_list), np.std(acc_list)

print('[%s]Accuracy: %1.5f+-%1.5f' % ("lfw", acc, std))
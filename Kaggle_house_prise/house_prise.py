import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import pandas as pd
import sys


def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    # plt.show()


torch.set_default_tensor_type(torch.FloatTensor)

# 读取数据集
train_data = pd.read_csv('../Kaggle_house_prise/train.csv')
test_data = pd.read_csv('../Kaggle_house_prise/test.csv')

# 数据shape
# print(train_data.shape)
# print(test_data.shape)

#print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# 将所有的训练数据和测试数据的79个特征按样本连结
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features.shape)

# 数据预处理
# 进行标准化
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# 减去均值再除以标准差
# 用0替换缺失值，因为标准化之后均值为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 将离散数值转换为指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# print(all_features.shape)

# 通过values属性得到Numpy格式的数据，并转成Tensor
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(
    train_data.SalePrice.values, dtype=torch.float).view(-1, 1)
# print(train_features[1])


# 线性回归和平方损失函数
loss = torch.nn.MSELoss()

# 根据输入数据确定网络大小


def get_net(feature_num):
    # 第一个参数是每个输入样本的大小，第二个参数是每个输出样本的大小，第三的参数bias默认为True
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.max(net(features), torch.tensor(1.0))  # 将小于1的值置1
        rmse = torch.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels, num_rpochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    print(train_ls)
    return train_ls, test_ls


# K折交叉验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
    return X_train, y_train, X_valid, y_valid

# 训练K次并返回训练和验证的平均误差


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    delta_train_valid = 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(
            net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                     range(1, num_epochs + 1), valid_ls,
                     ['train', 'valid'])
        print('fold %d, train rmse %f, valid rmse %f' %
              (i, train_ls[-2], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k,


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels,
                          num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse %f, avg valid rmse %f' %
      (k, train_l, valid_l))


# 预测函数
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    semilogy(range(1, num_epochs + 1), train_ls, 'epoch', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)
    return train_ls[-1]


train_rmse = train_and_pred(train_features, test_features, train_labels,
                            test_data, num_epochs, lr, weight_decay, batch_size)

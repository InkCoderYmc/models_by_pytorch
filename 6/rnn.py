#RNN实现
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import random
import sys
import zipfile

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def sgd(params, lr, batch_size):
    # 为了和原书保持一致，这里除以了batch_size，但是应该是不用除的，因为一般用PyTorch计算loss时就默认已经
    # 沿batch维求了平均了。
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

def load_data_jay_lyrics():
    """加载周杰伦歌词数据集"""
    with zipfile.ZipFile('../Datasets/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size

#随机采样
def data_iter_random(corpus_chars, batch_size, num_steps, device=None):
    num_examples = (len(corpus_chars) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    #返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_chars[pos: pos + num_steps]

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

#相邻采样
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len //batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()

#为了将词表示称向量输入到神经网络，简单办法是使用one-hot向量
def one_hot(x, n_class, dtype=torch.float32):
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res

x = torch.tensor([0, 2])
#print(one_hot(x, vocab_size))

#小批量转换
def to_onehot(X, n_class):
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

X = torch.arange(10).view(2, 5)
inputs = to_onehot(X, vocab_size)
#print(len(inputs), inputs[0].shape)
'''
5 torch.Size([2, 1027])
'''

#初始化模型参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
print('will use', device)

def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 0.01, size=shape), device=device, dtype=torch.float32)
        return torch.nn.Parameter(ts, requires_grad=True)

    #隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = torch.nn.Parameter(torch.zeros(num_hiddens, device=device, requires_grad=True))

    #输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs, device=device, requires_grad=True))

    return nn.ParameterList([W_xh, W_hh, b_h, W_hq, b_q])

#定义函数返回初始化的隐藏状态
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

#定义一个时间步中计算隐藏状态和输出
def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.matmul(X, W_xh) + torch.matmul(H, W_hh) + b_h)
        Y = torch.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

#简单测试
state = init_rnn_state(X.shape[0], num_hiddens, device)
inputs = to_onehot(X.to(device), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
#print(len(outputs), outputs[0].shape, state_new[0].shape)
'''
5 torch.Size([2, 1027]) torch.Size([2, 256])
'''

#RNN预测函数
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        #将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        #计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)

        #下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

#测试使用‘分开’为前缀创作长度为10个字符
print(predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx))


#裁剪梯度，防止出现梯度爆炸
def grad_clipping(params, theta, device):
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)


#困惑度 交叉熵损失函数做指数运算后得到的值
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens, 
                        vocab_size, device, corpus_indices, idx_to_char, 
                        char_to_idx, is_random_iter, num_epochs, num_steps, 
                        lr, clipping_theta, batch_size, pred_period, 
                        pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive

    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter: #使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)

        for X, Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:
                #使用detach函数从计算图分离隐藏状态，为了使模型参数的梯度计算只以来一次迭代读取的小批量序列，防止梯度计算开销太大
                for s in state:
                    s.detach_()

            inputs = to_onehot(X, vocab_size)
            #outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            outputs = torch.cat(outputs, dim=0)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(outputs, y.long())

            #梯度清零
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)
            sgd(params, lr, 1)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' %(epoch + 1, math.exp(l_sum / n), time.time() - start))

            for prefixe in prefixes:
                print(' -', predict_rnn(prefixe, pred_len, rnn, params, init_rnn_state, num_hiddens, vocab_size, device, idx_to_char, char_to_idx))

num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                    vocab_size, device, corpus_indices, idx_to_char,
                    char_to_idx, True, num_epochs, num_steps, lr,
                    clipping_theta, batch_size, pred_period, pred_len,
                    prefixes)
#预处理语言模型数据集，将其转换成字符级循环神经网络所需要的输入格式
import torch
import random
import zipfile

with zipfile.ZipFile('../Datasets/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
#print(corpus_chars[:40])

#为了方便打印，将黄行夫替换成空格，使用前10000字符来训练模型
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]
#print(corpus_chars)


#建立字符索引
#将每个字符映射成一个从0开始的连续整数。
#将数据集里所有不同字符取出来，然后逐一映射到索引来构造词典。
idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
#print(vocab_size)
#1027

#将训练数据集中每个字符转化为索引
corpus_chars = [char_to_idx[char] for char in corpus_chars]
sample = corpus_chars[:20]

#print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
#print('indices:', sample)

#对时序数据进行采样
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

#随机采样测试
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X:', X, '\nY:', Y, '\n')

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

for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')
'''
X:  tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],
        [15., 16., 17., 18., 19., 20.]]) 
Y: tensor([[ 1.,  2.,  3.,  4.,  5.,  6.],
        [16., 17., 18., 19., 20., 21.]]) 

X:  tensor([[ 6.,  7.,  8.,  9., 10., 11.],
        [21., 22., 23., 24., 25., 26.]]) 
Y: tensor([[ 7.,  8.,  9., 10., 11., 12.],
        [22., 23., 24., 25., 26., 27.]]) 
'''
import keras
import pandas as pd
import pandas_datareader.data as data
from datetime import datetime
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim

class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(D_in, H, batch_first=True,
                            num_layers=1)
        self.linear = nn.Linear(H, D_out)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = self.linear(output[:, -1, :])
        return output

D_in = 1  # 入力次元: 1
H = 200  # 隠れ層次元: 200
D_out = 1  # 出力次元: 1
epoch = 100  # 学習回数: 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net(D_in, H, D_out).to(device)
print("Device: {}".format(device))

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

train_loss_list = []  # 学習損失
test_loss_list = []  # 評価損失

# 学習（エポック）の実行
for i in range(epoch):
    # エポックの進行状況を表示
    print('---------------------------------------------')
    print("Epoch: {}/{}".format(i+1, epoch))

    # 損失の初期化
    train_loss = 0  # 学習損失
    test_loss = 0  # 評価損失

    # ---------学習パート--------- #
    # ニューラルネットワークを学習モードに設定
    net.train()
    # ミニバッチごとにデータをロードし学習
    for data, label in train_batch:
        # GPUにTensorを転送
        data = data.to(device)
        label = label.to(device)

        # 勾配を初期化
        optimizer.zero_grad()
        # データを入力して予測値を計算（順伝播）
        y_pred = net(data)
        # 損失（誤差）を計算
        loss = criterion(y_pred, label)
        # 勾配の計算（逆伝搬）
        loss.backward()
        # パラメータ（重み）の更新
        optimizer.step()
        # ミニバッチごとの損失を蓄積
        train_loss += loss.item()

    # ミニバッチの平均の損失を計算
    batch_train_loss = train_loss / len(train_batch)
    # ---------学習パートはここまで--------- #



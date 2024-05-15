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

# ---------評価パート--------- #
    # ニューラルネットワークを評価モードに設定
    net.eval()
    # 評価時の計算で自動微分機能をオフにする
    with torch.no_grad():
        for data, label in test_batch:
            # GPUにTensorを転送
            data = data.to(device)
            label = label.to(device)
            # データを入力して予測値を計算（順伝播）
            y_pred = net(data)
            # 損失（誤差）を計算
            loss = criterion(y_pred, label)
            # ミニバッチごとの損失を蓄積
            test_loss += loss.item()

    # ミニバッチの平均の損失を計算
    batch_test_loss = test_loss / len(test_batch)
    # ---------評価パートはここまで--------- #

    # エポックごとに損失を表示
    print("Train_Loss: {:.2E} Test_Loss: {:.2E}".format(
        batch_train_loss, batch_test_loss))
    # 損失をリスト化して保存
    train_loss_list.append(batch_train_loss)
    test_loss_list.append(batch_test_loss)

# 損失
plt.figure()
plt.title('Train and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, epoch+1), train_loss_list, color='blue',
         linestyle='-', label='Train_Loss')
plt.plot(range(1, epoch+1), test_loss_list, color='red',
         linestyle='--', label='Test_Loss')
plt.legend()  # 凡例
plt.show()  # 表示

net.eval()
# 推定時の計算で自動微分機能をオフにする
with torch.no_grad():
    # 初期化
    pred_ma = []
    true_ma = []
    for data, label in test_batch:
        # GPUにTensorを転送
        data = data.to(device)
        label = label.to(device)
        # 予測値を計算：順伝播
        y_pred = net(data)
        pred_ma.append(y_pred.view(-1).tolist())
        true_ma.append(label.view(-1).tolist())

pred_ma = np.array([elem for lst in pred_ma for elem in lst])
true_ma = np.array([elem for lst in true_ma for elem in lst])

#dst = np.array([255 if elm == max_index else 0 for elm in dst])

pred_ma = pred_ma.reshape(-1, 1)
true_ma = true_ma.reshape(-1, 1)

pred_ma = scaler.inverse_transform(pred_ma)
true_ma = scaler.inverse_transform(true_ma)

mae = mean_absolute_error(true_ma, pred_ma)
print("MAE: {:.3f}".format(mae))

date = df['Date'][-1*test_len:]  # テストデータの日付
test_close = df['Adj Close'][-1*test_len:].values.reshape(-1)  # テストデータの終値
plt.figure()
plt.title('YHOO Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.plot(date, test_close, color='black',
         linestyle='-', label='close')
plt.plot(date, true_ma, color='dodgerblue',
         linestyle='--', label='true_25MA')
plt.plot(date, pred_ma, color='red',
         linestyle=':', label='predicted_25MA')
plt.legend()  # 凡例
plt.xticks(rotation=30)
plt.show()

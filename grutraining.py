import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import KFold

mpl.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler

data_normalized = pd.read_excel("model8-1pearson-jianmo.xlsx")
print(data_normalized)
data_normalized = data_normalized.values
data_normalized = data_normalized[0:, 1:].astype(float)
kf = KFold(n_splits=6)


kf.get_n_splits(data_normalized)
print(kf)
for train_index, test_index in kf.split(data_normalized):
    print("Train", train_index, "Test", test_index)
    print(data_normalized[train_index])
    data_train = data_normalized[train_index]
    data_test = data_normalized[test_index]
    data_x_train = data_train[0:, 1:]
    data_y_train = data_train[0:, 0]
    data_x_test = data_test[0:, 1:]
    data_y_test = data_test[0:, 0]
    print("data_x_train:", data_x_train)
    print("data_y_train:", data_y_train)
    print("data_x_test:", data_x_test)
    print("data_y_test:", data_y_test)
    data_x_train = data_x_train.flatten()
    data_x_test = data_x_test.flatten()
    data_y_test = torch.FloatTensor(data_y_test).view(-1)
    data_y_train = torch.FloatTensor(data_y_train).view(-1)
    data_x_test = torch.FloatTensor(data_x_test).view(-1)
    data_x_train = torch.FloatTensor(data_x_train).view(-1)
    train_window = len(data_y_train)
    test_window = len(data_y_test)


    def create_inout_sequences(x_train, y_train, tw):
        inout_seq = []
        L = len(x_train)
        J = int(L / tw)
        for i in range(int(tw)):
            train_seq = x_train[J * i:(i + 1) * J]
            train_label = y_train[i]
            inout_seq.append((train_seq, train_label))
        return inout_seq


    train_inout_seq = create_inout_sequences(data_x_train, data_y_train, train_window)
    test_inout_seq = create_inout_sequences(data_x_test, data_y_test, test_window)
    print(train_inout_seq)
    print(test_inout_seq)


    class StockAttention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros((112, 112)))
            self.bias = nn.Parameter(torch.zeros(112))
            self.fc_combine = nn.Linear(dim * 2, dim)

        def forward(self, x):
            # x: (b, st, t, f)
            attn = F.softmax(self.weight + self.bias[None, :], dim=-1)
            y = torch.einsum('b i ..., j i -> b j ...', x, attn)
            x = torch.cat((x, y), -1)
            x = self.fc_combine(x)
            return x


    class GRU(nn.Module):
        def __init__(self, input_size=1, dim=32, conv1_kernel=3, hidden_layer_size=300):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size
            self.conv1 = nn.Conv1d(input_size, dim * 10, conv1_kernel, padding=28)
            self.conv2 = nn.Conv1d(dim * 10, 16, 1)
            self.conv3 = nn.Conv1d(16, dim * 2, 1)
            self.conv4 = nn.Conv1d(dim * 2, 1, 1)
            self.norm1 = nn.LayerNorm([dim * 10, 64])  # ([112, dim])
            self.norm2 = nn.LayerNorm([16, 64])
            self.norm3 = nn.LayerNorm([dim * 2, 64])
            self.norm4 = nn.LayerNorm([1, 64])
            self.rnn = nn.GRU(64, 16, 2)
            self.fc_out = nn.Linear(16, 1)

        def forward(self, input_seq):
            x = self.conv1(input_seq)
            x = F.gelu(x)
            x = self.norm1(x)
            x = self.conv2(x)
            x = F.gelu(x)
            x = self.norm2(x)
            x = self.conv3(x)
            x = F.gelu(x)
            x = self.norm3(x)
            x = self.conv4(x)
            x = F.gelu(x)
            x = self.norm4(x)
            x, hn = self.rnn(x)
            predictions = self.fc_out(x)

            return predictions


    model = GRU()
    # model.load_state_dict(torch.load('model_params_test5_8.pkl'))


    def loss_function(input_pred, input_labels):
        losst = abs(input_pred - input_labels)
        return losst


    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
    print(model)
    epochs = 2000

    for i in range(epochs):
        for seq, labels in train_inout_seq:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            seq = torch.unsqueeze(seq, 0)

            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    model.eval()
    predictive_y_for_testing = []

    for seq, labels in test_inout_seq:
        y_aim = []
        seq = torch.unsqueeze(seq, 0)
        predictive_y_for_testing.append(model(seq).item())
        y_aim.append(labels.tolist())
        predictive_y_for_testing2 = np.array(predictive_y_for_testing)
        y_aim2 = np.array(y_aim)
        loss = abs(predictive_y_for_testing2 - y_aim2) / y_aim

    print(predictive_y_for_testing)
    print(loss)

    torch.save(model.state_dict(), "model_params_test5_8_2.pkl")


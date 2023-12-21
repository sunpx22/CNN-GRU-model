import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch.nn.functional as F

data_normalized = pd.read_excel("model1-Qxz50%.xlsx")
print(data_normalized)
data_normalized = data_normalized.values
data_normalized = data_normalized[0:, 1:].astype(float)
data_x = data_normalized[0:, 0:10]
data_y = data_normalized[0:, -1]
data_x = data_x.flatten()
data_y = torch.FloatTensor(data_y).view(-1)
data_x = torch.FloatTensor(data_x).view(-1)
t_window = len(data_y)
def create_inout_sequences(x_train, y_train, tw):
    inout_seq = []
    L = len(x_train)
    J = int(L / tw)
    for i in range(int(tw)):
        train_seq = x_train[J * i:(i + 1) * J]
        train_label = y_train[i]
        inout_seq.append((train_seq, train_label))
    return inout_seq

test_inout_seq = create_inout_sequences(data_x, data_y, t_window)
print(test_inout_seq)


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
model.load_state_dict(torch.load('model_params_test6_1_7.pkl'))
def loss_function(input_pred, input_labels):
    losst = abs(input_pred - input_labels) / input_labels
    return losst
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

model.eval()
predictive_y_for_testing = []
y_aim = []
for seq, labels in test_inout_seq:

    seq = torch.unsqueeze(seq, 0)
    predictive_y_for_testing.append(model(seq).item())
    y_aim.append(labels.tolist())
    predictive_y_for_testing2 = np.array(predictive_y_for_testing)
    y_aim2 = np.array(y_aim)

loss = abs(predictive_y_for_testing2 - y_aim2) / y_aim
predictive_y_for_testing.append(sum(predictive_y_for_testing))
print(predictive_y_for_testing)
print(loss)
dataframe = pd.DataFrame(predictive_y_for_testing)
dataframe.to_excel('test1.xlsx')
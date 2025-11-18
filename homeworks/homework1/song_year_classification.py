import torch
import torch.nn as nn
import torch.optim as optim
import unittest
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

path_x_train = "homework_spbu_dl_2025/homeworks/data/YearPredictionMSD/train_x.csv"
path_y_train = "homework_spbu_dl_2025/homeworks/data/YearPredictionMSD/train_y.csv"
path_x_test = "homework_spbu_dl_2025/homeworks/data/YearPredictionMSD/test_x.csv"

X_train = torch.tensor(pd.read_csv(path_x_train).values, dtype=torch.float32)
y_train = torch.tensor(pd.read_csv(path_y_train)['year'].values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(pd.read_csv(path_x_test).values, dtype=torch.float32)
print(X_train.shape, y_train.shape, X_test.shape)

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NAdam(Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, schedule_decay=0.004, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, schedule_decay=schedule_decay, weight_decay=weight_decay)
        super(NAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NAdam does not support sparse gradients')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['m_schedule'] = 1.0
                m, v = state['m'], state['v']
                beta1, beta2 = group['betas']
                state['step'] += 1
                t = state['step']
                m_schedule = state['m_schedule']
                m_schedule_new = m_schedule * beta1
                m_schedule_next = m_schedule_new * (1 - 0.5 * (0.96 ** (t * group['schedule_decay'])))
                m_schedule_next_next = m_schedule_next * (1 - 0.5 * (0.96 ** ((t+1) * group['schedule_decay'])))
                state['m_schedule'] = m_schedule_next
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                m_t = beta1 * m + (1 - beta1) * grad
                v_t = beta2 * v + (1 - beta2) * grad * grad
                m_hat = m_t / (1 - m_schedule_next_next)
                grad_term = (beta1 * m_hat + (1 - beta1) * grad / (1 - m_schedule_next_next)) / (torch.sqrt(v_t / (1 - beta2 ** t)) + group['eps'])
                p.data.add_(grad_term, alpha=-group['lr'])
                state['m'], state['v'] = m_t, v_t
        return loss

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.numpy())
X_test_scaled = scaler.transform(X_test.numpy())
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = y_train.clone().detach()
if y_train_tensor.dtype != torch.float32:
    y_train_tensor = y_train_tensor.float()
if y_train_tensor.dim() == 1:
    y_train_tensor = y_train_tensor.unsqueeze(1)
X_train_scaled = X_train_scaled.to(device)
X_test_scaled = X_test_scaled.to(device)
y_train_tensor = y_train_tensor.to(device)

class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Dropout(0.2),
            nn.Sigmoid(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.Sigmoid(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.Sigmoid(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

model = NeuralNet(X_train.shape[1])
model = model.to(device)
criterion = nn.MSELoss()
optimizer = NAdam(model.parameters(), lr=0.0002)

dataset = TensorDataset(X_train_scaled, y_train_tensor)
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

y_train_np = y_train_tensor.cpu().numpy().ravel()
unique, counts = np.unique(y_train_np, return_counts=True)
use_sampler = False
sampler = None
if unique.shape[0] <= 2 and np.issubdtype(y_train_np.dtype, np.integer) or unique.shape[0] <= 10:
    sample_counts = {}
    for u, c in zip(unique, counts):
        sample_counts[u] = c
    weights = np.array([1.0 / sample_counts[float(y)] for y in y_train_np], dtype=np.float32)
    weights_tensor = torch.from_numpy(weights)
    indices = train_dataset.indices if hasattr(train_dataset, 'indices') else list(range(train_size))
    weights_for_train = weights_tensor[indices]
    sampler = WeightedRandomSampler(weights_for_train, num_samples=len(weights_for_train), replacement=True)
    use_sampler = True

batch_size = 128
if use_sampler:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.best_state = None
    def step(self, loss, model):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

early_stopper = EarlyStopping(patience=3, min_delta=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

epochs = 500
train_losses = []
val_losses = []
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss = epoch_loss / train_size
    train_losses.append(epoch_loss)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for vX, vy in val_loader:
            vout = model(vX)
            vloss = criterion(vout, vy)
            val_loss += vloss.item() * vX.size(0)
    val_loss = val_loss / val_size
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/(X_train_scaled.size()[0]//batch_size):.4f}")
    scheduler.step(val_loss)
    if early_stopper.step(val_loss, model):
        break

if early_stopper.best_state is not None:
    model.load_state_dict(early_stopper.best_state)

model.eval()
with torch.no_grad():
    y_pred = model(X_test_scaled)
    pred_year = torch.round(y_pred).squeeze().cpu().numpy().astype(int)
    
df = pd.DataFrame({'year': pred_year})
df.to_csv('pred_year.csv', index=False)

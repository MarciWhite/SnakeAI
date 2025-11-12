import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

    def save(self, file_name="model.pth"):
        model_folder_path = "./model/"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_path)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)

        if state.ndim == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0)
            action = action.unsqueeze(0)
            done = (done,)

        prediction = self.model(state)
        target = prediction.clone().detach()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]).detach())

            target[i][torch.argmax(action[i]).item()] = Q_new


        # Q_new = R + gamma * max(next_predicted Q)
        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()

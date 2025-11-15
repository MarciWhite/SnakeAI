import datetime
import json

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
        #self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        #x = F.relu(self.l2(x))
        x = self.l2(x)
        return x


    def save(self, score,mean_score, model_settings=None,game_settings=None, folder="./model/"):
        """
        Save the model weights and update metadata JSON.
        Automatically timestamps the file and tracks multiple models.
        """
        os.makedirs(folder, exist_ok=True)

        # Timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"model_{timestamp}.pth"
        file_path = os.path.join(folder, file_name)

        # Save model weights
        torch.save(self.state_dict(), file_path)

        # Load existing metadata or create new
        metadata_file = os.path.join(folder, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        else:
            metadata = { "highscore": 0,"models": []}

        # Update highscore if needed
        if score > metadata.get("highscore", 0):
            metadata["highscore"] = score

        # Add this model to list
        metadata["models"].append({
            "file": file_name,
            "file_path": file_path,
            "score": score,
            "mean_score": mean_score,
            "timestamp": timestamp,
            "model_settings": {
                "learning_rate": 0.001,
                "gamma": 0.8,
                "epsilon_start": 1.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 0.01,
                "batch_size": 1000,
                "max_memory": 100_000,
                "hidden_size": 256,
                "num_game": 0
            } if model_settings is None else model_settings,
            "game_settings": {
                "speed": 40,
                "width": 640,
                "height": 480,
                "block_size": 2,
                "snake_start_size": 3,
                "hard_boundary" : True
            } if game_settings is None else game_settings

        })

        # Save JSON
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Model saved: {file_name} | Score: {score}")

    def load(self, file_name, device):
        folder = "./model/"

        try:
            self.load_state_dict(torch.load(os.path.join(folder, file_name), map_location=device))
        except Exception as e:
            print("Error loading model:", e)
            return

        print(f"Successfully loaded {file_name} using {device}")

class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = device
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(self.device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.long).to(self.device)

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

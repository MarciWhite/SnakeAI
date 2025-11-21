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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
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
                "current_epsilon": 0.9,
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
        self.target_model = Linear_QNet(self.model.input_size, self.model.hidden_size,self.model.output_size,)

        # Copy the weights to synchronize them at the start
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()


        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.device = device

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float).to(self.device)

        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device).unsqueeze(1)

        dones = torch.tensor(np.array(dones), dtype=torch.bool).to(self.device)

        # Get all Q-values from the policy model
        pred = self.model(states)  # Shape: [BATCH_SIZE, 3]

        # Use .gather() to select the specific Q-value for the action we took
        pred_actions = pred.gather(1, actions).squeeze(1)  # Shape: [BATCH_SIZE]

        # Do ONE batch forward pass on the target network
        Q_next = self.target_model(next_states).detach()

        # Find the max Q-value for the next state
        max_Q_next = torch.max(Q_next, dim=1)[0]  # Shape: [BATCH_SIZE]

        # Y_t = R if done
        # Y_t = R + gamma * max_Q_next if not done
        Y_target = rewards + self.gamma * max_Q_next * (~dones)  # (~dones) acts as a (1 - done) mask

        # Calculate loss and backpropagate
        self.optimizer.zero_grad()
        loss = self.criterion(pred_actions, Y_target)
        loss.backward()
        self.optimizer.step()


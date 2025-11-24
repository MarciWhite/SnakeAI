import json
import os
from typing import Dict

import torch
import random
import numpy as np

from model import Linear_QNet, QTrainer
from game import SnakeGame, Direction, Point, BLOCK_SIZE
from collections import deque
import math

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
sign = lambda x: math.copysign(1, x)



plt.ion()
fig, ax = plt.subplots()



def plot(scores, mean_scores):
    ax.clear()
    ax.set_title('Training...')
    ax.set_xlabel('Number of Games')
    ax.set_ylabel('Score')

    ax.plot(scores, label='Score')
    ax.plot(mean_scores, label='Mean Score')

    ax.legend()
    ax.text(len(scores)-1, scores[-1], str(scores[-1]))
    ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    plt.tight_layout()
    plt.pause(0.1)  # brief pause to update the figure


class Agent():
    def __init__(self, model_metadata=None,settings=None):

        self.n_steps = 0

        self.max_memory = settings.get("max_memory", 100_000)
        self.batch_size = settings.get("batch_size", BATCH_SIZE)
        self.learning_rate = settings.get("learning_rate", 0.001)

        self.epsilon_decay = settings.get("epsilon_decay", 0.00004)
        self.epsilon_min = settings.get("epsilon_min", 0.0)
        self.epsilon_start = settings.get("epsilon_start", 0.95)

        self.max_idle_steps = 800
        self.steps_left = self.max_idle_steps

        self.gamma = settings.get("gamma", 0.8)
        self.hidden_size = settings.get("hidden_size", 256)

        # memory deque must use max_memory
        self.memory = deque(maxlen=self.max_memory)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize model and trainer
        self.model = Linear_QNet(12, self.hidden_size, 3).to(device)


        if model_metadata is None:
            self.n_games = 0
            self.epsilon = self.epsilon_start
        else:
            self.n_games = model_metadata["model_settings"].get("num_game", 0)
            self.epsilon = model_metadata["model_settings"].get("current_epsilon", 0)

            self.model.load(model_metadata["file"], device)
            print(f"Model highscore: {model_metadata['score']}\nNumber of games: {self.n_games}")

        self.trainer = QTrainer(self.model, lr=self.learning_rate, gamma=self.gamma, device=device)

    def export_settings(self) -> Dict:
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "current_epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "max_memory": self.max_memory,
            "hidden_size": self.hidden_size,
            "num_game": self.n_games
        }

    def export_stats(self) -> dict:
        """
        Returns a dict with runtime info about the agent:
        - epsilon
        - memory usage
        - number of games
        - learning parameters
        - estimated model accuracy
        """
        stats = {
            "memory_filled": len(self.memory),
            "max_memory": self.memory.maxlen,
            "num_games": self.n_games,
            "current_epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "hidden_size": self.hidden_size
        }
        return stats




    def get_state(self, game: SnakeGame):
        head = game.head

        horizontal = [Direction.RIGHT, Direction.LEFT]
        vertical = [Direction.UP, Direction.DOWN]
        state = [
            # Danger straight
            # Breakdown: we use custom implemented signum to check if we should add block size or subtract, we map it to 0 1 with modulus for each direction and subtract one so we get either -1 or 0 which correspond to -1 or +1 signum and the check if the game.direction is the same as the checked direction
            any([game.direction == i and game.is_collision(
                Point(sign((int(i.value) % 2) - 1) * BLOCK_SIZE + head.x, head.y)) for i in
                 horizontal]) or
            any([game.direction == i and game.is_collision(
                Point(head.x, sign((int(i.value) % 2) - 1) * BLOCK_SIZE + head.y)) for i in
                 vertical]),
            # Danger right
            any([game.direction == j and game.is_collision(
                Point(sign((int(i.value) % 2) - 1) * BLOCK_SIZE + head.x, head.y)) for i, j in
                 zip(horizontal, vertical[::-1])]) or
            any([game.direction == j and game.is_collision(
                Point(head.x, sign((int(i.value) % 2) - 1) * BLOCK_SIZE + head.y)) for i, j in
                 zip(vertical, horizontal)]),

            # Danger left
            any([game.direction == j and game.is_collision(
                Point(sign((int(i.value) % 2) - 1) * BLOCK_SIZE + head.x, head.y)) for i, j in
                 zip(horizontal, vertical)]) or
            any([game.direction == j and game.is_collision(
                Point(head.x, sign((int(i.value) % 2) - 1) * BLOCK_SIZE + head.y)) for i, j in
                 zip(vertical, horizontal[::-1])]),

            #Move direction
            *(i == game.direction for i in horizontal+vertical),

            # food direction
            game.closest_food.x < game.head.x,
            game.closest_food.x > game.head.x,
            game.closest_food.y < game.head.x,
            game.closest_food.x > game.head.x,

            self.steps_left/self.max_idle_steps # How many idle steps until killed normalised
        ]


        return np.array(state, dtype=float)



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        action_indices = [np.argmax(action).item() for action in actions]
        self.trainer.train_step(states, action_indices, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        action_idx = np.argmax(action).item()
        self.trainer.train_step([state], [action_idx], [reward], [next_state], [done])

    def reset_steps(self):
        # Reset steps after eating apple
        self.steps_left = self.max_idle_steps

    def get_action(self, state):
        # randomness in the beginning: tradeoff between exploration and exploitation
        # epsilon decays after each step
        self.epsilon = max(self.epsilon_min, self.epsilon-self.epsilon_decay)
        self.steps_left -= 1

        self.n_steps+=1

        if self.n_steps % 1000 == 0:
            self.trainer.update_target_network()
        #self.epsilon = max(self.epsilon_min, self.epsilon_start - (self.n_games * self.epsilon_decay))
        final_move = [0,0,0]
        if random.random() < self.epsilon:
            rand = random.randint(0, 2)
            final_move[rand] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

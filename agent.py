import torch
import random
import numpy as np
from model import Linear_QNet, QTrainer
from game import SnakeGame, Direction, Point, BLOCK_SIZE
from collections import deque
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
sign = lambda x: math.copysign(1, x)



plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()  # Create a single figure and axes

def plot(scores, mean_scores):
    ax.clear()  # Clear previous data, but keep axes
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
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.8 # discount rate, < 1
        self.memory = deque(maxlen=MAX_MEMORY)
        self.hidden_size = 256
        self.model = Linear_QNet(11, self.hidden_size, 3)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)
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
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.x,
            game.food.x > game.head.x
        ]

        return np.array(state, dtype=int)



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    def get_action(self, state):
        # randomness in the beginning: tradeoff between exploration and exploitation
        self.epsilon = max(1, 80 - self.n_games)
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            rand = random.randint(0, 2)
            final_move[rand] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame(ai=True)
    while 1:
        state_old = agent.get_state(game)

        #get move
        move = agent.get_action(state_old)

        reward, done, score = game.play_step(move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, move, reward, state_new, done)
        agent.remember(state_old, move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            plot_mean_scores.append(np.mean(plot_scores))
            plot(plot_scores, plot_mean_scores)
            print('Game:', agent.n_games, 'Score:', score, 'Reward:', reward)

if __name__ == '__main__':
    train()

#print([Point(sign((int(i.value) % 2) - 1)*BLOCK_SIZE + 0, 5) for i in [Direction.RIGHT, Direction.LEFT]])
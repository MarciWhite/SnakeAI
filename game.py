from typing import Dict

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

#reset
#reward
#play(action) -> direction
#game iteration
#is_collision

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    DOWN = 3
    UP = 4



Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 100
WIDTH = 640
HEIGHT = 480

class SnakeGame:

    def __init__(self, ai = True, settings=None, render=True):
        #init variables
        settings = settings or {}
        self.frame_iteration = None
        self.ai = ai
        self.render = render
        self.closest_food = None
        self.score = None
        self.snake = None
        self.head = None
        self.direction = None

        self.w = settings.get("width", WIDTH)
        self.h = settings.get("height", HEIGHT)
        self.speed = settings.get("speed", SPEED)
        self.block_size = settings.get("block_size", BLOCK_SIZE)
        self.start_size = settings.get("snake_start_size", 3)
        self.hard_boundary = settings.get("hard_boundary", True)

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # init game state
        self.reset()
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head] + [Point(self.head.x - i*self.block_size, self.head.y) for i in range(self.start_size-1)]

        self.score = 0
        self.closest_food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.h - self.block_size) // self.block_size) * self.block_size
            new_food = Point(x, y)
            if new_food not in self.snake:
                self.closest_food = new_food
                break
    def _distance(self, a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)/self.block_size

    def export_settings(self) -> Dict:
        return {
            "speed": self.speed,
            "width": WIDTH,
            "height": HEIGHT,
            "block_size": self.block_size,
            "snake_start_size": self.start_size,
            "hard_boundary": self.hard_boundary
        }


    def play_step(self, action=None):
        self.frame_iteration += 1
        # 1. collect user input
        if not self.ai:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN and not self._out_of_bounds([self.head]):
                    if event.key == pygame.K_a and self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_d and self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_w and self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_s and self.direction != Direction.UP:
                        self.direction = Direction.DOWN
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Stop")
            # Determine direction based on action [straight, right, left]
            clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
            curr = clock_wise.index(self.direction)
            action = np.array(action)
            move = int(np.argmax(action))  # 0=straight, 1=right, 2=left
            if move == 0:
                new_direction = clock_wise[curr]
            elif move == 1:
                new_direction = clock_wise[(curr + 1) % 4]
            else:
                new_direction = clock_wise[(curr - 1) % 4]
            self.direction = new_direction

        # 2. move
        prev_head = self.head  # store position before moving
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 3. compute reward, check for game over
        reward, done, score = self._get_reward(prev_head)
        if done:
            return reward, True, score

        # 4. update UI and clock
        if self.render:
            self._update_ui()
            self.clock.tick(self.speed)

        # 5. return reward and score
        return reward, False, score

    def _get_reward(self, prev_head):
        STEP_PENALTY = -0.05
        DISTANCE_WEIGHT = -0.05
        CLOSER_REWARD = 0.05
        FOOD_REWARD = 10
        DEATH_PENALTY = -10

        max_dist = self._distance(Point(0, 0), Point(WIDTH, HEIGHT))
        prev_dist = self._distance(prev_head, self.closest_food)
        new_dist = self._distance(self.head, self.closest_food)
        normalized_dist = new_dist / max_dist

        reward = STEP_PENALTY + DISTANCE_WEIGHT * normalized_dist

        # check collisions or timeout
        if self.is_collision() or self._out_of_bounds() or (self.ai and self.frame_iteration > len(self.snake) * (self.w * self.h) // (BLOCK_SIZE ** 2) // 10):
            return DEATH_PENALTY, True, self.score

        # food collected
        if self.head == self.closest_food:
            self.score += 1
            reward = FOOD_REWARD
            self._place_food()
        else:
            self.snake.pop()

        return reward, False, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if self.hard_boundary and (pt.x > self.w - self.block_size or pt.x < 0 or pt.y > self.h - self.block_size or pt.y < 0):
           return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False
    def _out_of_bounds(self, list=None):
        #Check if the snake ran off-screen
        if list is None: list = self.snake
        return all([(i.x > self.w - self.block_size or i.x < 0 or i.y > self.h - self.block_size or i.y < 0) for i in list])
    def _update_ui(self):
        self.display.fill(BLACK)
        inner_margin = self.block_size * 0.2  # 20% margin
        inner_size = self.block_size - 2 * inner_margin
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(
                self.display,
                BLUE2,
                pygame.Rect(pt.x + inner_margin, pt.y + inner_margin, inner_size, inner_size)
            )

        pygame.draw.rect(self.display, RED, pygame.Rect(self.closest_food.x, self.closest_food.y, self.block_size, self.block_size))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.DOWN:
            y += self.block_size
        elif direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)

        if self.hard_boundary: return

        if self.head.x > self.w - self.block_size or self.head.x < 0:
            self.head = Point(self.w - self.head.x - self.block_size, self.head.y)
        elif self.head.y > self.h - self.block_size or self.head.y < 0:
            self.head = Point(self.head.x, self.h - self.head.y - self.block_size)


#User controlled
if __name__ == '__main__':
    game = SnakeGame(False)
    score = 0
    # game loop
    while not game.ai:
        reward, game_over, score = game.play_step()

        if game_over:
            break

    print('Final Score', score)
    pygame.quit()
import pygame
import torch

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
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
SPEED = 30


class SnakeGame:

    def __init__(self, ai = False, w=1280, h=960):
        #init variables
        self.frame_iteration = None
        self.ai = ai
        self.hard_boundary = True
        self.food = None
        self.score = None
        self.snake = None
        self.head = None
        self.direction = None
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        # init game state
        self.reset()
    def reset(self):
        self.direction = Direction.RIGHT
        self.start_size = 8
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head] + [Point(self.head.x - i*BLOCK_SIZE, self.head.y) for i in range(self.start_size-1)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
    def test(self):
        sign = lambda x: math.copysign(1, x)
        head = self.head
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
            any([ game.direction == j and game.is_collision(
                Point(sign((int(i.value) % 2) - 1) * BLOCK_SIZE + head.x, head.y)) for i, j in
                 zip(horizontal, vertical[::-1])]) or
            any([game.direction == j and game.is_collision(
                Point(head.x, sign((int(i.value) % 2) - 1) * BLOCK_SIZE + head.y)) for i, j in
                 zip(vertical, horizontal)]),

            # Danger left
            any([game.direction == j  and game.is_collision(
                Point(sign((int(i.value) % 2) - 1) * BLOCK_SIZE + head.x, head.y)) for i, j in
                 zip(horizontal, vertical)]) or
            any([game.direction == j and game.is_collision(
                Point(head.x, sign((int(i.value) % 2) - 1) * BLOCK_SIZE + head.y)) for i, j in
                 zip(vertical, horizontal[::-1])]),
        ]

        print(state)


    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action=None):
        self.frame_iteration+=1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if not self.ai:
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
                # Determine direction based on action [straight, left, right]
                clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
                curr = clock_wise.index(self.direction)
                new_direction = self.direction
                if np.array_equal(action, [1,0,0]):
                    new_direction = clock_wise[curr]
                elif np.array_equal(action, [0,1,0]):
                    new_direction = clock_wise[(curr+1) % 4]
                else:
                    new_direction = clock_wise[(curr - 1) % 4]
                self.direction = new_direction


        # 2. move
        self._move(self.direction)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        if self.is_collision() or self._out_of_bounds() or (self.ai and self.frame_iteration > 100*len(self.snake)):
            reward = -10
            return reward, True, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward=10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        #self.test()
        # 6. return game over and score
        return reward, False, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if self.hard_boundary and (pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0):
           return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False
    def _out_of_bounds(self, list=None):
        #Check if the snake ran off-screen
        if list is None: list = self.snake
        return all([(i.x > self.w - BLOCK_SIZE or i.x < 0 or i.y > self.h - BLOCK_SIZE or i.y < 0) for i in list])
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

        if self.hard_boundary: return

        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0:
            self.head = Point(self.w - self.head.x - BLOCK_SIZE, self.head.y)
        elif self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            self.head = Point(self.head.x, self.h - self.head.y - BLOCK_SIZE)


#User controlled
if __name__ == '__main__':
    game = SnakeGame()

    # game loop
    while not game.ai:
        reward, game_over, score = game.play_step()

        if game_over:
            break

    print('Final Score', score)
    pygame.quit()
from hashlib import new
from math import fabs
from sys import flags
from turtle import shape
import pygame
import random
from collections import deque
from enum import Enum
from PIL import Image
from collections import namedtuple
import numpy as np

import os
#os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (125,125,125)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 0, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 25
SPEED = 1000

class SnakeGameAI:

    def __init__(self, w=250, h=250):
        self.w = w
        self.h = h
        self._frames = None
        self._num_last_frames = 4
        self.reward = 0
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self._frames = None
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    
    def get_last_frames(self, observation):
        """
        Gets the 4 previous frames of the game as the state.
        Credits goes to https://github.com/YuriyGuts/snake-ai-reinforcement.
        
        :param observation: The screenshot of the game
        :return: The state containing the 4 previous frames taken from the game
        """
        frame = observation
        if self._frames is None:
            self._frames = deque([frame] * self._num_last_frames)
        else:
            self._frames.append(frame)
            self._frames.popleft()
        state = np.asarray(self._frames)#.transpose()  # Transpose the array so the dimension of the state is (84,84,4)

        return state

    def initState(self):
        self._update_ui()
        return self.get_last_frames(self.screenshot())

    def screenshot(self):
        """
        Takes a screenshot of the game , converts it to grayscale, reshapes it to size INPUT_HEIGHT, INPUT_WIDTH,
        and returns a np.array.
        Credits goes to https://github.com/danielegrattarola/deep-q-snake/blob/master/snake.py
        """
        data = pygame.image.tostring(self.display, 'RGB')  # Take screenshot
        #data = pygame.surfarray.array3d(pygame.display.get_surface())
        image = Image.frombytes('RGB', (250, 250), data)
        image = image.convert('L')  # Convert to greyscale
        image = image.resize((84, 84))
        matrix = np.asarray(image.getdata(), dtype=np.uint8)
        matrix = (matrix - 128)/(128 - 1)  # Normalize from -1 to 1
        return matrix.reshape(image.size[0], image.size[1])


    def play_step(self, action):
        final_move = [0,0,0,0]
        final_move[action] = 1
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(final_move) # update the head
        self.snake.insert(0, self.head)
        ########
        # 3. check if game over
        reward = -0.02
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -1
            return self.get_last_frames(self.screenshot()), reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 2
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick()
        # 6. return game over and score
        return self.get_last_frames(self.screenshot()), reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(WHITE)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0, 0]):
            new_dir = Direction.RIGHT
        elif np.array_equal(action, [0, 1, 0, 0]):
            new_dir = Direction.LEFT
        elif np.array_equal(action, [0, 0, 1, 0]):
            new_dir = Direction.UP
        elif np.array_equal(action, [0, 0, 0, 1]):
            new_dir = Direction.DOWN


        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)




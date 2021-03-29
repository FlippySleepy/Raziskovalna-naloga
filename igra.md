# Raziskovalna-naloga

#tukaj so najprej vstavljene knjižnice
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import torch
from model import Linear_QNet


pygame.init()
font = pygame.font.Font('arial.ttf', 25) #tukaj je določena velikos in pa oblika pisave, ki je izpisana v igri



#pretvorba smeri v številke
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb zapisi pretvorjeni v barve 
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20 #to je nastavitev, ki spreminja velikost kačinih "členov"
SPEED = 40 #to je nastavitev za hitrost premikanja kače

#v temu bo potekala celotna igra
class SnakeGameAI:

    def __init__(self, w=120, h=120): #najprej je nastavljena velikost polja
        self.w = w
        self.h = h
        # tukaj je vstavljeno okno v katerem igra poteka in pa časovnik
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        
    #tukaj je podprogram, ki ob ponovnem zagonu igre postavi kačo na sredino polja obrnjeno v desno stran in pa ponastavi rezultat na nič
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
        Point(self.head.x - BLOCK_SIZE, self.head.y),
        Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    #tukaj je podprogram, ki naključno postavlja hrano na polje
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                torch.save(Linear_QNet, 'tensors.pt')
                pygame.quit()
                quit()


        # 2. premik
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. preveri če je igre konec
        reward = 0
        game_over = False
        if self.is_collision():
            print('col')
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.frame_iteration > 100*len(self.snake):
            print('time')
            game_over = True
            reward = -10
            return reward, game_over, self.score

        #4.tukaj je program, ki omogoča, da ko poberemo hrano, hrana izgine in se pojavi drugje in pa, da ko pobereš hrano dobiš nagrado
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. posodobitev ure in pa zaslona
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
        
    #ta podprogram preverja če se kača zadane v kar koli kar bi pomenilo konec igre
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # če zadane rob polja
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
            # če zadane sama sebe
        if pt in self.snake[1:]:
            return True

        return False
        
    #posodobitev zaslona
    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Rezultat: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    #tukaj so določene smeri za premikanje, ki jih uporablja agent
    def _move(self, action):

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

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

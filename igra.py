# Raziskovalna-naloga
# Igra.py s komentarji.

    # Izposojanje knjižnic
    import pygame 
    # Ena najbolj priljubljenih Pythonovih knjižnic namenjena ustvarjanju iger.
    import random
    from enum import Enum
    import numpy as np   
    from collections import namedtuple
    # Knjižnice namenjene matematičnim operacijam.
    
    import torch
    from model import Linear_QNet
    # Knjižnica za strojno učenje ter podprogram iz datoteke model.py.


    pygame.init() # Inicializacija modula Pygame.
    font = pygame.font.Font('arial.ttf', 25) # Določena vrsta ter velikost pisave. (Ni pomemben pri raziskavah, ima zgolj estetski namen)

    class Direction(Enum):
        RIGHT = 1
        LEFT = 2
        UP = 3
        DOWN = 4
    # Razred za pretvorbo smeri iz števil v konkretne besede. (Pomembno zaradi Pygame ukazov).

    Point = namedtuple('Point', 'x, y') # Definicija točke.
 
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)
    # Za lažjo uporabo, konstante nastavimo na RGB vrednosti.

    BLOCK_SIZE = 20 # Velikost kock v mreži, posledično kačinih členov.
    SPEED = 40 # Hitrost igre.
    
    class SnakeGameAI:
    # Razred igre. Tu bo potekala igra kača (brez strojnega učenja).
        def __init__(self, w=120, h=120): 
            self.w = w
            self.h = h            
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()
            self.reset()
        # Inicializacija začetnih spremenljivk objekta igre. Višina, širina mreže, naslov, ura...

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
        # Funkcija, ki ponastavi kačo v primeru smrti ali ob začetku igre (glej igra.py/48).
        # Obrnjena bo desno, začela pa bo na sredini mreže z dolžino 2.

        def _place_food(self):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food in self.snake:
                self._place_food()
        # Funkcija za naključno postavitev hrane v primeru, da jo je kača pojedla (glej igra.py/107) ali ob začetku igre (glej igra.py/60).
        
        # (Spodaj) Funkcija za potek igre. Igra je osnovana na iteracijah okvirov zato je to potrebno.
        def play_step(self, action):
            self.frame_iteration += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    torch.save(Linear_QNet, 'tensors.pt')
                    pygame.quit()
                    quit()
             # V primeru izhoda iz igre se shrani nevronska mreža v datoteko 'tensors.pt'.
            
            self._move(action) 
            self.snake.insert(0, self.head)
            
            # Premik kače glede na podano vrednost nevronske mreže.
            
            reward = 0
            game_over = False
            
            if self.is_collision():
                game_over = True
                reward = -10
                return reward, game_over, self.score
            # V primeru kačine smrti zaradi trčenja se modelu (model.py) vrne nagrado in točke.
            
            
            if self.frame_iteration > 100*len(self.snake):
                game_over = True
                reward = -10
                return reward, game_over, self.score
            # V primeru kačine smrti zaradi pomanjkanja hrane se modelu (model.py) vrne nagrado in točke.
            
            if self.head == self.food:
                self.score += 1
                reward = 10
                self._place_food()
            else:
                self.snake.pop()
            # V primeru, da je kača hrano pojedla, se nagrada in točke povečajo, hrane pa se postavi na novo mesto. 
            # Če do te točke kača ni pojedla hrane, ostane enako dolga, sicer se podaljša za en člen.
            
            self._update_ui()
            self.clock.tick(SPEED)
            # Delovanje ure, ki uravnava hitrost igre.
            
            return reward, game_over, self.score
            # Funkcija vrne nagrado, točke in stanje igre ob vsaki iteraciji.
                
        def is_collision(self, pt=None):
            if pt is None:
                pt = self.head
            
            if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
                return True
            # Če zadane rob polja.
                
            if pt in self.snake[1:]:
                return True
            # Če zadane svoj rep.

            return False
        # Funkcija preverja trčenje kače s svojim repom ali robom polja.
        
        
        def _update_ui(self):
        # Funckija za osveževanje/posodobitev igre ob vsaki iteraciji.
            self.display.fill(BLACK)
        
            for pt in self.snake:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            # Risanje kače.
            
            pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
            # Risanje hrane.
            
            text = font.render("Rezultat: " + str(self.score), True, WHITE)
            self.display.blit(text, [0, 0])
            # Izpis točk. 
            
            pygame.display.flip()
            # Skupaj z vrstico 138 namenjena osveževanju zaslona.

        
        def _move(self, action):
        # Funkcija za obračanje kače glede na zahtevo modela.
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
            # Iz vrednosti oblike [a, b, c] je treba pretvoriti smer v uporabno verzijo.

            self.direction = new_dir
            # Obračanje.
            
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
            # Končen premik kače.
            self.head = Point(x, y)

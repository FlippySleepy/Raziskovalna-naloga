#Raziskovalna naloga
Tukaj je razložen še en del programa in sicer agent

    #tukaj so vstavljene vse knjižnice in pa podprogrami iz drugih najnih programov 
    import torch
    import random
    import numpy as np
    from collections import deque
    from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
    from model import Linear_QNet, QTrainer
    from helper import plot


    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LR = 0.001


    class Agent:
        #v tem podprogramu program odpre beesedilni dokument in ga bere ter kasneje tudi spreminja
        def __init__(self):
            with open('record.txt', 'r') as f:
                self.n_games = int(f.read())
                print(self.n_games)

            self.epsilon = 0
            self.gamma = 0.9
            self.memory = deque(maxlen=MAX_MEMORY)
            self.model = Linear_QNet(11, 256, 3)
            self.model.load_state_dict(torch.load('./model\model.pth'))
            self.model.eval()
            self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
        #tukaj je določeno da agent ve kam kača gleda in kam se premika
        def get_state(self, game):
            head = game.snake[0]
            point_l = Point(head.x - BLOCK_SIZE, head.y)
            point_r = Point(head.x + BLOCK_SIZE, head.y)
            point_u = Point(head.x, head.y - BLOCK_SIZE)
            point_d = Point(head.x, head.y + BLOCK_SIZE)

            dir_l = game.direction == Direction.LEFT
            dir_r = game.direction == Direction.RIGHT
            dir_u = game.direction == Direction.UP
            dir_d = game.direction == Direction.DOWN

            state = [

                # nevarnost naravnost
                (dir_r and game.is_collision(point_r)) or
                (dir_l and game.is_collision(point_l)) or
                (dir_u and game.is_collision(point_u)) or
                (dir_d and game.is_collision(point_d)),

                # nevarnost na desni
                (dir_u and game.is_collision(point_r)) or
                (dir_d and game.is_collision(point_l)) or
                (dir_l and game.is_collision(point_u)) or
                (dir_r and game.is_collision(point_d)),

                # nevarnost na levi
                (dir_d and game.is_collision(point_r)) or
                (dir_u and game.is_collision(point_l)) or
                (dir_r and game.is_collision(point_u)) or
                (dir_l and game.is_collision(point_d)),

                # smeri premikanja
                dir_l,
                dir_r,
                dir_u,
                dir_d,

                # lokacija hrane
                game.food.x < game.head.x,
                game.food.x > game.head.x,
                game.food.y < game.head.y,
                game.food.y > game.head.y

            ]

            return np.array(state, dtype=int)

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))
           
        def train_long_memory(self):
            if len(self.memory) > BATCH_SIZE:
                mini_sample = random.sample(self.memory, BATCH_SIZE)  # number of tuples
            else:
                mini_sample = self.memory

            states, actions, rewards, next_states, dones = zip(*mini_sample)
            self.trainer.train_step(states, actions, rewards, next_states, dones)

        def train_short_memory(self, state, action, reward, next_state, done):
            self.trainer.train_step(state, action, reward, next_state, done)

        def get_action(self, state):
            # random moves: tradeoff exploration / exploitation
            self.epsilon = 80 - self.n_games
            final_move = [0, 0, 0]
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1

            return final_move

    #tukaj je program s katerima se kača uči
    def train():
        score = 0
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        with open('record.txt', 'r') as f:
            record = f.read()
        record = int(record)
        agent = Agent()
        game = SnakeGameAI()
        while True:
            # dobi staro stanje
            state_old = agent.get_state(game)

            # premikanje
            final_move = agent.get_action(state_old)
            # perform move
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # treniranje kratkoročnega spomina
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # zapomnitev
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # treniranje dolgoročnega spomina
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()
                    with open('record.txt', 'w') as f:
                        f.write(str(record))
                    with open('games.txt', 'w') as f:
                        f.write(str(agent.n_games))

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

                print('Povprečen rezultat: ', mean_score)


    if __name__ == '__main__':
        train()

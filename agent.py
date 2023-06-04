import torch
import random
import numpy as np

from collections import deque
from snake_game_ai import SnakeGameAI, Direction, Point

maxMem = 100_000
batchSize = 1000  # batches of sample which will be passed through the model
lr = 0.001  # learning rate of model

class Agent:

    def __init__(self):
        self.n_games = 0  # number of games
        self.epsilon = 0  # randomness
        self.gamma = 0  # discount rate
        self.memory = deque(maxlen= maxMem)
        self.model = None
        self.trainer = None

    def get_state(self, game):  # states to understand dangers food location, 11 such states
        head = game.snake[0]  # head of snake
        # points around the head to check for danger and food, this checks the 4 boxes around the head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        #  to see in which direction the snake is moving
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [  # the leven states to detect danger and food
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction, only one of te 4 values here will be true as it can only move in one direction ata time
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # stores the data(one tuple) into the deque, basically
        # stores all the test cases into memory ig

    def train_long_memory(self):
        if len(self.memory) > batchSize:  #  loads samples in batches of 1000 for testing ig not sure
            mini_sample = random.sample(self.memory, batchSize)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        #  trains the model with the batch samples acquired
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # to get the move this is the function which makes the snake move
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:  # as the number of games increases he value of epsilon will
            # decrease so will the random moves, thus making the snake move on the model generated moves
            move = random.randint(0, 2)
            final_move[move] = 1
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
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)  # done is gme over passed from snakegameAI
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score


            print('Game', agent.n_games, 'Score', score, 'Record:', record)


if __name__ == '__main__':
    train()
import time
import torch
import random
import numpy as np

import sokobanbot
from sokobanbot import Sokoban
from collections import deque
from model import QTrainer, Linear_QNet
import pickle
import os
import matplotlib.pyplot as plt
import sokobanbot

MAX_MEMORY = 100_000
BATCH_SIZE = 1024
LR = 0.001  # learning rate

games_to_train = 10_000
avg_track = 75


class Agent:

    def __init__(self):

        self.games_completed = 0
        self.games_played = 0

        self.epsilon = 1.0  # randomness
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999995
        temp_game = Sokoban()
        # Size to be passed into model
        input_size = temp_game.num_objects * 4 + 6
        self.gamma = 0.9  # cares about long term reward (very cool)
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft when memory is reached
        # Uses CUDA for training (if having eligible gpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Init model, .to(self.device) moves the data from RAM to VRAM so the gpu can train it
        self.model = Linear_QNet(input_size, 512, 4).to(self.device)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        # State array is as follows:
        """
        A ‘state’ array consisting of # boolean values, and normalised float: (6 + n*4 + n*4)

        N = amount of blocks/ holes

	    User ability to move:
		    UP, DOWN, LEFT, RIGHT
		USER position:
		    X, Y

        Each block:
		    Availability to user:
            VERTICAL, HORIZONTAL


        Each hole:
	        Availability to user:
            VERTICAL, HORIZONTAL

        """
        # len = 4
        state = [
            game.can_move_up(),
            game.can_move_down(),
            game.can_move_left(),
            game.can_move_right()
        ]

        # len = 2
        state.extend(game.player_state())
        # len = num_objects * 2
        state.extend(game.block_state())
        # len = num_objects * 2
        state.extend(game.hole_state())

        return np.array(state, dtype=int)  # convert bools and floats to np array,

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))  # pop left if MAX_MEMORY is reached

    # Trains AI on other random games too
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # returns list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # Trains AI on game that just finished
    def train_short_memory(self, state, action, reward, next_state, game_over):
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        """
        Decide which action to take given the current state.

        Uses epsilon-greedy strategy:
        - With probability epsilon: choose a random action (exploration)
        - Otherwise: choose the action with the highest predicted Q-value (exploitation)
        """

        # Update epsilon (exploration rate)
        # As the number of games increases, epsilon decreases
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Array denoting the move to be made, udlr
        final_move = [0, 0, 0, 0]

        # Decide whether to explore or exploit
        if random.random() < self.epsilon:
            move = random.randint(0, 3)
        else:
            # Convert state to a PyTorch tensor
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction, dim=1).item()

        final_move[move] = 1
        return final_move


def train():
    rewards = []
    record = 10_000_000
    agent = Agent()

    plt.ion()

    game = Sokoban()
    total_reward = 0
    cur_moves = 0

    global avg_track
    global games_to_train
    # For each game that gets complete, stores avg moves last for last {avg_track} games
    # When training is finished len(avg_moves) == games_to_train
    avg_moves = []
    # Total moves made last {avg_track} games
    moves_last_track = 0
    # List of moves in last {avg_track} games
    moves_made = []



    while agent.games_completed < games_to_train:
        # get old state
        state_old = agent.get_state(game)

        # get move
        get_move = agent.get_action(state_old)

        # perform move and get state
        reward, game_over, game_win = game.play_step(get_move)
        state_new = agent.get_state(game)

        # train short mem
        agent.train_short_memory(state_old, get_move, reward, state_new, game_over)

        # remember
        agent.remember(state_old, get_move, reward, state_new, game_over)
        total_reward += reward

        cur_moves += 1

        if game_over:
            agent.games_completed += 1
            if game_win:
                # allow the bot to learn more

                # Increments number of games won
                agent.games_completed += 1

                # Adds cur_moves made to moves_made list
                moves_made.append(cur_moves)
                # Add cur_moves to sum of moves made last 100 games
                moves_last_track += cur_moves
                # If there is more than {avg_track} moves made
                if len(moves_made) > avg_track:
                    # subtract the moves from {avg_track}th game ago
                    moves_last_track -= moves_made[0]
                    # Removes from list
                    moves_made.pop(0)
                # calculate average moves
                avg_moves.append(moves_last_track / len(moves_made))
                # keeps track of record
                if record > cur_moves:
                    record = cur_moves
                    # Saves this model, as it is 'seemingly' the best (could just be lucky scramble)
                    agent.model.save()

                print(f'Games: {agent.games_completed}, Record: {record}')

            # train long term mem
            game.reset()
            for _ in range(4):  # train 4x per episode
                agent.train_long_memory()

            rewards.append(total_reward)
            cur_moves = 0

            # Display Graph
            if agent.games_completed % 100 == 0:
                plt.clf()  # clear previous plot
                game_number = list(range(len(avg_moves)))
                plt.plot(game_number, avg_moves)
                plt.xlabel('Game Number')
                plt.ylabel(f'Average moves last {avg_track} games')
                plt.pause(0.1)  # updates the plot without blocking

            if agent.games_completed % 1000 == 0:
                print(f'Games: {agent.games_completed}, Record: {record}')




if __name__ == '__main__':
    train()
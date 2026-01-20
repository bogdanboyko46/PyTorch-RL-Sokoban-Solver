import time

import torch
import random
import numpy as np
from sokocopytemp import Sokoban
from collections import deque
from model import QTrainer, Linear_QNet

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # learning rate

class Agent:

    def __init__(self):
        
        self.number_of_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # cares about long term reward (very cool)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft when memory is reached
        self.model = Linear_QNet(12, 256, 4)
        self.trainer = QTrainer(self.model, LR, self.gamma)


    def get_state(self, game):

        # State array is as follows:
        """"
        A ‘state’ array consisting of # boolean values (T, F): (4 + n*8)

        N = amount of blocks
        M = amount of holes

	    User ability to move:
		    UP, DOWN, LEFT, RIGHT
        
        Each block:
		    Availability to user:
            UP, DOWN, LEFT, RIGHT

        Danger value (when user is in contact with block and next to unmovable pos)

        Each hole:
	        Availability of user:
	        UP, DOWN, LEFT, RIGHT

        """
        state = [
            game.can_move_up(),
            game.can_move_down(),
            game.can_move_left(),
            game.can_move_right()
        ]
        state.extend(game.block_state())
        state.extend(game.hole_state())


        return np.array(state, dtype=int) # convert bools to np array, convert bools to ints

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over)) # pop left if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

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
        self.epsilon = max(0, 80 - self.number_of_games)

        # Array denoting the move to be made, udlr
        final_move = [0, 0, 0, 0]

        # Decide whether to explore or exploit
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
        else:
            # Convert state to a PyTorch tensor
            state0 = torch.tensor(state, dtype=torch.float)

            # Get Quality Values for all actions from the network
            with torch.no_grad():  # no gradients needed when acting
                prediction = self.model(state0)

            # Choose the highest quality action
            move = torch.argmax(prediction).item()

        # Convert action index to one-hot encoding
        final_move[move] = 1
        return final_move


def train():
    steps_taken = [] # the # of steps the agent takes to win per game
    record = 10000000
    agent = Agent()
    game = Sokoban()

    while True:
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


        if game_over:
            if game_win:
                if record == 10000000 or record > game.moves_made:
                    record = game.moves_made
                    agent.model.save()

                steps_taken.append(current_steps)
                game.comp = True
            # train long term mem
            game.reset()
            agent.train_long_memory() 

            # reset current steps
            current_steps = 0
            agent.number_of_games += 1



            print(f'Games: {agent.number_of_games}, Record: {record}, Completed: {len(steps_taken)}')

if __name__ == '__main__':
    train()

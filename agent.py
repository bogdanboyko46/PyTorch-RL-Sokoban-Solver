import time

import torch
import random
import numpy as np
from sokobangamepy import Sokoban
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # learning rate

class Agent:

    def __init__(self):
        
        self.number_of_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft when memory is reached
        self.model = None # TODO
        self.trainer = None # TODO
        # TODO: model, trainer

    def get_state(self, game):

        # State array is as follows:
        """"
        A ‘state’ array consisting of # boolean values (T, F): (4 + n*5 + m*4)

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

        print(state)

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
        # do some random moves in the beginning
        self.epsilon = 80 - self.number_of_games
        
        final_move = [0,0,0,0] # UP, DOWN, LEFT, RIGHT
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            # give the agent the current state in torch.float format
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)

            # prediction returns a list for ex [2.7, 3, 8, 0]
            # "move" then gets the index of the maximum element
            move = torch.argmax(prediction).item()

            final_move[move] = 1

        return final_move
        
def train():
    steps_taken = [] # the # of steps the agent takes to win per game
    current_steps = 0
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
        
        current_steps += 1

        if game_over:
            # train long term mem
            game.reset()
            agent.train_long_memory() 
            
            if game_win:
                steps_taken.append(current_steps)

            # reset current steps
            current_steps = 0
            agent.number_of_games += 1
            
            if current_steps > max(steps_taken):
                # TODO: implement save function from  model.py
                pass

            print(f'Games: {agent.number_of_games}, Record: {max(steps_taken)}')

if __name__ == '__main__':
    agent = Agent()
    game = Sokoban()
    time.sleep(2)
    while True:
        agent.get_state(game)
        if game.play_step():
            break

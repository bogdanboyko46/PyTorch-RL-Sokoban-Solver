import time
import torch
import random
import numpy as np
from sokobanbot import Sokoban
from collections import deque
from model import QTrainer, Linear_QNet
import pickle
import os
import matplotlib.pyplot as plt

MAX_MEMORY = 100_000
BATCH_SIZE = 128
LR = 0.001  # learning rate

CKPT_PATH = "agent_checkpoint.pth"
MEM_PATH = "replay_memory.pkl"

class Agent:

    def __init__(self):

        self.number_of_games = 0
        self.epsilon = 1.0  # randomness
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999995

        self.gamma = 0.9  # cares about long term reward (very cool)
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft when memory is reached
        self.model = Linear_QNet(10, 256, 4)
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
            UP, DOWN, LEFT, RIGHT


        Each hole:
	        Availability to user:
	        UP, DOWN, LEFT, RIGHT

        """
        state = [
            game.can_move_up(),
            game.can_move_down(),
            game.can_move_left(),
            game.can_move_right()
        ]

        state.extend(game.player_state())
        state.extend(game.block_state())
        state.extend(game.hole_state())

        return np.array(state, dtype=int)  # convert bools and floats to np array,

    def remember(self, state, action, reward, next_state, game_over):
        self.memory.append((state, action, reward, next_state, game_over))  # pop left if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # returns list of tuples
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
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Array denoting the move to be made, udlr
        final_move = [0, 0, 0, 0]

        # Decide whether to explore or exploit
        if random.random() < self.epsilon:
            move = random.randint(0, 3)
        else:
            # Convert state to a PyTorch tensor
            state0 = torch.tensor(state, dtype=torch.float).unsqueeze(0)  # shape [1, features]
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction, dim=1).item()

        final_move[move] = 1
        return final_move


def load_replay_memory(path: str, max_memory: int) -> deque:
    if not os.path.exists(path):
        return deque(maxlen=max_memory)

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # data could be a list OR a deque; wrap it to enforce maxlen
        return deque(data, maxlen=max_memory)
    except (pickle.UnpicklingError, EOFError, OSError):
        return deque(maxlen=max_memory)


def save_replay_memory(memory: deque, path: str) -> None:
    # Save as list for robustness; reconstruct deque(maxlen=...) on load
    with open(path, "wb") as f:
        pickle.dump(list(memory), f)


def save_checkpoint(agent: Agent, ckpt_path: str = CKPT_PATH, mem_path: str = MEM_PATH) -> None:
    checkpoint = {
        "model_state": agent.model.state_dict(),
        "optimizer_state": agent.trainer.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "number_of_games": agent.number_of_games,
        "max_memory": MAX_MEMORY,
    }
    torch.save(checkpoint, ckpt_path)
    save_replay_memory(agent.memory, mem_path)


def load_checkpoint(agent: Agent, ckpt_path: str = CKPT_PATH, mem_path: str = MEM_PATH) -> None:
    # Load model/optimizer/metadata if present
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        agent.model.load_state_dict(checkpoint["model_state"])
        agent.trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.epsilon = float(checkpoint.get("epsilon", agent.epsilon))
        agent.number_of_games = int(checkpoint.get("number_of_games", agent.number_of_games))

    # Load replay memory if present (independent of checkpoint)
    agent.memory = load_replay_memory(mem_path, MAX_MEMORY)

def train():
    rewards = []
    record = 10_000_000
    agent = Agent()

    load_checkpoint(agent) # load the checkpoint if exists

    game = Sokoban()
    print(game.savedstate)
    total_reward = 0
    temp_moves = 0

    SAVE_EVERY = 10

    while agent.number_of_games < 650:
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

        temp_moves += 1

        if game_over:
            # If the bot completed the puzzle successfully
            if game_win:
                # allow the bot to learn more
                agent.number_of_games += 1

                if record == 10000000 or record > temp_moves:
                    record = temp_moves
                    agent.model.save()

                print(f'Games: {agent.number_of_games}, Record: {record}')
                # Resets with new randomly generated puzzle
                game.reset()

            # If the bot wasnt able to solve puzzle
            else:
                # Reloads the starting position of puzzle (same positions of blocks,holes and player)
                game.reload()

            # train long term mem
            agent.train_long_memory()

            # save the checkpoint every 10 moves
            if agent.number_of_games % SAVE_EVERY == 0:
                save_checkpoint(agent)

            rewards.append(total_reward)
            temp_moves = 0

    # final save
    save_checkpoint(agent)

    game_number = []
    for i in range(len(rewards)):
        game_number.append(i)

    plt.plot(game_number, rewards)
    plt.xlabel('Game Number')
    plt.ylabel('Total Reward')
    plt.show()


if __name__ == '__main__':
    train()
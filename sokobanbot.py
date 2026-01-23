import time

from fontTools.subset.svg import closure_element_ids

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import math
from torch.cuda.tunable import set_filename

# Initialize pygame modules
pygame.init()

font = pygame.font.Font('arial.ttf', 25)

# Enum for player movement directions
class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

# Point structure to store x and y coordinates
Point = namedtuple('Point', 'x, y')

# RGB color definitions
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
CYAN = (0, 255, 255)
PINK = (255, 0, 255)

# Size of each player block
BLOCK_SIZE = 80

class Sokoban:
    def __init__(self, w=720, h=720):
        # Screen width and height
        self.w = w
        self.h = h
        self.player = None
        self.blocks = None
        self.holes = None
        self.in_hole = 0
        # self.comp = False

        # Initialize game window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Sokoban')

        self.differences = dict()

        self.reset()

    def replace_difference(self, states):

        old_block = states[0]
        new_block = states[1]

        # compare the differences from the old state of block and new state
        # count the pos and neg change from each hole, 1, -1 denoting a closer path to a hole, farther path respectively
        closer_count = 0

        # initialize block's new state

        self.differences[new_block] = dict()

        for hole in self.differences[old_block].keys():
            old_diff = self.differences[old_block][hole]
            new_diff = math.sqrt((hole.x - new_block.x) ** 2 + (hole.y - new_block.y) ** 2) / BLOCK_SIZE
            closer_count += 1 if new_diff < old_diff else -1
            self.differences[new_block][hole] = new_diff

        self.differences[old_block].clear()

        if closer_count > 0:
            print("A BLOCK WAS PUSHED CLOSER TO HOLES")
            return 5
        else:
            print("A BLOCK WAS AWAY FROM HOLES")
            return -5

    def reset(self):
        print("-----------------------------------------------------------------------------------------------------------------")
        self.moves_made = 0
        self.blocks = set()
        self.holes = set()
        self.in_hole = 0
        self.differences.clear()

        x = random.randint(0, 12) * BLOCK_SIZE
        y = random.randint(0, 12) * BLOCK_SIZE
        self.player = Point(x, y)

        while len(self.blocks) < 2:
            x = random.randint(3, 7) * BLOCK_SIZE
            y = random.randint(3, 7) * BLOCK_SIZE
            if not Point(x, y) in self.blocks and Point(x, y) != self.player:
                self.blocks.add(Point(x, y))

        while len(self.holes) < 2:
            x = random.randint(2, 8) * BLOCK_SIZE
            y = random.randint(2, 8) * BLOCK_SIZE
            if not Point(x, y) in self.holes and not Point(x, y) in self.blocks and Point(x, y) != self.player:
                self.holes.add(Point(x, y))

        for block in self.blocks:
            self.differences[block] = dict()

            for hole in self.holes:
                self.differences[block][hole] = math.sqrt((hole.x - block.x) ** 2 + (hole.y - block.y) ** 2) / BLOCK_SIZE

    def hole_within_bounds(self):
        for hole in self.holes:
            if hole.y in (0, self.h - BLOCK_SIZE)or hole.x in (0, self.w - BLOCK_SIZE):
                return True
        return False

    def top_bottom_borders(self, y):
        return y in (0, self.h - BLOCK_SIZE)

    def left_right_borders(self, x):
        return x in (0, self.w - BLOCK_SIZE)

    def unmovable_block_detect(self):
        for block in self.blocks:
            # if the block is already in a hole
            if block in self.holes:
                continue

            # if the block is in a corner
            if block in (Point(0, 0),
                         Point(self.w - BLOCK_SIZE, 0),
                         Point(0, self.h - BLOCK_SIZE),
                         Point(self.w - BLOCK_SIZE, self.h - BLOCK_SIZE)):
                return True

            # if block is stuck in the top / bottom borders
            if self.top_bottom_borders(block.x):

                if self.hole_within_bounds():
                    return True

                if (Point(block.x + BLOCK_SIZE, block.y) in self.blocks
                or Point(block.x - BLOCK_SIZE, block.y) in self.blocks):
                    return True

            # if the block is stuck in the left / right borders
            if self.left_right_borders(block.x):

                if self.hole_within_bounds():
                    return True

                if (Point(block.x, block.y + BLOCK_SIZE) in self.blocks
                or Point(block.x, block.y - BLOCK_SIZE) in self.blocks):
                    return True

        return False

    def adjacent(self, x1, y1, x2, y2):
        pt = Point(x1, y1)

        if pt == Point(x2 - BLOCK_SIZE, y2):
            return Direction.RIGHT
        elif pt == Point(x2 + BLOCK_SIZE, y2):
            return Direction.LEFT
        elif pt == Point(x2, y2 - BLOCK_SIZE):
            return Direction.DOWN
        elif pt == Point(x2, y2 + BLOCK_SIZE):
            return Direction.UP

        return None

    def play_step(self, action):
        # TODO: return respective vars: reward, game_over, game_win

        # Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # action is [up, down, left, right]
        if isinstance(action, (list, tuple, np.ndarray)):
            idx = int(np.argmax(action))
            action = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT][idx]

        self.moves_made += 1

        reward = -1 # encourage shortest path
        game_over = False

        # old states
        old_ct_in_hole = self.in_hole
        old_x, old_y = self.player.x, self.player.y

        block_states = self._move(action)
        self._update_ui()

        if block_states:
            # detected that a block was moved, apply change to current difference dict
            reward += self.replace_difference(block_states)
            # can either be negative or positive depending on if block was moved towards holes

        # main logic reward system
        if old_x == self.player.x and old_y == self.player.y:
            reward -= 5

        # if a block gets pushed onto a hole, give the agent a positive reward
        if self.in_hole > old_ct_in_hole:
            if self.in_hole == len(self.holes):
                reward += 100
                game_over = True
                return reward, game_over, True

            else:
                reward += 10

            self.moves_made = 0
        elif self.in_hole < old_ct_in_hole:

            # the agent pushed a block off a hole, return a negative reward
            reward -= 5

        # if agent pushes a block into a corner, heavily discourage
        if self.unmovable_block_detect() or self.moves_made > 400:
            reward -= 10
            game_over = True
            return reward, game_over, False

        # return
        # time.sleep(.01)
        return reward, game_over, False

    # Moves the player, Returns True if a block is moved too
    def _move(self, direction):
        x = self.player.x
        y = self.player.y
        block_states = []

        if direction == Direction.RIGHT and self.can_move_right():
            if Point(x + BLOCK_SIZE, y) in self.blocks:
                old_pos =  Point(x + BLOCK_SIZE, y)
                self.blocks.remove(old_pos)
                if old_pos in self.holes:
                    self.in_hole -= 1
                new_pos = Point(x + BLOCK_SIZE * 2, y)
                self.blocks.add(new_pos)
                if new_pos in self.holes:
                    self.in_hole += 1
                block_states = [old_pos, new_pos]
            x += BLOCK_SIZE

        elif direction == Direction.LEFT and self.can_move_left():
            if Point(x - BLOCK_SIZE, y) in self.blocks:
                old_pos = Point(x - BLOCK_SIZE, y)
                self.blocks.remove(old_pos)
                if old_pos in self.holes:
                    self.in_hole -= 1
                new_pos = Point(x - BLOCK_SIZE * 2, y)
                self.blocks.add(new_pos)
                if new_pos in self.holes:
                    self.in_hole += 1
                block_states = [old_pos, new_pos]
            x -= BLOCK_SIZE

        elif direction == Direction.DOWN and self.can_move_down():
            if Point(x, y + BLOCK_SIZE) in self.blocks:
                old_pos = Point(x, y + BLOCK_SIZE)
                self.blocks.remove(old_pos)
                if old_pos in self.holes:
                    self.in_hole -= 1
                new_pos = Point(x, y + BLOCK_SIZE * 2)
                self.blocks.add(new_pos)
                if new_pos in self.holes:
                    self.in_hole += 1
                block_states = [old_pos, new_pos]
            y += BLOCK_SIZE

        elif direction == Direction.UP and self.can_move_up():
            if Point(x, y - BLOCK_SIZE) in self.blocks:
                old_pos = Point(x, y - BLOCK_SIZE)
                self.blocks.remove(old_pos)
                if old_pos in self.holes:
                    self.in_hole -= 1
                new_pos = Point(x, y - BLOCK_SIZE * 2)
                self.blocks.add(new_pos)
                if new_pos in self.holes:
                    self.in_hole += 1
                block_states = [old_pos, new_pos]
            y -= BLOCK_SIZE

        self.player = Point(x, y)
        return block_states

    def _update_ui(self):
        self.display.fill(BLACK)

        p_pt = self.player
        pygame.draw.rect(self.display, BLUE,
                         pygame.Rect(p_pt.x, p_pt.y, BLOCK_SIZE, BLOCK_SIZE))
        for b_pt in self.blocks:
            pygame.draw.rect(self.display, RED,
                             pygame.Rect(b_pt.x, b_pt.y, BLOCK_SIZE, BLOCK_SIZE))

        for h_pt in self.holes:
            if h_pt in self.blocks:
                pygame.draw.rect(self.display, GREEN,
                                 pygame.Rect(h_pt.x, h_pt.y, BLOCK_SIZE, BLOCK_SIZE))
            elif h_pt == p_pt:
                pygame.draw.rect(self.display, CYAN,
                                 pygame.Rect(h_pt.x, h_pt.y, BLOCK_SIZE, BLOCK_SIZE))
            else:
                pygame.draw.rect(self.display, WHITE,
                                 pygame.Rect(h_pt.x, h_pt.y, BLOCK_SIZE, BLOCK_SIZE))

        # Update the screen
        pygame.display.flip()
        pass

    def can_move_right(self) -> bool:
        x = self.player.x
        y = self.player.y

        if (x + BLOCK_SIZE) < self.w:
            new_x = x + BLOCK_SIZE
            if Point(new_x, y) in self.blocks:
                # Checks if block cant be pushed (out of bounds, or another block to blocks right)
                if new_x + BLOCK_SIZE >= self.w or Point(new_x + BLOCK_SIZE, y) in self.blocks:
                    return False
            return True
        return False

    def can_move_left(self) -> bool:
        x = self.player.x
        y = self.player.y
        if (x - BLOCK_SIZE) >= 0:
            new_x = x - BLOCK_SIZE
            if Point(new_x, y) in self.blocks:
                # Checks if block cant be pushed (out of bounds, or another block to blocks left)
                if new_x - BLOCK_SIZE < 0 or Point(new_x - BLOCK_SIZE, y) in self.blocks:
                    return False
            return True
        return False

    def can_move_down(self) -> bool:
        x = self.player.x
        y = self.player.y
        if (y + BLOCK_SIZE) < self.h:
            new_y = y + BLOCK_SIZE
            if Point(x, new_y) in self.blocks:
                # Checks if block cant be pushed (out of bounds, or another block to blocks down)
                if new_y + BLOCK_SIZE >= self.w or Point(x, new_y + BLOCK_SIZE) in self.blocks:
                    return False
            return True
        return False

    def can_move_up(self) -> bool:
        x = self.player.x
        y = self.player.y
        if (y - BLOCK_SIZE) >= 0:
            new_y = y - BLOCK_SIZE
            if Point(x, new_y) in self.blocks:
                # Checks if block cant be pushed (out of bounds, or another block to blocks up)
                if new_y - BLOCK_SIZE < 0 or Point(x, new_y - BLOCK_SIZE) in self.blocks:
                    return False
            return True
        return False

    def overlapping(self, point):
        return point in self.blocks and point in self.holes

    def player_pos(self):
        return [self.player.x / BLOCK_SIZE, self.player.y / BLOCK_SIZE]

    def block_state(self):
        res = []
        x1 = self.player.x
        y1 = self.player.y
        # UP, DOWN, LEFT, RIGHT
        for block in self.blocks:
            x2 = block.x
            y2 = block.y

            # if the block is inside a hole, then append -1
            if self.overlapping(block):
                res.append(-1)
                res.append(-1)
            else:
                res.append((x1 - x2) / BLOCK_SIZE)
                res.append((y1 - y2) / BLOCK_SIZE)

        return res

    def hole_state(self):
        res = []
        x1 = self.player.x
        y1 = self.player.y
        # UP, DOWN, LEFT, RIGHT
        for hole in self.holes:
            x2 = hole.x
            y2 = hole.y

            if self.overlapping(hole):
                res.append(-1)
                res.append(-1)
            else:
                res.append((x1 - x2) / BLOCK_SIZE)
                res.append((y1 - y2) / BLOCK_SIZE)

        return res
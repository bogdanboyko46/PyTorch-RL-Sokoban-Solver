from http.cookiejar import debug

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

import time


# Initialize pygame modules
pygame.init()

font = pygame.font.Font('arial.ttf', 25)

# Enum for player movement directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


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
        self.render = False
        self.debug_mode = False

        # Screen width and height
        self.w = w
        self.h = h
        # Player
        self.player = None
        self.blocks = None
        self.holes = None
        self.in_hole = 0
        self.paths = None
        self.tot_block_ct = 0
        self.block_hole_pairs = None
        self.num_objects = 2
        # self.moves_made = 0

        # Initialize game window
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Sokoban')

        self.reset()

    def paired(self, point):
        return point in self.block_hole_pairs

    def reset(self):
        x_p = random.randint(0, 8) * BLOCK_SIZE
        y_p = random.randint(0, 8) * BLOCK_SIZE
        self.moves_made = 0
        self.block_hole_pairs = set()
        self.player = Point(x_p, y_p)
        self.in_hole = 0
        self.blocks = set()
        self.holes = set()
        self.paths = dict()

        while len(self.blocks) < self.num_objects:
            x = random.randint(0, 7) * BLOCK_SIZE
            y = random.randint(0, 7) * BLOCK_SIZE

            if Point(x, y) != self.player:
                self.blocks.add(Point(x, y))

        self.tot_block_ct = len(self.blocks)

        while len(self.holes) < self.num_objects:
            x = random.randint(0, 8) * BLOCK_SIZE
            y = random.randint(0, 8) * BLOCK_SIZE

            if Point(x, y) != self.player and Point(x, y) not in self.blocks:
                self.holes.add(Point(x, y))

        for block in self.blocks:
            self.paths[block] = dict()

            for hole in self.holes:
                self.paths[block][hole] = abs(block.x - hole.x) / BLOCK_SIZE + abs(block.y - hole.y) / BLOCK_SIZE

    def update_paths(self, old_pos, new_pos, old_in_hole):

        """
        1. Check if the number of block-hole pairs changed. If so, update tot_reward, add block and hole to dict
        2. If the above condition didn't occur, update paths and return reward based off move made
        """

        # NEW: Block pushed from one hole to another (net in_hole change is 0)
        if old_in_hole == self.in_hole and old_pos in self.block_hole_pairs and new_pos in self.holes:
            self.block_hole_pairs.remove(old_pos)
            self.block_hole_pairs.add(new_pos)

            # hole at old_pos is now free
            for block in self.paths:
                self.paths[block][old_pos] = abs(block.x - old_pos.x) / BLOCK_SIZE + abs(
                    block.y - old_pos.y) / BLOCK_SIZE

            # hole at new_pos is now occupied
            for block in self.paths:
                if new_pos in self.paths[block]:
                    del self.paths[block][new_pos]

            if self.debug_mode:
                print("PUSHED FROM HOLE TO HOLE!")
            return 0

        if old_in_hole < self.in_hole:

            self.block_hole_pairs.add(new_pos)

            # delete the old block's positions
            del self.paths[old_pos]

            # a block was pushed into a hole, delete the hole's dist from each block to essentially ignore it
            for block in self.paths:

                del self.paths[block][new_pos]

            if self.debug_mode:
                print("PUSHED INTO HOLE!")
            return 50

        elif old_in_hole > self.in_hole:
            # a block was pushed off a hole, add the block and the previously paired hole to "paths"

            self.block_hole_pairs.remove(old_pos)
            self.paths[new_pos] = dict()

            for hole in self.holes:

                # skip hole if its occupied by a hole
                if self.paired(hole):
                    continue

                self.paths[new_pos][hole] = abs(new_pos.x - hole.x) / BLOCK_SIZE + abs(new_pos.y - hole.y) / BLOCK_SIZE

            for block in self.paths:

                self.paths[block][old_pos] = abs(block.x - old_pos.x) / BLOCK_SIZE + abs(block.y - old_pos.y) / BLOCK_SIZE

            if self.debug_mode:
                print("PUSHED OFF HOLE")
            return -50


        closest_dist_reached, longest_dist_reached = None, None
        # if none of the above conditions are true, then we will proceed with the dynamic reward system
        self.paths[new_pos] = dict()

        for hole in self.holes:

            # skip hole if occupied by block
            if self.paired(hole):
                continue

            old_dist = self.paths[old_pos][hole]
            new_dist = abs(new_pos.x - hole.x) / BLOCK_SIZE + abs(new_pos.y - hole.y) / BLOCK_SIZE

            self.paths[new_pos][hole] = new_dist
            """
            A small multiplier of 1.5 is applied to pos rewards as equal neg/pos rewards in magnitude generally didn't work
            The '7' denotes the farthest a block can be from a hole in stepping distance
            Keep a constant negative reward of -1 as the block moves away from the hole
            The negative reward of -1 simply based off the conditions in old/new distances creates a conflict when dealing with multiple blocks/holes
            If a block is close to a hole but the tot reward is net negative due to a large amt of holes, it may create redundancy
            """
            if new_dist < old_dist:
                # a closer distance has been acheived
                closest_dist_reached = min(7 if not closest_dist_reached else closest_dist_reached, new_dist)
            else:
                longest_dist_reached = max(1 if not longest_dist_reached else longest_dist_reached, new_dist)

        # delete the old block's positions
        del self.paths[old_pos]

        return self.tot_block_ct * 3 / closest_dist_reached if closest_dist_reached else longest_dist_reached / self.tot_block_ct

    def immovable_block_detect(self):
        # create a dict denoting the number of blocks / holes in each border
        block_ct_borders = [0, 0, 0, 0] # UP, DOWN, LEFT, RIGHT
        hole_ct_borders = [0, 0, 0, 0] # UP, DOWN, LEFT, RIGHT

        for block in self.blocks:
            # hole in occupied
            if block in self.holes:
                continue
            if block in (Point(0, 0),
                         Point(self.w - BLOCK_SIZE, 0),
                         Point(0, self.h - BLOCK_SIZE),
                         Point(self.w - BLOCK_SIZE, self.h - BLOCK_SIZE)):
                return True

            # border presence check
            if block.x == 0:
                block_ct_borders[2] += 1
            elif block.x == self.w - BLOCK_SIZE:
                block_ct_borders[3] += 1
            elif block.y == 0:
                block_ct_borders[0] += 1
            elif block.y == self.h - BLOCK_SIZE:
                block_ct_borders[1] += 1

            # if the block is within the top, bottom, left, or right borders where an unoccupied hole is not available on the same border, the game must end

        # get the border count for each hole
        for hole in self.holes:
            # hole is occupied
            if hole in self.blocks:
                continue

            if hole.x == 0:
                hole_ct_borders[2] += 1
            elif hole.x == self.w - BLOCK_SIZE:
                hole_ct_borders[3] += 1
            elif hole.y == 0:
                hole_ct_borders[0] += 1
            elif hole.y == self.h - BLOCK_SIZE:
                hole_ct_borders[1] += 1

        # compare arrays -> if the # of holes in each border >= corresponding index in hole_ct_borders, there is still a way to win
        for i in range(0, len(block_ct_borders)):
            if block_ct_borders[i] > hole_ct_borders[i]:
                return True

        return False

    def play_step(self, action):
        # TODO: return respective vars: reward, game_over, game_win

        # Handle user input
        # if self.render:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #             quit()

        # action is [up, down, left, right]
        if isinstance(action, (list, tuple, np.ndarray)):
            idx = int(np.argmax(action))
            action = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT][idx]

        self.moves_made += 1

        # check if game is over and collect reward values
        game_over = False

        # initialize reward to -.1, (time constraint) negative reward
        reward = -0.1

        # get old player states
        old_x, old_y = self.player.x, self.player.y
        old_block_hole_pairs = self.in_hole

        # execute move from agent action
        old_pushed_block_pos, new_pushed_block_pos = self._move(action)

        if old_pushed_block_pos and new_pushed_block_pos:
            reward += self.update_paths(old_pushed_block_pos, new_pushed_block_pos, old_block_hole_pairs)

        elif old_x == self.player.x and old_y == self.player.y:
            reward -= 5

        # check if agent completed the game
        if self.in_hole == len(self.holes):
            reward += 200
            game_over = True
            return reward, game_over, True

        # check if agent moved a block into an immovable state
        if self.immovable_block_detect() or self.moves_made > 1600:
            reward -= 5
            game_over = True
            return reward, game_over, False

        if self.render:
            self._update_ui()

        # return
        return reward, game_over, False

    # Moves the player, Returns True if a block is moved too
    def _move(self, direction):
        x = self.player.x
        y = self.player.y
        old_pushed_block_pos = None
        new_pushed_block_pos = None

        if direction == Direction.RIGHT and self.can_move_right():
            if Point(x + BLOCK_SIZE, y) in self.blocks:
                old_pushed_block_pos =  Point(x + BLOCK_SIZE, y)
                if old_pushed_block_pos in self.holes:
                    self.in_hole -= 1
                new_pushed_block_pos = Point(x + BLOCK_SIZE * 2, y)
                if new_pushed_block_pos in self.holes:
                    self.in_hole += 1

            x += BLOCK_SIZE

        elif direction == Direction.LEFT and self.can_move_left():
            if Point(x - BLOCK_SIZE, y) in self.blocks:
                old_pushed_block_pos = Point(x - BLOCK_SIZE, y)
                if old_pushed_block_pos in self.holes:
                    self.in_hole -= 1
                new_pushed_block_pos = Point(x - BLOCK_SIZE * 2, y)
                if new_pushed_block_pos in self.holes:
                    self.in_hole += 1
            x -= BLOCK_SIZE

        elif direction == Direction.DOWN and self.can_move_down():
            if Point(x, y + BLOCK_SIZE) in self.blocks:
                old_pushed_block_pos = Point(x, y + BLOCK_SIZE)
                if old_pushed_block_pos in self.holes:
                    self.in_hole -= 1
                new_pushed_block_pos = Point(x, y + BLOCK_SIZE * 2)
                if new_pushed_block_pos in self.holes:
                    self.in_hole += 1
            y += BLOCK_SIZE

        elif direction == Direction.UP and self.can_move_up():
            if Point(x, y - BLOCK_SIZE) in self.blocks:
                old_pushed_block_pos = Point(x, y - BLOCK_SIZE)
                if old_pushed_block_pos in self.holes:
                    self.in_hole -= 1
                new_pushed_block_pos = Point(x, y - BLOCK_SIZE * 2)
                if new_pushed_block_pos in self.holes:
                    self.in_hole += 1
            y -= BLOCK_SIZE

        self.player = Point(x, y)

        if old_pushed_block_pos and new_pushed_block_pos:
            self.blocks.remove(old_pushed_block_pos)
            self.blocks.add(new_pushed_block_pos)

        return old_pushed_block_pos, new_pushed_block_pos

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

    def player_state(self):
        return [self.player.x / BLOCK_SIZE, self.player.y / BLOCK_SIZE]

    def block_state(self):
        res = []
        x1 = self.player.x
        y1 = self.player.y
        # UP, DOWN, LEFT, RIGHT
        for block in self.blocks:
            # Gets the position of block from player
            x2 = block.x
            y2 = block.y
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
            res.append((x1 - x2) / BLOCK_SIZE)
            res.append((y1 - y2) / BLOCK_SIZE)

        return res
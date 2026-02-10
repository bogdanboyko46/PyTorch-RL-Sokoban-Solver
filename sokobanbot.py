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
        # Screen width and height
        self.w = w
        self.h = h
        self.player = None
        self.blocks = None
        self.holes = None
        self.moves_made = None
        self.in_hole = 0
        self.paths = None

        # Initialize game window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Sokoban')

        self.reset()

    # used when initial game start / when blocks are pushed off a hole
    def add_to_paths(self, block):
        self.paths[block] = dict()
        for hole in self.holes:

            if hole in self.blocks:
                continue

            self.paths[block][hole] = (abs(block.x - hole.x) / BLOCK_SIZE) + (abs(block.y - hole.y) / BLOCK_SIZE)

    def remove_from_paths(self, block):
        if block in self.paths:
            del self.paths[block]

    # Resets the Sokoban game with all new positions
    def reset(self):
        x_p = random.randint(0, 8) * BLOCK_SIZE
        y_p = random.randint(0, 8) * BLOCK_SIZE
        # Moves made, used to reset (if over 650 moves made), and to track records
        self.moves_made = 0
        # Players starting position
        self.player = Point(x_p, y_p)
        self.in_hole = 0
        self.blocks = set()
        self.holes = set()
        # Gets Manhattan distance of block and holes (for rewards)
        self.paths = dict()

        num_blocks = 1;

        while len(self.blocks) < num_blocks:
            x = random.randint(1, 6) * BLOCK_SIZE
            y = random.randint(1, 6) * BLOCK_SIZE

            # If that position isn't already in use
            if Point(x, y) != self.player:
                self.blocks.add(Point(x, y))

        # Generates hole positions
        while len(self.holes) < num_blocks:
            x = random.randint(0, 8) * BLOCK_SIZE
            y = random.randint(0, 8) * BLOCK_SIZE

            # If that position isn't already in use
            if Point(x, y) != self.player and Point(x, y) not in self.blocks:
                self.holes.add(Point(x, y))

        # Finds manhattan path distances
        for block in self.blocks:
            self.add_to_paths(block)

    def replace_path(self, old_pos, new_pos):
        # if pushed into a hole stop tracking that block
        if new_pos in self.holes:
            self.remove_from_paths(old_pos)
            return 0

        if old_pos not in self.paths:
            self.add_to_paths(old_pos)

        # recompute all distances for new_pos fresh
        self.paths[new_pos] = {}

        ct = 0
        old_closest_hole = {"Point": None, "Distance": float("inf")}
        new_closest_hole = {"Point": None, "Distance": float("inf")}

        for hole in self.holes:
            if hole in self.blocks:
                continue

            # don't assume hole key exists for old_pos
            old_dist = self.paths[old_pos].get(hole)
            if old_dist is None:
                old_dist = (abs(old_pos.x - hole.x) / BLOCK_SIZE) + (abs(old_pos.y - hole.y) / BLOCK_SIZE)
                self.paths[old_pos][hole] = old_dist  # keep it consistent going forward

            new_dist = (abs(new_pos.x - hole.x) / BLOCK_SIZE) + (abs(new_pos.y - hole.y) / BLOCK_SIZE)

            ct += 1 if new_dist < old_dist else -1

            if old_dist < old_closest_hole["Distance"]:
                old_closest_hole["Point"] = hole
                old_closest_hole["Distance"] = old_dist

            if new_dist < new_closest_hole["Distance"]:
                new_closest_hole["Point"] = hole
                new_closest_hole["Distance"] = new_dist

            self.paths[new_pos][hole] = new_dist

        # remove old_pos mapping
        self.remove_from_paths(old_pos)

        if ct == 0 and old_closest_hole["Point"] == new_closest_hole["Point"]:
            return 10
        return 10 if ct > 0 else -1


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
        #time.sleep(.05)

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
        # check if game is over and collect reward values
        game_over = False

        # initialize reward to -1, (time constraint) negative reward
        reward = -0.5

        # get old player states
        old_x, old_y = self.player.x, self.player.y
        old_in_hole = self.in_hole

        # execute move from agent action
        old_pos, new_pos = self._move(action)
        self._update_ui()

        if old_x == self.player.x and old_y == self.player.y:
            reward -= 5

        elif old_pos and new_pos:

            # update reward based off if the agent pushed a block closer to any of the holes
            reward += self.replace_path(old_pos, new_pos)

        # check if agent completed the game
        if self.in_hole == len(self.holes):
            reward += 200
            game_over = True
            return reward, game_over, True
        elif self.in_hole != old_in_hole:
            if self.in_hole > old_in_hole:

                self.moves_made = 0
                reward += 100

            else:

                reward -= 10

        # check if agent moved a block into an immovable state
        if self.immovable_block_detect() or self.moves_made > 5000:
            reward -= 5
            game_over = True
            self._update_ui()
            time.sleep(0.2)
            return reward, game_over, False

        # return
        return reward, game_over, False

    # Moves the player, Returns True if a block is moved too
    def _move(self, direction):
        x = self.player.x
        y = self.player.y
        old_pos = None
        new_pos = None

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
            y -= BLOCK_SIZE

        self.player = Point(x, y)
        return old_pos, new_pos

    def _update_ui(self):
        self.display.fill(BLACK)

        moves_str = f"{self.moves_made}"
        text = font.render(moves_str, True, PINK)
        self.display.blit(text, [self.w - self.w/10, 30])
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
                if new_y + BLOCK_SIZE >= self.h or Point(x, new_y + BLOCK_SIZE) in self.blocks:
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
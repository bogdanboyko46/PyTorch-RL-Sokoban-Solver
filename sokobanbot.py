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
    def __init__(self, w=480, h=480):
        # Screen width and height
        self.w = w
        self.h = h
        self.player = None
        self.blocks = None
        self.holes = None
        self.in_hole = 0
        self.paths = None

        # Initialize game window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Sokoban')

        self.reset()

    def reset(self):
        x = random.randint(0, 2) * BLOCK_SIZE
        y = random.randint(0, 2) * BLOCK_SIZE
        self.moves_made = 0
        self.player = Point(x, y)
        self.in_hole = 0
        self.blocks = set()
        self.holes = set()
        self.paths = dict()

        self.blocks.add(Point(BLOCK_SIZE * 2, 3 * BLOCK_SIZE))
        self.holes.add(Point(BLOCK_SIZE * 3, BLOCK_SIZE * 2))

        for block in self.blocks:
            self.paths[block] = dict()

            for hole in self.holes:
                self.paths[block][hole] = (abs(block.x - hole.x) / BLOCK_SIZE) + (abs(block.y - hole.y) / BLOCK_SIZE)

    def replace_path(self, old_pos, new_pos):
        # incremented / decremented based off if a block got closer or farther from each block
        closer_count = 0
        self.paths[new_pos] = dict()

        for hole in self.holes:
            old_dist = self.paths[old_pos][hole]
            new_dist = (abs(new_pos.x - hole.x) / BLOCK_SIZE) + (abs(new_pos.y - hole.y) / BLOCK_SIZE)
            closer_count += 1 if old_dist > new_dist else -1

            # add path
            self.paths[new_pos][hole] = new_dist

        # delete old path
        del self.paths[old_pos]

        if closer_count == 0:
            return 0

        return 5 if closer_count > 0 else -1


    def unmovable_block_detect(self):
        for block in self.blocks:
            if block in self.holes:
                continue
            if block in (Point(0, 0),
                         Point(self.w - BLOCK_SIZE, 0),
                         Point(0, self.h - BLOCK_SIZE),
                         Point(self.w - BLOCK_SIZE, self.h - BLOCK_SIZE)):
                return True
        return False


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

        # check if game is over and collect reward values
        game_over = False

        # initialize reward to -1, (time constraint) negative reward
        reward = -0.5

        # execute move from agent action
        old_pos, new_pos = self._move(action)

        if old_pos and new_pos:
            # update reward based off if the agent pushed a block closer to any of the holes
            reward += self.replace_path(old_pos, new_pos)

        # check if agent completed the game
        if self.in_hole == len(self.holes):
            reward += 50
            game_over = True
            return reward, game_over, True

        # check if agent moved a block into an unmovable state
        if self.unmovable_block_detect() or self.moves_made > 50:
            reward -= 15
            game_over = True
            return reward, game_over, False

        self._update_ui()

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
                moved_block = True
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
                moved_block = True
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
                moved_block = True
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
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

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
        self.in_hole = 0

        # Initialize game window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Sokoban')

        self.reset()

    def reset(self):
        x = random.randint(0, 8) * BLOCK_SIZE
        y = random.randint(0, 8) * BLOCK_SIZE
        self.moves_made = 0
        self.player = Point(x, y)
        self.in_hole = 0
        self.blocks = set()
        while len(self.blocks) < 3:
            x = random.randint(1, 7) * BLOCK_SIZE
            y = random.randint(1, 7) * BLOCK_SIZE
            if not Point(x, y) in self.blocks and Point(x, y) != self.player:
                self.blocks.add(Point(x, y))

        self.holes = set()
        while len(self.holes) < 3:
            x = random.randint(0, 8) * BLOCK_SIZE
            y = random.randint(0, 8) * BLOCK_SIZE
            if not Point(x, y) in self.holes and not Point(x, y) in self.blocks and Point(x, y) != self.player:
                self.holes.add(Point(x, y))
        pass

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

        # get old block in hole state to compare after move is completed
        old_in_hole_ct = self.in_hole

        # get old position state to check if agent attempted to move a block into a wall, into another block, or attempted to move itself into a wall
        old_x, old_y = self.player.x, self.player.y


        # check if game is over and collect reward values
        game_over = False

        # initialize reward to -1, (time constraint) negative reward
        reward = -1

        # execute move from agent action
        if self._move(action):
            reward -= 0.5

        # compare old and new in hole states
        if old_in_hole_ct > self.in_hole:
            reward -= 10
        elif old_in_hole_ct < self.in_hole:
            reward += 10

        # check if agent moved a block into an unmovable state
        if self.unmovable_block_detect() or self.moves_made > 6000:
            reward -= 15
            game_over = True
            return reward, game_over, False

        # check if agent completed the game
        if self.in_hole == len(self.holes):
            reward += 50
            game_over = True
            return reward, game_over, True

        # # compare old and new agent positions
        # if old_x == self.player.x and old_y == self.player.y:
        #     reward -= 5

        # adjacent_to_block = False
        # for block in self.blocks:
        #     if self.adjacent(self.player.x, self.player.y, block.x, block.y):
        #         adjacent_to_block = True
        #         break
        #
        # if adjacent_to_block:
        #     reward += 2
        #     self.frames_without_contact = 0
        # else:
        #     self.frames_without_contact += 1

        # if self.frames_without_contact > 10:
        #     reward -= 5

        # update UI
        self._update_ui()

        # return
        return reward, game_over, False

    # Moves the player, Returns True if a block is moved too
    def _move(self, direction):
        x = self.player.x
        y = self.player.y
        moved_block = False
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
                moved_block = True
            y -= BLOCK_SIZE

        self.player = Point(x, y)
        return moved_block

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

    def block_state(self):
        res = []
        x1 = self.player.x
        y1 = self.player.y
        # UP, DOWN, LEFT, RIGHT
        for block in self.blocks:
            x2 = block.x
            y2 = block.y
            res.append(x2 - x1)
            res.append(y2 - y1)

        return res

    def hole_state(self):
        res = []
        x1 = self.player.x
        y1 = self.player.y
        # UP, DOWN, LEFT, RIGHT
        for hole in self.holes:
            x2 = hole.x
            y2 = hole.y
            res.append(x2 - x1)
            res.append(y2 - y1)

        return res

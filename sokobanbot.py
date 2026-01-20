import time
import pygame
import random
from enum import Enum
from collections import namedtuple

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

        # Initialize game window
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Sokoban')

        self.reset()

    def reset(self):
        x = random.randint(0, 8) * BLOCK_SIZE
        y = random.randint(0, 8) * BLOCK_SIZE
        self.player = Point(x, y)

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

    def play_step(self):
        # TODO: return respective vars: reward, game_over, game_win

        # Handle user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self._move(Direction.LEFT)
                elif event.key == pygame.K_RIGHT:
                    self._move(Direction.RIGHT)
                elif event.key == pygame.K_UP:
                    self._move(Direction.UP)
                elif event.key == pygame.K_DOWN:
                    self._move(Direction.DOWN)
                elif event.key == pygame.K_r:
                    return True

        # Update display
        return self._update_ui()

    def _update_ui(self):
        num_holes = len(self.blocks)
        num_comp = 0
        # Clear screen
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
                num_comp += 1
            elif h_pt == p_pt:
                pygame.draw.rect(self.display, CYAN,
                                 pygame.Rect(h_pt.x, h_pt.y, BLOCK_SIZE, BLOCK_SIZE))
            else:
                pygame.draw.rect(self.display, WHITE,
                                 pygame.Rect(h_pt.x, h_pt.y, BLOCK_SIZE, BLOCK_SIZE))

        if num_holes == num_comp:
            text = font.render("Complete!", True, PINK)
            self.display.blit(text, [self.w / 2 - 100, self.w / 2])
            pygame.display.flip()
            time.sleep(3)
            return True
        # Update the screen
        pygame.display.flip()
        return False

    def _move(self, direction):
        x = self.player.x
        y = self.player.y

        if direction == Direction.RIGHT and self.can_move_right():
            if Point(x + BLOCK_SIZE, y) in self.blocks:
                self.blocks.remove(Point(x + BLOCK_SIZE, y))
                self.blocks.add(Point(x + BLOCK_SIZE * 2, y))
            x += BLOCK_SIZE
        elif direction == Direction.LEFT and self.can_move_left():
            if Point(x - BLOCK_SIZE, y) in self.blocks:
                self.blocks.remove(Point(x - BLOCK_SIZE, y))
                self.blocks.add(Point(x - BLOCK_SIZE * 2, y))
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN and self.can_move_down():
            if Point(x, y + BLOCK_SIZE) in self.blocks:
                self.blocks.remove(Point(x, y + BLOCK_SIZE))
                self.blocks.add(Point(x, y + BLOCK_SIZE * 2))
            y += BLOCK_SIZE
        elif direction == Direction.UP and self.can_move_up():
            if Point(x, y - BLOCK_SIZE) in self.blocks:
                self.blocks.remove(Point(x, y - BLOCK_SIZE))
                self.blocks.add(Point(x, y - BLOCK_SIZE * 2))
            y -= BLOCK_SIZE

        self.player = Point(x, y)

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
            res.append(y1 > y2)
            res.append(y1 < y2)
            res.append(x1 > x2)
            res.append(x1 < x2)
        return res

    def hole_state(self):
        res = []
        x1 = self.player.x
        y1 = self.player.y
        # UP, DOWN, LEFT, RIGHT
        for hole in self.holes:
            x2 = hole.x
            y2 = hole.y
            res.append(y1 > y2)
            res.append(y1 < y2)
            res.append(x1 > x2)
            res.append(x1 < x2)
        return res

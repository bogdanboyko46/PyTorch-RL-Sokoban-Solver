# w = wall (unmovable)
# b = block (movable, if grid[i][j] says b then the block is on an empty space, not a hole)
# h = free hole (put block in it, but blocks not already in)
# e = empty (no obstacle in the way)
# c = block in hole (theres a hole that currently has block in it)
import random

# 2d character array will consist of the above 5 objects
grid = [[]]
# tuple of players coordinates on grid
player = []
# blocks in game
blocks = 0
# blocks in the hole
blocksComp = 0

def main():
    solve_with_random()

def solve_with_random():
    global grid, player, blocksComp, blocks, player
    blocks = 1
    blocksComp = 0
    player = [0, 1]
    grid = [['e', 'e', 'e'], ['e', 'b', 'e'], ['e', 'e', 'h']]
    while blocksComp != blocks:
        blocksComp = 0
        blocks = 1
        grid = [['e', 'e', 'e'], ['e', 'b', 'e'], ['e', 'e', 'h']]
        player = [0, 1]
        # Solves with random moves
        for i in range(1,80):
            randnum = random.randint(1,4)
            match randnum:
                case 1:
                    move_left()
                case 2:
                    move_right()
                case 3:
                    move_down()
                case 4:
                    move_up()
            if blocksComp == blocks:
                break
    print_grid()
    print(player)


def move_up() -> bool:
    return move_vertical(-1)
def move_down() -> bool:
    return move_vertical(1)
def move_left() -> int:
    return move_horizontal(-1)
def move_right() -> int:
    return move_horizontal(1)
def print_grid():
    global grid
    global player
    y = player[0]
    x = player[1]
    cur = grid[y][x]
    grid[y][x] = 'P'
    for g in grid:
        print(g)
    grid[y][x] = cur

def move_vertical(dir: int) -> bool:
    # player cur pos
    global player, blocksComp
    y = player[0]
    x = player[1]
    newy = y + dir
    fary = y + (dir * 2)
    gridlen = len(grid)

    # if out of bounds or above object is a wall return false (not possible to move)
    if newy < 0 or newy >= gridlen or grid[newy][x] == 'w':
        return False

    # if above object is empty space or a free hole
    elif grid[newy][x] == 'e' or grid[newy][x] == 'h':
        player = [newy, x]

    # if above object is a block, or a hole with block in it
    elif grid[newy][x] == 'c' or grid[newy][x] == 'b':
        # if the next above object is oob or not vacant space
        if fary < 0 or fary >= gridlen or grid[fary][x] == 'w' or grid[fary][x] == 'c' or grid[fary][x] == 'b':
            return False

        # updates grid coord above (where the block is being pushed)
        # empty -> block
        if grid[fary][x] == 'e':
            grid[fary][x] = 'b'
        # hole -> complete (because block went into hole so this grid is 'complete' (can still change))
        else:
            grid[fary][x] = 'c'
            blocksComp += 1

        # updates new coord (where block was and players moving to)
        # block -> empty (because block has been moved)
        if grid[newy][x] == 'b':
            grid[newy][x] = 'e'
        # complete -> hole (block was pushed from hole)
        else:
            grid[newy][x] = 'h'
            blocksComp -= 1

        # update player location
        player = [newy, x]
    return True

def move_horizontal(dir: int) -> bool:
    global player, blocksComp
    y = player[0]
    x = player[1]
    newx = x + dir
    farx = x + (dir * 2)
    gridlen = len(grid[0])


    # if out of bounds or above object is a wall return false (not possible to move)
    if newx < 0 or newx >= gridlen or grid[y][newx] == 'w':
        return False

    # if above object is empty space or a free hole
    elif grid[y][newx] == 'e' or grid[y][newx] == 'h':
        player = [y, newx]

    # if above object is a block, or a hole with block in it
    elif grid[y][newx] == 'c' or grid[y][newx] == 'b':
        # if the next above object is oob or not vacant space
        if farx < 0 or farx >= gridlen or grid[y][farx] == 'w' or grid[y][farx] == 'c' or grid[y][farx] == 'b':
            return False

        # updates grid coord above (where the block is being pushed)
        # empty -> block
        if grid[y][farx] == 'e':
            grid[y][farx] = 'b'
        # hole -> complete (because block went into hole so this grid is 'complete' (can still change))
        else:
            grid[y][farx] = 'c'
            blocksComp += 1

        # updates new coord (where block was and players moving to)
        # block -> empty (because block has been moved)
        if grid[y][newx] == 'b':
            grid[y][newx] = 'e'
        # complete -> hole (block was pushed from hole)
        else:
            grid[y][newx] = 'h'
            blocksComp -= 1

        # update player location
        player = [y, newx]
    return True

main()



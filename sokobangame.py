# w = wall (unmovable)
# b = block (movable, if grid[i][j] says b then the block is on an empty space, not a hole)
# h = free hole (put block in it, but blocks not already in)
# e = empty (no obstacle in the way)
# c = block in hole (theres a hole that currently has block in it)

# 2d character array will consist of the above 5 objects
grid = [[]]
# tuple of players coordinates on grid
player = []

def main():
    global grid
    global player
    grid = [['e', 'e', 'e'], ['e', 'b', 'e'], ['e', 'e', 'e']]
    player = [0, 1]
    print("before")
    print(f"player location {player}")
    for g in grid:
        print(g)
    move_down()
    print(f"player location {player}")
    move_left()
    print(f"player location {player}")
    move_down()
    print(f"player location {player}")
    move_right()
    print("after")
    print(f"player location {player}")
    for g in grid:
        print(g)
def move_up() -> bool:
    return move_vertical(-1)
def move_down() -> bool:
    return move_vertical(1)
def move_left() -> int:
    return move_horizontal(-1)
def move_right() -> int:
    return move_horizontal(1)

def move_vertical(dir: int) -> bool:
    # player cur pos
    global player
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

        # updates new coord (where block was and players moving to)
        # block -> empty (because block has been moved)
        if grid[newy][x] == 'b':
            grid[newy][x] = 'e'
        # complete -> hole (block was pushed from hole)
        else:
            grid[newy][x] = 'h'

        # update player location
        player = [newy, x]
    return True

def move_horizontal(dir: int) -> bool:
    global player
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

        # updates new coord (where block was and players moving to)
        # block -> empty (because block has been moved)
        if grid[y][newx] == 'b':
            grid[y][newx] = 'e'
        # complete -> hole (block was pushed from hole)
        else:
            grid[y][newx] = 'h'

        # update player location
        player = [y, newx]
    return True

main()



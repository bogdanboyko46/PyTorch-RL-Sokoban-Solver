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
    grid = [['e', 'h', 'e'], ['e', 'b', 'e'], ['e', 'e', 'e']]
    player = [2, 1]
    print("before")
    print(f"player location {player}")
    for g in grid:
        print(g)
    move_up()
    print("after")
    print(f"player location {player}")
    for g in grid:
        print(g)
def move_up() -> bool:
    # player cur pos
    global player
    y = player[0]
    x = player[1]

    # the soon to be y position (-1 because moving up)
    newy = y - 1

    # The y grid 2 grids above current (needed if pushing a block up)
    abovey = y-2

    # if out of bounds or above object is a wall return false (not possible to move)
    if newy < 0 or grid[newy][x] == 'w':
        return False

    # if above object is empty space or a free hole (moves up)
    elif grid[newy][x] == 'e' or grid[newy][x] == 'h':
        player = [newy, x]

    # if above object is a block, or a hole with block in it
    elif grid[newy][x] == 'c' or grid[newy][x] == 'b':
        # if the next above object is oob or not vacant space
        if abovey < 0 or grid[abovey][x] == 'w' or grid[abovey][x] == 'c' or grid[abovey][x] == 'b':
            return False

        # updates grid coord above (where the block is being pushed)
        # empty -> block
        if grid[abovey][x] == 'e':
            grid[abovey][x] = 'b'
        # hole -> complete (because block went into hole so this grid is 'complete' (can still change))
        else:
            grid[abovey][x] = 'c'

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

main()



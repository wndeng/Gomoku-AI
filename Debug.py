# Copyright @ William Deng, 2018


def show_grid(grid):
    # Used for debugging game as well as MTCS
    print("Current State\n")
    R = grid.shape[0]
    C = grid.shape[1]
    print("   ", end="")
    for i in range(0, C):
        if i >= 9:
            print(i+1, "", end="")
        else:
            print(i+1, " ", end="")
    print("\n")
    for i in range(0, R):
        if i >= 9:
            print(i+1, "", end="")
        else:
            print(i+1, " ", end="")
        for j in range(0, C):
            print(str(grid[i,j]), " ",end="")
        print("\n")
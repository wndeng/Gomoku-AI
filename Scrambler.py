# Copyright @ William Deng, 2018
import random as rng
import copy

def scramble(board_state):
    # Exploits board invariance of gomoku, and 'enhance' the data by adding in reflection/rotation.

    new_board = copy.deepcopy(board_state)
    raw_board = new_board.state
    temp_board = copy.copy(raw_board)
    choices = [0, 1, 2, 3, 4, 5, 6, 7]
    pick = rng.SystemRandom().choice(choices)
    temp = '.'
    if pick == 0: # Return as is
        return board_state

    elif pick == 1: # Flip (across y axis)
        for i in range(0, 19):
            for j in range(0, 10):
                temp = raw_board[i, j]
                raw_board[i, j] = raw_board[i, 18 - j]
                raw_board[i, 18 - j] = temp
        return new_board

    elif pick == 2: # Rotate 90 degrees cw
        for i in range(0, 19):
            for j in range(0, 19):
                raw_board[i, j] = temp_board[18 - j, i]
        return  new_board

    elif pick == 3: # Rotate 90 degrees cw + flip
        for i in range(0, 19):
            for j in range(0, 19):
                raw_board[18 - i, j] = temp_board[18 - j, i]
        return  new_board

    elif pick == 4: # Rotate 180 degrees cw
        for i in range(0, 19):
            for j in range(0, 19):
                raw_board[i, 18 - j] = temp_board[18 - i, j]
        return  new_board

    elif pick == 5: # Rotate 180 degrees cw + flip
        for i in range(0, 19):
            for j in range(0, 19):
                raw_board[i, 18 - j] = temp_board[18 - i, 18 - j]
        return  new_board

    elif pick == 6: # Rotate 270 degrees cw
        for i in range(0, 19):
            for j in range(0, 19):
                raw_board[i, j] = temp_board[j, 18 - i]
        return  new_board

    elif pick == 7: # Rotate 270 degrees cw + flip
        for i in range(0, 19):
            for j in range(0, 19):
                raw_board[i, 18 - j] = temp_board[j, 18 - i]
        return  new_board

    else:
        return  new_board
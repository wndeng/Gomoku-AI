# Copyright @ William Deng && Miller Wang, 2018

# Implemented using the rule PRO. Black must go first, and pick middle point (9,9). White must go second,
# and pick somewhere immediately next to the middle point. Black goes next, and pick somewhere at least 2
# points away from the middle point
# Rules from site: http://www.vcpr.cz/en/help-and-rules/gomoku-rules/

import numpy as np
import Debug as db
import Encoding as Encode
import copy


class BoardState:
    # Abstraction for game state

    def __init__(self, raw_board, encoding, valid_mask, count):
        self.state = raw_board # array representation of board
        self.encoding = encoding # encoding for the state
        self.valid_mask = valid_mask # Mask
        self.count = count # Count
        self.game_won = 0 # if current state is a terminal state


class GomokuGame:
    # Game implementation for Gomoku
    def __init__(self, size, debug):
        # import numpy as np
        if size != 15 and size != 19:
            raise ValueError('Size must be 15 or 19')

        self.board_size = size
        self.board = np.chararray((size, size), unicode = True)
        mid = int((self.board_size-1)/2)
        self.board[:,:] = '.'
        self.board[mid,mid] = 'X'
        self.game_state = "active"
        self.Debug_flag = debug

    # Functions for user input play
    def get_count(self, board, r, c, player, direction, count):
        if (0 <= r < self.board_size and 0 <= c < self.board_size) and \
           ((player == 0 and board[r,c] == 'X') or
            (player == 1 and board[r,c] == 'O')):
            count += 1
            if direction == "U":
                return self.get_count(board, r-1, c, player, "U", count)

            elif direction == "D":
                return self.get_count(board, r+1, c, player, "D", count)

            elif direction == "L":
                return self.get_count(board, r, c-1, player, "L", count)

            elif direction == "R":
                return self.get_count(board, r, c+1, player, "R", count)

            elif direction == "UL":
                return self.get_count(board, r-1, c-1, player, "UL", count)

            elif direction == "UR":
                return self.get_count(board, r-1, c+1, player, "UR", count)

            elif direction == "LL":
                return self.get_count(board, r+1, c-1, player, "LL", count)

            elif direction == "LR":
                return self.get_count(board, r+1, c+1, player, "LR", count)
            else:
                print("Fatal Error, direction unknown") # Should not happen if get_count is privately accessed
                exit(10)

        else :
            return count

    def game_won(self, board, move, player):
        r, c = move
        horizontal_count = 1 + self.get_count(board, r, c-1, player, "L", 0) + \
                               self.get_count(board, r, c+1, player, "R", 0)
        vertical_count = 1 + self.get_count(board, r-1, c, player, "U", 0) + \
                                self.get_count(board, r+1, c, player, "D", 0)
        positive_diag_count = 1 + self.get_count(board, r-1, c+1, player, "UR", 0) + \
                                self.get_count(board, r+1, c-1, player, "LL", 0)
        negative_diag_count = 1 + self.get_count(board, r-1, c-1, player, "UL", 0) + \
                                self.get_count(board, r+1, c+1, player, "LR", 0)

        if horizontal_count == 5 or vertical_count == 5 or \
           positive_diag_count == 5 or negative_diag_count == 5:
            return True
        else:
            return False

    def get_game_state(self):
        return self.board

    # Functions for evaluation and competitive play
    def make_move(self, player, r, c):
        # import Debug as db
        if self.Debug_flag == 1:
            valid = False
            r = -1
            c = -1
            while not valid:
                move = input("")
                r, c = move.split(",")
                r = int(r) - 1
                c = int(c) - 1 # Board index starts at 1, array index starts at 0
                if r < self.board_size and r >= 0 & \
                        c < self.board_size and c >= 0:
                    valid = True

                if self.board[r, c] != '.':
                    print("Current position occupied")
                    valid = False
                    continue

                if player == -2:
                    mid = int((self.board_size-1)/2)
                    if not (mid-1 <= r <= mid+1) or \
                        not (mid-1 <= c <= mid+1):
                        print("First move of White must place immediately beside mid point")
                        valid = False
                        continue

                if player == -1:
                    mid = int((self.board_size-1)/2)
                    if (mid-3 < r < mid+3) and \
                        (mid-3 < c < mid+3):
                        print("Second move of Black must place greater than a distance of 3 from the mid point")
                        valid = False
                        continue
        else:
            r -= 1
            c -= 1
        if player == 0 or player == -1:
            self.board[r,c] = 'X'
        else:
            self.board[r,c] = 'O'

        db.show_grid(self.get_game_state())

        return r,c

    def start_game(self):
        # import Debug as db

        if self.Debug_flag == 0:
            return 0
        else:
            # 1 = Black, 2 = White
            print("Starting new Game!")
            db.show_grid(self.board)
            print("White move: ")
            self.make_move(-2, -1, -1)
            print("Black move: ")
            self.make_move(-1, -1, -1)
            player = 2
            i = 1
            while i <= self.board_size **2:
                if player == 1:
                    print("Black Move: ", end="")
                    if self.game_won(self.board, self.make_move(1, -1, -1), 1):
                        print("Player 1 won on move", i)
                        exit(1)
                    else:
                        player = 2
                else:
                    print("White Move: ", end="")
                    if self.game_won(self.board, self.make_move(2, -1, -1), 2):
                        print("Player 2 won on move", i)
                        exit(2)
                    else:
                        player = 1
                i += 1

            print("Game is a tie") # Total moves equal entire board space, so game cannot proceed and its a tie
            exit(0)

    # Functions for MCTS

    def get_empty_board_state(self):
        # Return initial game state. First move is fixed. Requires special mask

        # 1 = black, 'X'; 2 = white, 'O'
        board = np.chararray((19, 19), unicode = True)
        mid = int((self.board_size-1)/2)
        board[:, :] = '.'
        board[mid,mid] = 'X'

        mask = np.zeros((19,19), dtype=np.int32) # Mask all moves not immediately beside the middle 'X'
        for i in range(mid - 1, mid + 2):
            for j in range(mid - 1, mid + 2):
                mask[i,j] = 1
        mask[mid,mid] = 0
        count = 1
        code = Encode.encode(board, count)
        board.game_won = 0
        return BoardState(board, code, mask, count)

    def get_second_board_state(self, first_board_state, move):
        # Return second game state. Requires special mask

        r,c = move
        second_board_state = copy.deepcopy(first_board_state)
        mask = np.ones((19,19), dtype=np.int32) # Mask all moves within 3 moves of the middle 'X'
        mid = int((self.board_size - 1) / 2)
        for i in range(mid - 3, mid + 4):
            for j in range(mid - 3, mid + 4):
                mask[i,j] = 0
        second_board_state.valid_mask = mask
        second_board_state.state[r, c] = 'O'
        second_board_state.count += 1
        second_board_state.encoding = Encode.encode(second_board_state.state, second_board_state.count)
        return second_board_state

    def get_third_board_state(self, second_board_state, move):
        # Return third game state. Requires new mask

        r,c = move
        second_board_state.state[r, c] = 'X'
        third_board_state = copy.deepcopy(second_board_state)
        mask = np.ones((19,19), dtype=np.int32) # Unmask all positions except for occupied
        for i in range(0, 19):
            for j in range(0, 19):
                if third_board_state.state[i,j] != '.':
                    mask[i,j] = 0
        third_board_state.valid_mask = mask
        third_board_state.count += 1
        third_board_state.encoding = Encode.encode(third_board_state.state, third_board_state.count)
        return third_board_state

    def get_new_board_state(self, board, move, player):
        # Returns new game state depending on the current state 'board'

        new_board = copy.deepcopy(board)
        if move == (-1, -1):
            return new_board
        if new_board.count == 1:
            return self.get_second_board_state(new_board, move)
        elif new_board.count == 2:
            return self.get_third_board_state(new_board,move)
        else:
            # Once past the 3rd state, the mask is updated following the same pattern.

            r,c = move
            if self.game_won(new_board.state, move, player):
                new_board.game_won = 1
            if player == 0:
                new_board.state[r, c] = 'X'
            else:
                new_board.state[r, c] = 'O'

            new_board.valid_mask[r, c] = 0
            new_board.count += 1
            new_board.encoding = Encode.encode(new_board.state, new_board.count)

        return new_board




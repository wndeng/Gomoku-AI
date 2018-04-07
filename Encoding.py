# Copyright @ William Deng, 2018


def encode(state, count):
    # String encoding using 361 characters. middle point is always 'X', and thus ignored. A count
    # of number of pieces on the board is appended at the front to quickly compare state encodings
    # and the rest of the encodings are board positions starting from the middle expanding outwards,
    # since games tend to finish within about 60 moves and PRO rule focuses game in the middle of the board

    code = str(count)
    i = 8
    j = 11
    while i >= 0:
        for ind in range(i, j):
            code += state[i, ind]
            code += state[j - 1, ind]

        for ind2 in range(i + 1, j - 1):
            code += state[ind2, i]
            code += state[ind2, j - 1]
        i -= 1
        j += 1
    return code

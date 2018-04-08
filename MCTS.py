# Copyright @ William Deng, 2018

import numpy as np
import Constant_Parameters as Param
import Game as G
import Debug as db
import copy
import Scrambler as Mixed_Signal_Generator


class MCTS_state:
    # This is the abstraction for state when referenced by MCTS. It includes for each state:

    # parent state: Which state did it come from

    # action_value: Calculated action value from neural network

    # MCTS_data: Data that contains three fields: State, NN_input, and player. player field is used to
    # decide sign of z value in data output once a MCTS_state terminates the game

    # visit_total: This is used for the debug function as a checksum for edges with respect to total simulations,
    # and it is also used in equations that determine which move to make and which path to simulate

    # edges: All transitions that can move this game state to another valid game state. Transition is abstracted by the
    # "Edge" class

    # mask: This contains the mask that is associated with this state's valid transitions. It is used later to place
    # each edge in the correct positions.

    def __init__(self, this_board_state, parent_state, model, player):
        self.this_board_state = copy.deepcopy(this_board_state)
        self.parent_state = copy.deepcopy(parent_state)
        self.action_value = -1
        self.MCTS_data = []
        self.value_data = []
        self.player = player
        self.visit_total = 0
        scrambled_board_state1 = Mixed_Signal_Generator.scramble(this_board_state)
        scrambled_board_state2 = Mixed_Signal_Generator.scramble(this_board_state)
        NN_input = np.zeros((1, 2, 19, 19), dtype = np.int32) # This is input that is used to generate prediction data

        # This adds scrambled board states as the actual input for training, which will make actual training more
        # robust because these states are game-wise identical
        # first NN input is for this state, second NN input is a new set of data created by reversing the placement
        # of white and black input layers, and reversing the final z value. The input state is re-scrambled.
        NN_input_scrambled = np.zeros((1, 2, 19, 19), dtype=np.int32)

        # Assign ind for layers. The player's own layer is on top (ind = 0)
        if player == 1:
            black = 0
            white = 1
        else: # player == 0
            black = 1
            white = 0

        for i in range(0, 19): # Create input for Neural Network
            for j in range(0, 19):
                if this_board_state.state[i,j] == 'X':
                    NN_input[0, black, i, j] = 1
                if this_board_state.state[i,j] == 'O':
                    NN_input[0, white, i, j] = 1
                if scrambled_board_state1.state[i,j] == 'X':
                    NN_input_scrambled[0, black, i, j] = 1
                if scrambled_board_state1.state[i,j] == 'O':
                    NN_input_scrambled[0, white, i, j] = 1


        value, prob = model.predict(NN_input)

        self.NN_input = NN_input_scrambled
        self.edge_prob = prob.flatten()
        self.edges = []
        self.action_value = value.flatten()
        mask = this_board_state.valid_mask.flatten()
        self.mask = mask
        sum = 0

        # Apply a softmax to the unmasked (valid) edges, and save them
        for i in range(0, len(self.edge_prob)):
            if mask[i] == 1:
                sum += np.exp(self.edge_prob[i])
        for i in range(0, len(self.edge_prob)):
            if mask[i] == 1:
                self.edges.append(Edge(np.exp(self.edge_prob[i]) / sum, i))

    def parent_encoding(self):
        return self.parent_state.encoding

    def self_encoding(self):
        return self.this_board_state.encoding

    def get_best_move(self, is_root):
        # This function returns the 'best move based on chosen edge with highest max(Q+U)

        if self.this_board_state.count == 361: # Game board is full
            return -1, (-1, -1)

        # Implementation of equation in literature
        DIR = np.random.dirichlet([Param.DIR_CONSTANT] * len(self.edges))
        QU_vec = []
        for i in range(0, len(self.edges)):
            Q = self.edges[i].mean_action_value
            U = Param.CPUCT * self.edges[i].prior_prob * np.sqrt(self.visit_total)/(1 + self.edges[i].visit_count)
            if is_root:
                U = U * (DIR[i]*Param.EPSILON + self.edges[i].prior_prob * (1 - Param.EPSILON))

            QU_vec.append(Q+U)

        # Pick an edge with the maximum Q + U value. If there are more than 1 edge with max Q + U value
        # randomly choose one.
        QU_arr = np.array(QU_vec)
        available_edges = np.where(QU_arr == max(QU_arr))[0]
        chosen_edge = np.random.choice(available_edges)
        ind = self.edges[chosen_edge].grid_ind
        r = np.int32(np.floor(ind / 19))
        c = ind - r * 19
        return chosen_edge, (r, c)


class Edge:
    # This class is the abstraction for transition. Each state has possible moves, and each move is an edge, from the
    # current state to a new state reached by placing the current player's piece on a valid position on the board

    # It contains:
    # visit_count: How many times this edge have been used

    # total_action_value: Total action value from states reached through this edge

    # mean_action_value: Action value averaged by total visit count

    # prior_prob: The probability assigned to this edge from current state

    # grid_ind: The index in a flattened representation of the game state (1 x 361) that links a board position
    # to this edge

    def __init__(self, prob, ind):
        self.visit_count = 0
        self.total_action_value = 0.0
        self.mean_action_value = 0.0
        self.prior_prob = prob
        self.grid_ind = ind

    def update_edge(self, action_value):
        # Backwards pass of MCTS

        self.visit_count += 1
        self.total_action_value += action_value
        self.mean_action_value = self.total_action_value / self.visit_count


class MCTS:
    # This is the abstraction for MCTS, which handles all simulations, and interfaces with the game to generate training
    # data according to specified parameters in Constant_Parameters.py
    # It contains:
    # model: Neural Network model

    # state_tree: Keeps track of states in the tree

    # game_instance: An instance of Gomoku, without error checks, declaring wins, etc. Provides game API as a
    # 'black box' that can update game state, given the move and player, and state if the game is won

    # sim_count: Number of simulations to run according to Constant_Parameters.py

    def __init__(self, model, sim_count):
        self.model = model
        self.state_tree = []
        self.game_instance = G.GomokuGame(19, 0)
        self.data_set = []
        self.sim_count = sim_count

    def add_MCTS_state(self, MCTS_state):
        # Append a state to the current state tree
        self.state_tree.append(MCTS_state)

    def simulate(self, root_board_state, player, tau):
        # Conduct MCTS simulations

        initial_player = player
        sim_count = self.sim_count
        root_exist = False

        # if Root is in the state tree, find it in there, and use its data from previous simulations
        for i in range(0, len(self.state_tree)):
            if root_board_state.encoding == self.state_tree[i].self_encoding():
                root_MCTS_state = self.state_tree[i]
                root_exist = True

        # Rarely, it is possible by noise that an unexplored edge be chosen, and thus if the root is unexplored,
        # initiate it and add it to state tree
        if not root_exist:
            root_MCTS_state = MCTS_state(root_board_state, root_board_state, self.model, player)
            self.add_MCTS_state(root_MCTS_state)

        # Do a single simulation with the root, with is_root set to True to enable dirichlet noise. Then, search for
        # a leaf (unexplored) node. If the node is already explored (inside state_tree), then append corresponding edge
        # ind and state to the simulation_state_list. This list keeps track of what states and edges were used in this
        # simulation. Thus, the state and edge information are readily available at each simulation step.
        for j in range(0, sim_count):
            simulation_state_list = []
            stop = False
            current_board_state = root_MCTS_state.this_board_state
            ind, move = root_MCTS_state.get_best_move(is_root = True)
            player = initial_player
            simulation_state_list.append((root_MCTS_state, ind))
            while not stop:
                match_found = False
                next_board_state = self.game_instance.get_new_board_state(current_board_state, move, player)

                for i in range(0, len(self.state_tree)): # Keep moving until leaf node
                    if next_board_state.encoding == self.state_tree[i].self_encoding():
                        ind, move = self.state_tree[i].get_best_move(is_root = False)
                        current_board_state = next_board_state
                        match_found = True
                        player = self.other(player)
                        previous_MCTS_state = self.state_tree[i]
                        simulation_state_list.append((previous_MCTS_state, ind))

                # Add leaf node to state_tree, do backwards pass and update all edges in simulation_state_list
                if not match_found:
                    new_MCTS_state = MCTS_state(next_board_state, current_board_state, self.model, player)
                    self.add_MCTS_state(new_MCTS_state)
                    if next_board_state.game_won:
                        new_MCTS_state.action_value = 1
                        self.state_tree.remove(new_MCTS_state)

                    # Assign final score depending on the initial player if game is won by current player
                    if player != initial_player:
                        who = -1.0
                    else:
                        who = 1.0
                    for i in range(0, len(simulation_state_list)):
                        edge_ind = simulation_state_list[i][1]
                        simulation_state_list[i][0].edges[edge_ind].update_edge(who*new_MCTS_state.action_value)
                        simulation_state_list[i][0].visit_total += 1
                        who *= -1.0
                    stop = True

        k = 0
        edge_vector = []

        # Put edge data of root into a list
        for i in range(0, 361):
            if root_MCTS_state.mask[i] == 1:
                root_MCTS_state.MCTS_data.append(root_MCTS_state.edges[k].visit_count/root_MCTS_state.visit_total)
                edge_vector.append(root_MCTS_state.edges[k].visit_count/root_MCTS_state.visit_total)
                k += 1
            else:
                root_MCTS_state.MCTS_data.append(0.0)

        self.add_MCTS_data(root_MCTS_state) # add data in

        # Use tau do decide how to select the best move
        if tau == 0:
            edge_ind_vector = np.random.multinomial(1, edge_vector)
            edge_ind = np.where(edge_ind_vector == 1)[0][0]
        else: # tau = 1
            edge_arr = np.array(edge_vector)
            edge_ind_vector = np.where(edge_arr == max(edge_arr))[0]
            edge_ind = np.random.choice(edge_ind_vector)

        decider_ind = root_MCTS_state.edges[edge_ind].grid_ind
        r = np.int32(np.floor(decider_ind / 19))
        c = decider_ind - r * 19

        # Remove the root from state_tree since it is no loner needed
        self.state_tree.remove(root_MCTS_state)
        return r, c

    def other(self, player):
        if player == 1:
            return 0
        else:
            return 1

    def new_root(self, root, edge, player):
        # Get the new root state based on chosen edge and current state
        return self.game_instance.get_new_board_state(root, edge, player)

    def add_MCTS_data(self, state):
        self.data_set.append((copy.copy(state.NN_input), copy.copy(state.MCTS_data), copy.copy(state.player)))

    def run(self):
        # Run the MCTS, set tau according to Constant_Parameters.py, and process data to be returned

        tau_timer = Param.TEMP_THRESHOLD
        steps_count = 0
        player = 1
        tau = -1
        previous_board_state = self.game_instance.get_empty_board_state()
        next_board_state = previous_board_state
        new_MCTS_state = MCTS_state(next_board_state, previous_board_state, self.model, player)
        self.add_MCTS_state(new_MCTS_state)

        is_tie = False
        while not (next_board_state.game_won or is_tie):
            # early in the game, tau = 1 encourages exploration such that chance of each edge being selected
            # is proportional to its visit count
            # After a set number of moves, tau is set at 0, which deterministically set node
            # to be explored after MCTS simulations as chosen from nodes with the highest visit count

            steps_count += 1
            if tau_timer > 0:
                tau = 1
                tau_timer -= 1
            if tau_timer == 0:
                tau = 0
            edge = self.simulate(next_board_state, player, tau)
            child_board_state = self.new_root(next_board_state, edge, player)
            self.prune_state_tree(next_board_state, child_board_state)
            player = self.other(player)
            next_board_state = child_board_state

            # If nobody won and game board is full, then it is a tie
            if next_board_state.count == 361 and not next_board_state.game_won:
                is_tie = True

        db.show_grid(next_board_state.state) # This shows the terminal state of the game

        # Assign final z score depending on who won
        winner = self.other(player) # Player switched before checking game condition, so switch back
        packed_data = []
        if winner == 1:
            score = 1.0
        else: # winner == 0
            score = -1.0

        if is_tie: # Ignore winner, assign 0 to all z values
            score = 0.0
        # Every move creates 2 states, with score reversed. This means every odd state must have opposite score
        # with respect to the previous even state.
        for i in range(0, int(len(self.data_set)/2)):
            packed_data.append((copy.copy(self.data_set[i][0]), copy.copy(self.data_set[i][1]), score))
            packed_data.append((copy.copy(self.data_set[i+1][0]), copy.copy(self.data_set[i+1][1]), score* -1.0))
            score *= -1.0
        return packed_data

    def prune_state_tree(self, parent_board_state, child_board_state):
        # Remove unused branches. Because the search space is not quantized, there is the possibility of having
        # cycles in the tree, and thus recursively deleting states might end up deleting a state that has also been
        # explored by the new root, which will cause problems
        for item in self.state_tree:
            if item.parent_encoding() == parent_board_state.encoding and \
               child_board_state.encoding != item.self_encoding():
                self.state_tree.remove(item)

    def print_state_tree(self):
        # Function can be used to see the state of current stored states in the master state list.
        # Used for debugging.

        print("printing state tree")
        print("total number of states: ", len(self.state_tree))
        print("________________________________")
        for i in range(0, len(self.state_tree)):
            print("state " + str(i))
            db.show_grid(self.state_tree[i].this_board_state.state)
            print("state total edge visited: ", self.state_tree[i].visit_total)
            print("state action value: ", self.state_tree[i].action_value)
            print("state total edges:", len(self.state_tree[i].edges))
            print("state edge information: ")
            for j in range(0, len(self.state_tree[i].edges)):
                print("edge ", str(j))
                print("visit count: ", self.state_tree[i].edges[j].visit_count)
                print("mean action value: ", self.state_tree[i].edges[j].mean_action_value)
                print("prior probability: ", self.state_tree[i].edges[j].prior_prob)

            print("________________________________")

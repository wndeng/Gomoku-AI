# Copyright @ William Deng, 2018

# ______________________________________________COMMENTS___________________________________________________
# This is my implementation of an AI that uses AlphaZero architecture that plays the game Gomoku.
# AlphaZero architecture is referenced here:
#   https://arxiv.org/pdf/1712.01815.pdf
#   https://deepmind.com/documents/119/agz_unformatted_nature.pdf
# Please email me @ wndeng@ucdavis.edu if you think I wrote something wrong or had parts incorrectly implemented :D
# _________________________________________________________________________________________________________

import Jakiro as J
import Constant_Parameters as Param
import MCTS as MCTS
import copy as copy


def training_worker(model, sim_count):
    MCTS_tree = MCTS.MCTS(model, sim_count)
    training_data = MCTS_tree.run()
    return training_data


def main():

    network = J.NeuralNetwork()
    model = network.build_model()
    # Load previously computed model either by hand, from Param, or start at 0. Load weights can be synchronized with
    # param.TRAIN_COUNT_START, so one can stop and restart training from their last saved model

    name = "Iteration_" + str(Param.TRAIN_COUNT_START) + "_.hdf5"
    if Param.TRAIN_COUNT_START == 0:
        model.save("Iteration_0_.hdf5")

    network.model.load_weights(name) # Replace argument with name if want automatic weight loading
    train_count = copy.copy(Param.TRAIN_COUNT_START + 1)
    input_state = []
    target_prob = []
    target_val = []
    max_count = copy.copy(train_count + Param.STOP_COUNT)
    while train_count <= max_count:
        data_packet = training_worker(model, Param.MCTS_SIMULATION_COUNT)
        for data_set in data_packet:
            input_state.append(data_set[0][0, :, :, :])
            target_prob.append(data_set[1])
            target_val.append(data_set[2])

        # Shorten training set by removing earlier data if total count exceeds MAX_DATA_COUNT
        if len(input_state) >= Param.MAX_DATA_COUNT:
            start = len(input_state) - Param.MAX_DATA_COUNT - 1
            input_state = input_state[start:-1]
            target_prob = target_prob[start:-1]
            target_val = target_val[start:-1]

        # Start training once enough seed data is collected
        if len(input_state) > Param.INITIAL_DATA_SEED:
            network.train(input_state, target_val,target_prob, train_count)
            train_count += 1


main()



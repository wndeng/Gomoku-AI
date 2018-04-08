# Copyright @ William Deng, 2018

import Constant_Parameters as Param
from keras.models import Model
from keras.layers import BatchNormalization as BN, LeakyReLU as LR, Conv2D, Dense, add, Input, Flatten
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import tensorflow as tf


def loss_function(y_true, y_pred):
    # This custom loss function masks illegal moves (nodes that are not visited, with y_true = 0)
    # The masking is done by creating a mask that is true when y_true = 0. This mask is used as a
    # condition to create a new set of y_pred, where every element that is masked will be set to
    # -999, so the computed softmax value is practically 0, and will not affect y_pred distribution
    # of valid moves.



    invalid = tf.fill(tf.shape(y_true), 0.0)
    cond = tf.equal(y_true, invalid)
    mask = tf.fill(tf.shape(y_true), -999.0)
    y_pred_masked = tf.where(cond, mask, y_pred)

    # The custom loss function is a tensorflow loss function that combines a softmax layer with cross entropy. This used
    # rather than applying a softmax as the last layer of the policy head and then using cross entropy as loss because
    # when these two functions are used together, the actual result requires less computation and is more stable.
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred_masked)
    return loss


class NeuralNetwork:
    def __init__(self):
        # import Constant_Parameters as Param
        self.learning_rate = Param.INITIAL_LEARNING_RATE
        self.model = self.build_model()

    def predict(self, state):
        return self.model.predict(state)

    def train(self, input_state, target_val, target_prob, count):
        # Standard training abstraction

        print("Training iteration: ", str(count))
        training_input = []
        prob_output = []
        val_output = []
        for i in range(0, Param.BATCH_TRAIN_COUNT * Param.BATCH_SIZE):
            num = np.random.random_integers(0, len(input_state) - 1)
            training_input.append(input_state[num])
            prob_output.append(target_prob[num])
            val_output.append(target_val[num])

        name = "Iteration_" + str(count) + "_.hdf5"
        checkpoint = self.checkpoint(count, name)

        self.model.fit(np.array(training_input),
                       [np.array(val_output), np.array(prob_output)],
                       batch_size = Param.BATCH_SIZE,
                       epochs = Param.EPOCHS,
                       callbacks = checkpoint)

    def checkpoint(self, iteration, name):
        # Callback used for saving the model at set interval
        if iteration % Param.SAVE_INTERVAL == 0:
            file = open("iteration.txt", 'r+')
            current_count = int(file.readline())
            file.close()
            open("iteration.txt", 'w').close()
            file = open("iteration.txt", 'r+')
            file.write(str(current_count + Param.SAVE_INTERVAL))
            file.close()
            return [ModelCheckpoint(name,
                                    verbose=1,
                                    save_best_only=False,
                                    mode='auto')]
        else:
            return []

    def residual_layer(self, initial_input):
        # Residual layer as defined in literature
        output_1 = Conv2D(
            filters = Param.FILTER_PER_LAYER,
            kernel_size = Param.FILTER_KERNEL_SIZE,
            data_format = "channels_first",
            padding = 'same',
            use_bias = False,
            activation='linear',
            kernel_regularizer=regularizers.l2(Param.L2_REG_CONST)
        )(initial_input)

        output_2 = BN(axis=1)(output_1)

        output_3 = LR()(output_2)

        output_4 = Conv2D(
            filters = Param.FILTER_PER_LAYER,
            kernel_size = Param.FILTER_KERNEL_SIZE,
            data_format = "channels_first",
            padding = 'same',
            use_bias = False,
            activation='linear',
            kernel_regularizer=regularizers.l2(Param.L2_REG_CONST)
        )(output_3)

        output_5 = BN(axis=1)(output_4)

        output_6 = add([initial_input, output_5])

        final_output = LR()(output_6)

        return final_output

    def policy_head(self, initial_input):
        # Policy head as defined in literature
        output_1 = Conv2D(
            filters = 2,
            kernel_size = (1,1), # 1x1 convolution
            data_format = "channels_first",
            padding = 'same',
            use_bias = False,
            activation='linear',
            kernel_regularizer=regularizers.l2(Param.L2_REG_CONST),
            kernel_initializer = 'random_uniform'
        )(initial_input)

        output_2 = BN(axis=1)(output_1)

        output_3 = LR()(output_2)

        output_4 = Flatten()(output_3)

        final_output = Dense(
            361, # 19 * 19,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(Param.L2_REG_CONST),
            name = 'policy_head'
        )(output_4)

        return final_output

    def value_head(self, initial_input):
        # Value head as defined in literature
        output_1 = Conv2D(
            filters = 1,
            kernel_size = (1,1), # 1x1 convolution
            data_format = "channels_first",
            padding = 'same',
            use_bias = False,
            activation='linear',
            kernel_regularizer=regularizers.l2(Param.L2_REG_CONST)
        )(initial_input)

        output_2 = BN(axis=1)(output_1)

        output_3 = LR()(output_2)

        output_4 = Flatten()(output_3)

        output_5 = Dense(
            Param.FILTER_PER_LAYER, # Use same convention of having the same number of nodes here as filter layers
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(Param.L2_REG_CONST)
        )(output_4)

        output_6 = LR()(output_5)

        final_output = Dense(
            1,
            use_bias=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(Param.L2_REG_CONST),
            name = 'value_head'
        )(output_6)

        return final_output

    def build_model(self):
        # This function constructs the model
        initial_input = Input(Param.INPUT_DIM)

        output_1 = Conv2D(
            filters = Param.FILTER_PER_LAYER,
            kernel_size = Param.FILTER_KERNEL_SIZE,
            data_format = "channels_first",
            padding = 'same',
            use_bias = False,
            activation ='linear',
            kernel_regularizer = regularizers.l2(Param.L2_REG_CONST)
        )(initial_input)

        output_2 = BN(axis=1)(output_1)

        res_input = LR()(output_2)
        res_output = res_input # Initialize res_output

        # Add residual layers
        for i in range(0, Param.RES_LAYER_COUNT):
            res_output = self.residual_layer(res_input)
            res_input = res_output

        value = self.value_head(res_output)
        policy = self.policy_head(res_output)
        model = Model(inputs=[initial_input], outputs=[value, policy])

        # The fit method uses the Adam optimizer because SGD tends to cause the loss function to become unstable
        # (nans during training), and overall Adam performs better than SGD.
        model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': loss_function},
                      optimizer = Adam(lr=Param.INITIAL_LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon = 10E-8),
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5}
        )

        return model

    def show_network_summary(self):
        # Shows an internal tensor flow chart of the neural network
        print(self.model.summary())
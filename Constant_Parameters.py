# Contains constants used in neural network and MTCS

MCTS_SIMULATION_COUNT = 30 # Number of MCTS simulations per move
TEMP_THRESHOLD = 3 # How many turns before switching to tau = 0
CPUCT = 0.2 # CPUCT value for modified PUCT algorithm
EPSILON = 0.25 # Epsilon value for  dirichlet noise
DIR_CONSTANT = 0.5 # Constant used for dirichlet function
RES_LAYER_COUNT = 8 # Number of residual layers
FILTER_PER_LAYER = 128 # Number of filters used
FILTER_KERNEL_SIZE = (3,3) # Size of kernel filter
INPUT_DIM = (2, 19, 19) # Input to convnet
BATCH_SIZE = 4 # Size of each mini-batch
EPOCHS = 1 # Number times a single set of data will be trained
L2_REG_CONST = 0.0001 # L2 regularization constant
INITIAL_LEARNING_RATE = 0.02 # Learning rate for fitting
STOP_COUNT = 9 # How many iterations will be trained before program terminates
BATCH_TRAIN_COUNT = 50 # Number of mimi-batches per epoch
SAVE_INTERVAL = 10 # How many training iterations before saving the model.
TRAIN_COUNT_START = 0 # This controls which iteration to start. Set at 0 and do not change for automatic restarting
INITIAL_DATA_SEED = 500 # This controls how many data points are required before first sampling
MAX_DATA_COUNT = 2000 # This controls maximum count of recent data that are available to be sampled for training

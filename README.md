# AlphaZero AI for Gomoku
by William Deng

## Short Introduction
AlphaZero is a generalization of the famous AlphaGo Zero AI, which defeated the previous AI chess champion, AlphaGo Master. AlphaZero has only minor changes compared to AlphaGo Zero, such as removing the use of game-specific add-ons (rotating/flipping game states when training due to Go's board invariance), and removing the concept of 'best player'. Instead, the AI will continuously train from a pool of data collected from recent games, which is then used to simulate more games.

## Guide to Code

### Notes
* Special thanks to Miller Wang 老司机 for debugging the game implementation with me!
* For my implementation of the AlphaZero algorithm, I used applicable add-ons, but without the best player. The network will be trained continuously.

* The code will run on the newest version of tensorflow-gpu, but will only run on tensorflow-1.6 for the cpu version, and the "loss function" in Jakiro.py will need to have the tensorflow loss function switched from:
    ```
    tf.nn.softmax_cross_entropy_with_logits()
    ```
    
    to 
    
    ```
    tf.nn.softmax_cross_entropy_with_logits_v2()
    ```
* Due to python threading not being able to use extra cores because of GIL, and that multi-processing queue has issues pickling Keras neural network models as well as weights, I will skip parallelizing MCTS and using A3C in training pipeline. These will be implemented once I edit my code to run on a distributed system, such as google cloud. More information are available in "Further work".
* AlphaGo Zero used for each epoch 700,000 sets of mini-batches containing 2,048 evaluated board positions. This is not feasible to run on my machine so I scaled down the training to be a 'proof of concept', using 200 sets of mini-batches containing 4 board positions. 
* A self-play as well as 'human vs. bot' framework will be implemented once the code is ported to run on google cloud.
* All code written using PyCharm 2017

### Python files documentaion
This section contains information on all the python files in the repository. More information are available in the files.

**Constant_Parameters.py**:
This file contains MCTS tuning parameters (dirchlet function constant, CPUCT, etc), as well as logistic values (batch size, number of epochs, etc).

**Jakiro.py**:
This file contains the custom neural network class that implements Alpha Zero's convolutional network, with slight changes. Due to reduction in complexity of the game, the number of residual layers have been reduced to 8 (not including initial convolution layer), and the number of filters used are reduced to 128. These numbers are chosen arbitrarily, and they will be tuned once the program can be ran effectively (definitely not effective on my local machine).

**MCTS.py**:
This file contains the class that implements Alpha Zero's guided Monte Carlo Tree Search. It takes in a neural network model and a simulation count and return the a data packet containing training data in the form of (neural network state input, MCTS calculated probabilities in a (1 x 361) list, and a z value of 1 or -1).

**Scrambler.py**
This file contains a scrambling function that takes in a board state and returns a randomly flipped and/or rotated board state. Used to exploit board invariance for training.

**Master.py**
This file is the driver for the program. It sets up the pipeline for training and apply data control (enforcing data sampling size, preparing data seed, etc)

**Debug.py**
This file contains a function used for debugging the game as well as MTCS. It prints out the 2D board state in a better format than simple print.

**Encoding.py**
This file contains a function that encodes a given board state into a string. The encoding is heuristically optimized to reduce comparison time.

**Game.py**
This file contains the implementation of Gomoku using PRO rules. There are two separate APIs available. The first API is for command-line play between two players, and the second API is for interfacing with MCTS.

### Training Pipeline
1. Master initiates neural network, either from default (Xavier Uniform) weight initializer if starting from scratch, or from a previously saved model file.
2. Master than proceed to call a MCTS search that when finished, return a data packet containg training data. If there is not enough data to meet minimum sampling seed, Master will repeatedly call MCTS using the same model until there are enough samples.
3. Once there is enough data, the model is trained by taking data from sampling pool randomly. Training parameters are set in Constant_Parameters.
4. This process is repeated continuously until a specific number of training iterations have occured. Throughout the training, models are saved with their specific iteration count in the name for identification at set intervals, set in Constant_Parameters.
5. Print statements for iterations and the final state of a board after an instance of MCTS are displayed by default as a way to visualize the training process.

## Future Works
This program is only a proof of concept, and thus impractical for actual usage. For the next couple months I will be working on a Kaggle project, through which I will hopefully learn the basics of cloud computing. From there, I will revisit this code and adapt it for training on google cloud platform by parallelizing the pipeline as well as MCTS for efficient computing. 

## Sources used
This section contains useful URLs that I used to learn the theory behind AlphaZero and do this project.

**AlphaGo Zero literature**
https://www.nature.com/articles/nature24270

**AlphaZero literature**
https://arxiv.org/pdf/1712.01815v1.pdf

**A github repo by AppliedDataSciencePartners that contains very useful code that implements AlphaZero AI for Connect4**
https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning

**A AlphaZero Cheat sheet covering the major points, with great visuals, also by AppliedDataSciencePartners**
https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0

**Keras documentation**
https://keras.io/

**Tensorflow documentation**
https://www.tensorflow.org/versions/r1.0/api_docs/

**A clear description of AlphaZero by Stanford**
https://web.stanford.edu/~surag/posts/alphazero.html

**A good article that summarizes the strengths of the AlphaZero method**
https://hackernoon.com/the-3-tricks-that-made-alphago-zero-work-f3d47b6686ef


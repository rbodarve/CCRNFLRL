# Deep Q-learning for playing Flappy Bird



## Introduction

This is an implementation of a Deep Q-learning algorithm in Python. The code is based on [this](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch).  

## Setup

The requirements are:

* **python 3.8**
* **pytorch**
* **pygame**
* **cv2**
* **numpy**
* **tensorboard** (for visualization)

It is recommended to use a virtual environment for running this code.

## Training

```
python train.py
```

will start training the model using default parameters. Inside train.py you will find the different arguments that you can supply the program for changing the specifications of the training.

For example, the following code will train a model for 50000 iterations and save the model file as test:

```
python train.py --num_iters=50000 --save_file="test"
```

The models are saved inside trained_models directory and the visualization files can be viewed by running

```
tensorboard --logdir=tensorboard
```

on the root directory.

You can modify the value fps in the file src/flappy_bird.py to change the speed of the simulation.

## Testing

```
python test.py –model_file “test”
```

will load the model named test inside the trained_model directory and run the simulation over and over.  It is recommended to run the testing simulation at 30 fps.

## Links

Video available at [YouTube](https://www.youtube.com/watch?v=YYr8VPLTh0o) 

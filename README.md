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

## Review notes (2026-10-17)

1. Run all commands from the repo root, because the asset paths are relative. Create `trained_models/` before training. The test command above uses an en dash and curly quotes; type plain `--model_file test`.
2. `test.py` loads a whole pickled model from `trained_models/`. The default name `flappy_bird` does not exist, so pass `--model_file flappy_bird_50000` or `--model_file test`.
3. The 2026-10-17 run check used Python 3.10.21, torch 2.5.1+cu121 (CUDA), pygame 2.6.1, opencv 5.0.0, numpy 2.1.3 and tensorboardX 2.6.4. The code also imports `tensorboardX` and Pillow, which the requirements list above does not name.
4. Known errors: the reward shaping in `train.py` gives +10 on every surviving frame, and `--force_gpu False` is still read as True. `torch.load` fails on torch 2.6 or newer, and every training run deletes the `tensorboard/` folder.
5. Known limits: the local `flappy_bird_50000` model (not in git) scored 0 in all 67 episodes of the run check. With 100k entries, the replay memory on the GPU needs more than 4 GB of VRAM, and `test.py` runs forever without printing scores.
6. Keep the MIT notice in `LICENSE`. This repo modifies the upstream code: it changes the sprites and the reward shaping, and it adds a target network and a BatchNorm model.

"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-4)  # Increased from 1e-6 for faster learning
    parser.add_argument("--gamma", type=float, default=0.99)  # Discount factor
    parser.add_argument("--initial_epsilon", type=float, default=0.1)  # Start with more exploration
    parser.add_argument("--final_epsilon", type=float, default=1e-4)  # End with exploitation
    parser.add_argument("--num_iters", type=int, default=200000)  # Increased from 20k to 200k iterations
    parser.add_argument("--replay_memory_size", type=int, default=100000,  # Doubled for better experience replay
                        help="Maximum size of replay memory")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--save_file", type=str, default="test")
    parser.add_argument("--target_update_freq", type=int, default=1000,  # Update target network every N iterations
                        help="Frequency of target network updates")
    parser.add_argument("--pipe_reward", type=float, default=10.0,  # Increased reward for passing pipe
                        help="Reward for passing each pipe")
    parser.add_argument("--death_penalty", type=float, default=-500,  # Reduced harshness
                        help="Penalty for collision")
    parser.add_argument("--force_gpu", type=bool, default=True,
                        help="Force GPU usage (exit if GPU not available)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID to use (default: 0)")

    args = parser.parse_args()
    return args


def train(opt):
    # GPU Setup
    if opt.force_gpu:
        if not torch.cuda.is_available():
            print("ERROR: --force_gpu=True but CUDA is not available!")
            print("Run 'python check_gpu.py' to diagnose GPU issues")
            print("Or use --force_gpu=False to fall back to CPU")
            exit(1)
        device = torch.device(f"cuda:{opt.gpu_id}")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(123)
        print(f"✓ Using GPU: {torch.cuda.get_device_name(opt.gpu_id)}")
    else:
        device = torch.device("cpu")
        torch.manual_seed(123)
        print("⚠ Using CPU (training will be slow)")
    
    model = DeepQNetwork()
    target_model = DeepQNetwork()  # Target network for stable learning
    target_model.load_state_dict(model.state_dict())
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(log_dir=opt.log_path + "/" + opt.save_file)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    game_state = FlappyBird()
    image, reward, terminal, score = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image).to(device)
    
    model.to(device)
    target_model.to(device)
    criterion.to(device)
    
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]
    target_model.eval()  # Target network in eval mode
    
    replay_memory = []
    iter = 0
    while iter < opt.num_iters:
        prediction = model(state)[0]
        # Exploration or exploitation
        epsilon = opt.final_epsilon + ((opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            action = randint(0, 1)
        else:

            #action = torch.argmax(prediction)[0]
            action = torch.argmax(prediction).item()

        next_image, reward, terminal, score = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image).to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        
        # Reward shaping for better learning
        if terminal:
            reward = opt.death_penalty  # Use configurable death penalty
        elif reward > 0.05:  # Pipe passed
            reward = opt.pipe_reward  # Use configurable pipe reward
        else:
            reward = 0.1  # Small reward for survival
        
        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)
        next_state_batch = next_state_batch.to(device)
        current_prediction_batch = model(state_batch)
        
        # Use target network for Q-value estimation (Double DQN approach)
        with torch.no_grad():
            next_prediction_batch = target_model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1
        
        # Update target network periodically
        if iter % opt.target_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
        
        if iter % 100 == 0:  # Log every 100 iterations to reduce overhead
            print("Iteration: {}/{}, Action: {}, Loss: {:.4f}, Epsilon: {:.4f}, Reward: {:.1f}, Q-value: {:.4f}, Score: {}".format(
                iter + 1,
                opt.num_iters,
                "FLAP" if action == 1 else "HOLD",
                loss.item(),
                epsilon, reward, torch.max(prediction).item(), score))
            writer.add_scalar('Train/Loss', loss, iter)
            writer.add_scalar('Train/Epsilon', epsilon, iter)
            writer.add_scalar('Train/Reward', reward, iter)
            writer.add_scalar('Train/Q-value', torch.max(prediction), iter)
            writer.add_scalar('Train/Score', score, iter)
        
        if (iter+1) % 50000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter+1))
            print("Model checkpoint saved at iteration {}".format(iter+1))
    torch.save(model, "{}/{}".format(opt.saved_path,opt.save_file))


if __name__ == "__main__":
    opt = get_args()
    train(opt)

"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing



def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--model_file", type=str, default="flappy_bird")
    parser.add_argument("--force_gpu", type=bool, default=True,
                        help="Force GPU usage (exit if GPU not available)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID to use (default: 0)")

    args = parser.parse_args()
    return args


def test(opt):
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
        print("⚠ Using CPU")
    
    model = torch.load("{}/{}".format(opt.saved_path, opt.model_file))
    model.to(device)
    model.eval()
    game_state = FlappyBird()
    image, reward, terminal, score = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image).to(device)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()

        next_image, reward, terminal, score = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image).to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

        state = next_state


if __name__ == "__main__":
    opt = get_args()
    test(opt)

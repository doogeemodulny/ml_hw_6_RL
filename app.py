import torch

from model import NeuralNetwork
from preprocessing import DEVICE
from evaluation import test
from training import Trainer


def main(mode):
    if mode == 'test':
        if DEVICE == torch.device('cuda'):
            model = torch.load('current_model_2000000.pth').eval()
            model.to(DEVICE)
        else:
            model = torch.load('current_model_2000000.pth', map_location='cpu', weights_only=False).eval()
        test(model)
    elif mode == 'train':
        model = NeuralNetwork()
        model.to(DEVICE)
        trainer = Trainer(model)
        trainer._initialize_weights()
        trainer.train()


if __name__ == "__main__":
    main('test')
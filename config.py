# config.py

DEVICE = 'cpu'

NUMBER_OF_ACTIONS = 2
IMAGE_HEIGHT = 84
STATE_FRAMES = 4  
IMAGE_WIDTH = 84
IMAGE_CHANNELS = 1

GAMMA = 0.99
START_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY_SIZE = 10000
MINIBATCH_SIZE = 32
LEARNING_RATE = 1e-6
NUMBER_OF_ITERATIONS = 2000000


CONV1_PARAMS = {'in_channels': 4, 
                'out_channels': 32, 
                'kernel_size': 8, 
                'stride': 4
                }
CONV2_PARAMS = {'in_channels': 32, 
                'out_channels': 64, 
                'kernel_size': 4, 
                'stride': 2
                }
CONV3_PARAMS = {'in_channels': 64, 
                'out_channels': 128, 
                'kernel_size': 3, 
                'stride': 2
                }
CONV4_PARAMS = {'in_channels': 128, 
                'out_channels': 64, 
                'kernel_size': 3, 
                'stride': 1
                }
FC1_PARAMS = {'in_features': 256, 
              'out_features': 512
              }
FC2_PARAMS = {'in_features': 512, 
              'out_features': 2
              }
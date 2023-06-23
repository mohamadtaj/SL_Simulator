from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import argparse
from main import *

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#session = InteractiveSession(config=config)

parser = argparse.ArgumentParser()

# Parse command line arguments
parser.add_argument("--mode", "--mode", type=str, help="mode of training (Swarm Learning: swarm, Centralized: centralized, Local: local)", default="swarm")
args = vars(parser.parse_args())

mode = args["mode"]

num_nodes = 2

# Neural networks parameters
BATCH_SIZE = 128
EPOCHS = 20
optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
loss = 'binary_crossentropy'

print()
print()
print(f'Starting the simulation...')

if (mode == 'swarm'):
    main_swarm (mode, num_nodes, EPOCHS, BATCH_SIZE, optimizer, loss)
elif (mode == 'centralized'):
    main_centralized (mode, EPOCHS, BATCH_SIZE, optimizer, loss)
elif (mode == 'local'):
    main_local (mode, num_nodes, EPOCHS, BATCH_SIZE, optimizer, loss)
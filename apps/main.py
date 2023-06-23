# GPU Configuration
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# session = InteractiveSession(config=config)


from peer import *
from data_loader import *
from swarm import *
from centralized import *
from tensorflow import keras
import pandas as pd
    
    
def main_swarm (mode, num_nodes, EPOCHS, BATCH_SIZE, optimizer, loss):
    
    path = './data/test'
    
    x_test, y_test = load_data(path)
    
    input_size = x_test.shape[1:]
    output_size = 1
    
    env = FL_Env(num_nodes)
    env.create_nodes()
    
    env.initialize_peers()    
    env.connect_net()    
    env.set_coordinator()
   
    print()
    print()
    for node in env.nodes:
        print(f'node {node.id}:')
        print(f'x_train shape: {node.x_train.shape}')
        print(f'y_train shape: {node.y_train.shape}')
        print()
        
    print('--------------------------------------------')
    
    train_accuracy, test_accuracy, train_loss, test_loss = train_network_swarm (env, input_size, output_size, x_test, y_test, BATCH_SIZE, EPOCHS, optimizer, loss)  

    export_results(mode, train_accuracy, test_accuracy, train_loss, test_loss)    


def main_centralized (mode, EPOCHS, BATCH_SIZE, optimizer, loss):
    
    train_path = './data/train'
    test_path = './data/test'
    
    x_train, y_train = load_data(train_path)
    x_test, y_test = load_data(test_path)

    input_size = x_train.shape[1:]
    output_size = 1
         
    print('--------------------------------------------')
    print(f'x_train shape: {x_train.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'x_test shape: {x_test.shape}')
    print(f'y_test shape: {y_test.shape}')
    print()
    print('--------------------------------------------')
    
    train_accuracy, test_accuracy, train_loss, test_loss = train_centralized (x_train, y_train, x_test, y_test, BATCH_SIZE, EPOCHS, input_size, output_size, optimizer, loss)  

    export_results (mode, train_accuracy, test_accuracy, train_loss, test_loss)
    
    
def main_local (mode, num_nodes, EPOCHS, BATCH_SIZE, optimizer, loss):
    
    test_path = './data/test'
    
    x_test, y_test = load_data(test_path)
    
    input_size = x_test.shape[1:]
    output_size = 1
    
    env = FL_Env(num_nodes)
    env.create_nodes()
    
    print()
    print("Local models training... ")
    print()
    
    for node in env.nodes:
        
        print()
        print(f'Node {node.id} Model Training...')
        print()
        
        print('--------------------------------------------')
        print(f'x_train shape: {node.x_train.shape}')
        print(f'y_train shape: {node.y_train.shape}')
        print(f'x_test shape: {x_test.shape}')
        print(f'y_test shape: {y_test.shape}')
        print()
        print('--------------------------------------------')
        
        train_accuracy, test_accuracy, train_loss, test_loss = train_centralized (node.x_train, node.y_train, x_test, y_test, BATCH_SIZE, EPOCHS, input_size, output_size, optimizer, loss)  

        export_results_local (mode, node.id, train_accuracy, test_accuracy, train_loss, test_loss)
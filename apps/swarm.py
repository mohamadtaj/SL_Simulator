import numpy as np
from utils import *

# train function for the typical SL model
def train_model_swarm(env, BATCH_SIZE):

    avg_loss = []
    avg_accuracy = []

    nodes_loss = []
    nodes_accuracy = []
    
    env.set_coordinator() # randomly set the coordinator for this training round
    coordinator = env.next_coordinator 
    
    # locally train at each node
    for node in env.nodes:
    
        node.train(BATCH_SIZE)
        nodes_loss = np.append(nodes_loss, node.loss)
        nodes_accuracy = np.append(nodes_accuracy, node.accuracy)       
   
    coordinator.take_avg_params() # merge the models at the coordinator 
    coordinator.broadcast_params() # broadcast the new model to all the nodes 
    
    avg_loss = np.append(avg_loss, np.mean(nodes_loss))
    avg_accuracy = np.append(avg_accuracy, np.mean(nodes_accuracy))
  
    return avg_loss, avg_accuracy # the average loss and accuracy throughout all the nodes
    
# evaluate function for the typical SL model    
def evaluate_model_swarm(env, x_test, y_test, BATCH_SIZE):
    
    test_batches = mini_batches(x_test, y_test, BATCH_SIZE) # create batches from the test samples
    BATCH_NUM = len(test_batches)
    
    avg_loss = []
    avg_accuracy = []
    
    node_loss = []
    node_accuracy = []
    
    node = env.nodes[1] # choose one of the nodes to test the model on the test data (all the nodes have the same model).
    node.model.set_weights(node.parameters) # set the final model parameters 
            
    for batch_iter in range(BATCH_NUM):
        
        x, y = test_batches[batch_iter]
        loss, accuracy = node.model.test_on_batch(x,y)
      
        node_loss = np.append(node_loss,loss)
        node_accuracy = np.append(node_accuracy,accuracy)
            
    avg_loss = np.append(avg_loss, np.mean(node_loss))
    avg_accuracy = np.append(avg_accuracy, np.mean(node_accuracy))
    
    return avg_loss, avg_accuracy    
    
# main training loop function    
def train_network_swarm(env, input_size, output_size, x_test, y_test, BATCH_SIZE, EPOCHS,  optimizer, loss):
    
    train_loss=[]
    test_loss = []
    train_accuracy = []
    test_accuracy = []    
    
    print()
    print("Swarm Model Training... ")
    print()
    
    for node in env.nodes:
        node.define_model(input_size, output_size, optimizer, loss) # initialize the model at each node
        
    # loop over the number of epochs    
    for epoch_num in range(EPOCHS):
    
        avg_train_loss, avg_train_accuracy = train_model_swarm(env, BATCH_SIZE) # perform one round of training

        avg_test_loss, avg_test_accuracy = evaluate_model_swarm(env, x_test, y_test, BATCH_SIZE) # perform one round of evaluation
        
        train_loss.append(avg_train_loss)
        test_loss.append(avg_test_loss)
        train_accuracy.append(avg_train_accuracy)
        test_accuracy.append(avg_test_accuracy)
        
        print("EPOCH {} FINISHED".format(epoch_num + 1))
        print("----------------------------Train Acuuracy = ", train_accuracy[-1])
        print("-----------------------------Test Acuuracy = ", test_accuracy[-1])
        print("----------------------------Train Loss = ", train_loss[-1])
        print("-----------------------------Test Loss = ", test_loss[-1])

    return train_accuracy, test_accuracy, train_loss, test_loss
      
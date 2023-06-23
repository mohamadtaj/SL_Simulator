import numpy as np
import math
import os

"""creating mini-batches for training"""   
def mini_batches(X, Y, mini_batch_size):
    
    m = X.shape[0]
    mini_batches = []

    permutation = np.random.permutation(m)
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation]

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[(k+1)*mini_batch_size : m]
        mini_batch_Y = shuffled_Y[(k+1)*mini_batch_size : m]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches    
         

"""exporting the accuracy and loss"""    
def export_results(mode, train_accuracy, test_accuracy, train_loss, test_loss):

    if not os.path.exists("results"):
        os.makedirs("results")

    np.save("./results/"+mode+"_train_acc.npy", train_accuracy)
    np.save("./results/"+mode+"_test_acc.npy", test_accuracy)
    np.save("./results/"+mode+"_train_loss.npy", train_loss) 
    np.save("./results/"+mode+"_test_loss.npy", test_loss)    
    
def export_results_local(mode, node_id, train_accuracy, test_accuracy, train_loss, test_loss):

    if not os.path.exists("results"):
        os.makedirs("results")

    np.save("./results/"+mode+"_"+str(node_id)+"_train_acc.npy", train_accuracy)
    np.save("./results/"+mode+"_"+str(node_id)+"_test_acc.npy", test_accuracy)
    np.save("./results/"+mode+"_"+str(node_id)+"_train_loss.npy", train_loss) 
    np.save("./results/"+mode+"_"+str(node_id)+"_test_loss.npy", test_loss) 
import numpy as np
from utils import *
from models import *

# train function for the centralized model
def train_model_cent(x_train, y_train, BATCH_SIZE, model):

    avg_loss = []
    avg_accuracy = []
    
    
    batch = mini_batches(x_train, y_train, BATCH_SIZE) # create batches from the training samples
    BATCH_NUM = len(batch)
    for batch_iter in range(BATCH_NUM):
  
        x, y = batch[batch_iter]
        loss, accuracy = model.train_on_batch(x,y)
        avg_loss = np.append(avg_loss, loss)
        avg_accuracy = np.append(avg_accuracy, accuracy)
        
    return avg_loss, avg_accuracy

# evaluate function for the centralized model
def evaluate_model_cent(x_test, y_test, BATCH_SIZE, model):
    
    avg_loss = []
    avg_accuracy = []
    
    batch = mini_batches(x_test, y_test, BATCH_SIZE) # create batches from the test samples
    BATCH_NUM = len(batch)
    for batch_iter in range(BATCH_NUM):
      
        x, y = batch[batch_iter]
        loss, accuracy = model.test_on_batch(x,y)
        
        avg_loss = np.append(avg_loss, loss)
        avg_accuracy = np.append(avg_accuracy, accuracy)
    
    return avg_loss, avg_accuracy      

# main training loop function
def train_centralized(x_train, y_train, x_test, y_test, BATCH_SIZE, EPOCHS, input_size, output_size, optimizer, loss):

    centralized_model = model(input_size, output_size)
    centralized_model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])  
    
    total_train_loss = []
    total_test_loss = []
    total_train_accuracy = []
    total_test_accuracy = []
    
    
    print()
    print("Centralized model training... ")
    print()
    
    # loop over the number of epochs 
    for epoch_num in range(EPOCHS):
    
        train_loss, train_accuracy = train_model_cent(x_train, y_train, BATCH_SIZE, model=centralized_model) # perform one round of training
        test_loss, test_accuracy = evaluate_model_cent(x_test, y_test, BATCH_SIZE, model=centralized_model) # perform one round of evaluation

        
        total_train_loss.append(np.mean(train_loss))
        total_test_loss.append(np.mean(test_loss))
        total_train_accuracy.append(np.mean(train_accuracy))
        total_test_accuracy.append(np.mean(test_accuracy))
        
        print("EPOCH {} FINISHED".format(epoch_num + 1))
        print("----------------------------Train Acuuracy = ", total_train_accuracy[-1])
        print("-----------------------------Test Acuuracy = ", total_test_accuracy[-1])
        print("----------------------------Train Loss = ", total_train_loss[-1])
        print("-----------------------------Test Loss = ", total_test_loss[-1])        

    return total_train_accuracy, total_test_accuracy, total_train_loss, total_test_loss
   
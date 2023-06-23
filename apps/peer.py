import numpy as np
from utils import *
from models import *
from data_loader import *
import pandas as pd


class FL_Node:
    def __init__(self, env, x_train, y_train, nodes_sizes, id):
    
        self.id = id
        self.env = env
        self.peers = None
        self.connections = {}
        self.active = True        
        self.x_train = x_train
        self.y_train = y_train
        self.model = None
        self.parameters = None
        self.network_params = []
        self.loss = None
        self.accuracy = None
        self.next_coordinator = None
        self.nodes_sizes = nodes_sizes
        
        self.neighbor = None
    
    # connect the node to a given peer
    def connect(self, peer):
        conn = FL_Connection(self, peer)
        self.connections[peer] = conn
        if not peer.is_connected(self):
            peer.connect(self)
    
    # connect the node to all the other nodes
    def connect_all(self):
        for peer in self.peers:
            self.connect(peer)

    # set the node's peers
    def set_peers(self):
        self.peers = self.env.nodes.copy()
        self.peers.remove(self)

    # check if there is a conenction between the node and a given peer
    def is_connected(self, peer):
        return peer in self.connections
     
     
    def send(self, receiver, msg, broadcast):
        conn = self.connections[receiver]
        conn.deliver(msg, broadcast)

    def receive(self, msg, broadcast):
        if (broadcast):
            self.update_params(msg)
        else:    
            self.network_params.append(msg)
    
    # send the model parameters to the coordinator
    def share_params(self):
        broadcast = False
        if(self.next_coordinator != self):
            msg = (self.id, self.parameters)
            self.send(self.next_coordinator, msg, broadcast)
        
    def update_params(self, params):
        self.parameters = params

    # merge the models (performed by the acting coordinator)
    def take_avg_params(self):   
        self_params = (self.id, self.parameters)
        self.network_params.append(self_params)
        
        ids = [x[0] for x in self.network_params]

        params = np.array([x[1] for x in self.network_params])
        sizes = np.array([self.nodes_sizes[id] for id in ids])
        
        avg = np.dot(sizes, params)/np.sum(sizes)
        self.network_params = []
        self.update_params (avg)      
    
    # broadcast the new model to all the nodes (performed by the acting coordinator)
    def broadcast_params(self): 
        broadcast = True
        for peer in self.peers:
            self.send(peer, self.parameters, broadcast)
    
    # calculate the total number of batches at the node
    def total_batches(self, BATCH_SIZE):
        return len (mini_batches(self.x_train, self.y_train, BATCH_SIZE))

    # initialize the model    
    def define_model(self, input_size, output_size, optimizer, loss):
        SL_model = model(input_size, output_size)
        SL_model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
        self.parameters = SL_model.get_weights()
        self.model = SL_model
    
    
    def num_samples(self):
        return len(self.x_train)

    # train function    
    def train(self, BATCH_SIZE):

        self.loss = 0
        self.accuracy = 0
        self.model.set_weights(self.parameters)
        batches = mini_batches(self.x_train, self.y_train, BATCH_SIZE) # prepare batches for training
        BATCH_NUM = len(batches)
        
        batch_loss = []
        batch_accuracy = []
        
        # loop over the batches
        for batch_iter in range(BATCH_NUM):
            
            x, y = batches[batch_iter]
            loss, accuracy = self.model.train_on_batch(x,y)         
            
            batch_loss = np.append(batch_loss, loss)
            batch_accuracy = np.append(batch_accuracy, accuracy)
            
            
        self.loss = np.mean(batch_loss)
        self.accuracy = np.mean(batch_accuracy)
        
        weights = self.model.get_weights() # get the model parameters after one round of training
        self.update_params(weights) # save the model parameters in the node parameters
        self.share_params() # share the parameters with the coordinator
                            
# managing the connection between two peers for sending and receiving parameters        
class FL_Connection:
    def __init__(self, sender, receiver):
        self.sender = sender
        self.receiver = receiver
        
    def deliver(self, msg, broadcast):
        self.receiver.receive(msg, broadcast)
        
# the main simulation environment for the swarm network    
class FL_Env:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = None
        self.next_coordinator = None
        self.nodes_sizes = None
    
    # create the nodes
    def create_nodes(self):

        nodes = []
        nodes_sizes_dict = {}
        
        for i in range(1, self.num_nodes+1):
        
            path = './data/node_'+str(i)
            x, y = load_data(path)

            nodes_sizes_dict[i] = x.shape[0]
            
            nodes.append(FL_Node(self, x, y, nodes_sizes_dict, i))


        self.nodes = nodes
        self.nodes_sizes = nodes_sizes_dict
    
    # initialize peers at the network
    def initialize_peers(self):
        for node in self.nodes:
            node.set_peers() 
    
    # make a connection among all the nodes
    def connect_net(self):
        for node in self.nodes:
            node.connect_all() 
    
    # set the coordinator
    def set_coordinator(self):
        active_nodes = [node for node in self.nodes if node.active==True]

        lucky = np.random.choice(active_nodes)
        self.next_coordinator = lucky
        for node in self.nodes:
            node.next_coordinator = lucky  
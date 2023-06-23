# SL_Simulator
This repository contains a program for simulating a swarm learning framework. It also contains options for running the simulation for local training at each node and also centralized training where the training is perform on all the data.

### Modes:
| Mode  | Option |
| ------------- | ------------- |
| Centralized Training  | --mode centralized  |
| Local Training  | --mode local  |
|  Swarm Learning | --mode swarm  |

## Dataset Preparation:
The data should be placed in the same directory as the python files in a folder named "data". 
For Swarm Learning, each node's data should be in a seperate folder with the name "node_1", "node_2", etc (./data/node_1).
For Centralized learning, the data should be placed in a folder named "train" inside the "data" folder (./data/train).
And finally the test data should be in a folder named "test" (./data/test).

## Run:
To run the simulation, simply execute the run.py file with the **mode** argument.
For example, to run the simulation for the Swarm Learning framework, type in:

```
python run.py --mode swarm

```

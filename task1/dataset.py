import pickle
import os
import os.path as osp
import abc_py
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

train_circuit = ['adder', 'alu2', 'apex3', 'apex5', 'arbiter', 'b2', 'c1355', 'c2670', 'c5315',
                 'c6288', 'ctrl', 'frg1', 'i7', 'i8', 'int2float', 'log2', 'm3', 'max', 'max512', 
                 'multiplier', 'priority', 'prom2', 'table5'] 

def get_data(state):
    _abc = abc_py.AbcInterface()
    _abc.start()
    _abc.read(state)
    data = {}
    numNodes = _abc.numNodes()
    print(numNodes)
    data['node_type'] = np.zeros(numNodes, dtype = int)
    data['num_inverted_predecessors'] = np.zeros(numNodes, dtype=int)
    edge_src_index = []
    edge_target_index = []
    for nodeIdx in range(numNodes):
        aigNode = _abc.aigNode(nodeIdx)
        nodeType = aigNode.nodeType()
        data['num_inverted_predecessors'][nodeIdx] = 0
        if nodeType == 0 or nodeType == 2:
            data['node_type'][nodeIdx] = 0
        elif nodeType == 1:
            data['node_type'][nodeIdx] = 1
        else:
            data['node_type'][nodeIdx] = 2
            if nodeType == 4:
                data['num_inverted_predecessors'][nodeIdx] = 1
            if nodeType == 5:
                data['num_inverted_predecessors'][nodeIdx] = 2
        if(aigNode.hasFanin0()):
            fanin = aigNode.fanin0()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
        if(aigNode.hasFanin1()):
            fanin = aigNode.fanin1()
            edge_src_index.append(nodeIdx)
            edge_target_index.append(fanin)
    data['edge_index'] = torch.tensor([edge_src_index, edge_target_index], dtype=torch.long)
    data['node_type'] = torch.tensor([data['node_type']])
    data['num_inverted_predecessors'] = torch.tensor([data['num_inverted_predecessors']])
    data['nodes'] = numNodes
    return data

def generate_aig(state, is_train = True):
    circuitName, actions = state.split('_')
    if is_train :
        circuitPath = './InitialAIG/train/' + circuitName + '.aig'
    else:
        circuitPath = './InitialAIG/test/' + circuitName + '.aig'
    libFile = './lib/7nm/7nm.lib'
    logFile = './log/' + state + '.log'
    nextState = './aig/' + state + '.aig' # current AIG file
    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }
    actionCmd = ''
    for action in actions:
        actionCmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = "./yosys/yosys-abc -c \"read " + circuitPath + ";" + actionCmd + "; read_lib " + libFile + "; write " + nextState + "; print_stats\" > " + logFile
    os.system(abcRunCmd)

def get_dataset():
    train_set = []
    test_set = []
    data_folder = "./project_data"
    for file in os.listdir(data_folder):
        if file.endswith(".pkl"):
            with open(osp.join(data_folder, file), "rb") as f:
                input_target = pickle.load(f)
                states = input_target['input']
                targets = input_target['target']
                for state, target in zip(states, targets):
                    circuit = state.split('_')[0]
                    if circuit in train_circuit:
                        generate_aig(state)
                        data = get_data('aig/' + state + '.aig')
                        node_features = torch.cat([data['node_type'], data['num_inverted_predecessors']], axis=1)
                        new_data = Data(x=node_features, edge_index=data['edge_index'], y=target)
                        train_set.append(new_data)
                    else:
                        generate_aig(state, False)
                        data = get_data('aig/' + state + '.aig')
                        node_features = torch.cat([data['node_type'], data['num_inverted_predecessors']], axis=1)
                        new_data = Data(x=node_features, edge_index=data['edge_index'], y=target)
                        test_set.append(new_data)
    return train_set, test_set

if __name__ == "__main__":
    get_dataset()
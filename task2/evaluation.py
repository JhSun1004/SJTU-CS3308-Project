import os
import abc_py
import numpy as np
import re
import torch
from tqdm import *
from myGCN import GCN
from torch_geometric.data import Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_circuit = ['adder', 'alu2', 'apex3', 'apex5', 'arbiter', 'b2', 'c1355', 'c2670', 'c5315',
                 'c6288', 'ctrl', 'frg1', 'i7', 'i8', 'int2float', 'log2', 'm3', 'max', 'max512', 
                 'multiplier', 'priority', 'prom2', 'table5'] 

synthesisOpToPosDic = {
    0: "refactor",
    1: "refactor -z",
    2: "rewrite",
    3: "rewrite -z",
    4: "resub",
    5: "resub -z",
    6: "balance"
}

AIG = '/home/yuan/桌面/ML/project/InitialAIG/test/alu4.aig'
libFile = '/home/yuan/桌面/ML/project/lib/7nm/7nm.lib'
logFile = '/home/yuan/桌面/alu4/alu4.log'
output_dir = '/home/yuan/桌面/alu4'

def get_data(state):
    _abc = abc_py.AbcInterface()
    _abc.start()
    _abc.read(state)
    data = {}
    numNodes = _abc.numNodes()
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
    actionCmd = ''
    for action in actions:
        actionCmd += (synthesisOpToPosDic[int(action)] + ';')
    abcRunCmd = "./yosys/yosys-abc -c \"read " + circuitPath + ";" + actionCmd + "; read_lib " + libFile + "; write " + nextState + "; print_stats\" > " + logFile
    os.system(abcRunCmd)

def data_process(data):
    x = torch.cat([data['node_type'], data['num_inverted_predecessors']], dim=0)
    x = x.T
    x = torch.tensor(x, dtype=torch.float32)
    edge_index = data['edge_index']
    new_data = Data(x = x, edge_index = edge_index, y = data['y'])
    return new_data

def evaluate_1(state):
    model = GCN().to(device)
    model.load_state_dict(torch.load('./model/final.pth'))
    circuit = state.split('_')[0]
    generate_aig(state)
    data = get_data('aig/' + circuit + '.aig')
    data = data_process(data)
    return model(data)


def evaluate_2(state, is_train=True):
    circuitName, actions = state.split('_')
    circuitPath = '/home/yuan/桌面/' + state + '.aig'
    libFile = '/home/yuan/桌面/ML/project/lib/7nm/7nm.lib'
    logFile = '/home/yuan/桌面/alu4/' + circuitName + '.log'
    nextState = state + '.aig'

    synthesisOpToPosDic = {
        0: "refactor",
        1: "refactor -z",
        2: "rewrite",
        3: "rewrite -z",
        4: "resub",
        5: "resub -z",
        6: "balance"
    }
    abcRunCmd = f"/home/yuan/桌面/yosys/yosys-abc -c 'read {circuitPath}; read_lib {libFile}; map; topo; stime' > {logFile}"
    with open(logFile) as f:
        areaInformation = re.findall(r'[0-9.]+', f.readlines()[-1])
    eval = float(areaInformation[-8]) * float(areaInformation[-3])
    # print(f"areaInformation[-8]: {areaInformation[-8]} | areaInformation[-3]: {areaInformation[-3]}")
    print(f"{nextState}: eval: {eval}")

    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    abcRunCmd = f"/home/yuan/桌面/yosys/yosys-abc -c 'read {circuitPath}; {RESYN2_CMD}; read_lib {libFile}; write {nextState}; map; topo; stime' > {logFile}"
    os.system(abcRunCmd)

    with open(logFile) as f:
        areaInformation = re.findall(r'[0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-8]) * float(areaInformation[-3])
    print(f"baseline: {baseline}")
    eval = 1 - eval / baseline
    return eval


def argmax(lst):
    if not lst:
        return None
    return max(enumerate(lst), key=lambda x: x[1])[0]

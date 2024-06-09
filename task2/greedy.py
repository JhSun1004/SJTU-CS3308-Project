import re
import os
import torch
import numpy as np
import abc_py

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

def evaluate_2(state, is_train = True):
    circuitName, actions = state.split('_')
    circuitPath = '/home/yuan/桌面/ML/project/InitialAIG/test/' + circuitName + '.aig'
    libFile = '/home/yuan/桌面/ML/project/lib/7nm/7nm.lib'
    logFile = '/home/yuan/桌面/alu4/' + circuitName + '.log'
    # nextState = os.path.join(output_dir, state + '.aig')
    nextState = state + '.aig'  # current AIG file
    
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
    eval = float(areaInformation[-9]) * float(areaInformation[-4])
    print(f"{nextState}: eval: {eval}")

    RESYN2_CMD = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance;"
    abcRunCmd = f"/home/yuan/桌面/yosys/yosys-abc -c 'read {circuitPath}; {RESYN2_CMD}; read_lib {libFile}; write {nextState}; map; topo; stime > {logFile}'"
    os.system(abcRunCmd)
    
    with open(logFile) as f:
        areaInformation = re.findall(r'[0-9.]+', f.readlines()[-1])
        baseline = float(areaInformation[-9]) * float(areaInformation[-4])
    print(f"baseline: {baseline}")        
    eval = 1- eval/baseline
    return eval

def argmax(lst):
    if not lst:
        return None
    return max(enumerate(lst), key=lambda x: x[1])[0]


aig_now = "alu4_"
for step in range(10):
    childs = []
    print(f"step: {step}")
    for child in range(7):
        print(f"child: {child}")
        # childFile = 'alu4_' + str(child) + '.aig'
        childFile = os.path.join(output_dir, str(aig_now) + str(child) + '.aig')
        print(childFile)
        abcRunCmd = f"/home/yuan/桌面/yosys/yosys-abc -c 'read {AIG}; {synthesisOpToPosDic[child]}; read_lib {libFile}; write {childFile}; print_stats' > {logFile}"
        os.system(abcRunCmd)
        
        childFile = childFile.replace("/home/yuan/桌面/alu4/", "")
        print(childFile)
        
        childs.append(childFile)
        print(f"childs: {childs}")

    state_without_extension = [s.replace('.aig', '') for s in childs]
    print(state_without_extension)
    childScores = []
    for circuitName in state_without_extension:
        state = circuitName
        score = evaluate_2(state)
        print(f"score: {score}")
        childScores.append(score)
    
    action = argmax(childScores)
    print(action)
    AIG = childs[action]
    print(f"childs[action]: {childs[action]}")
    aig_now = childs[action].replace('.aig', '')
    print(f"aig_now: {aig_now}")
    
abcRunCmd = f"/home/yuan/桌面/yosys/yosys-abc -c 'read {AIG}; read_lib {libFile}; map; topo; stime > {logFile}'"
    
os.system(abcRunCmd)

with open(logFile) as f:
    areaInformation = re.findall(r'[0-9.]+', f.readlines()[-1])
    adpVal = float(areaInformation[-9]) * float(areaInformation[-4])
    print(f"adpVal: {adpVal}")
    
print((baseline - adpVal) / baseline)
return (baseline - adpVal) / baseline

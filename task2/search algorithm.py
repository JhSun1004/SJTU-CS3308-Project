import os
import heapq
import random
import math

# DFS
def depth_first_search(AIG, output_dir, libFile, synthesisOpToPosDic, evaluate_2, max_depth=10):
    def dfs(state, depth, path):
        if depth == max_depth:
            return path, evaluate_2(state.replace('.aig', ''))
        
        best_path, best_score = path, -float('inf')
        for child in range(7):
            childFile = os.path.join(output_dir, str(state) + str(child) + '.aig')
            abcRunCmd = f"/home/yuan/桌面/yosys/yosys-abc -c 'read {AIG}; {synthesisOpToPosDic[child]}; read_lib {libFile}; write {childFile}; print_stats' > {logFile}"
            os.system(abcRunCmd)
            new_state = childFile.replace(output_dir, "").replace('.aig', '')
            new_path, new_score = dfs(new_state, depth + 1, path + [new_state])
            if new_score > best_score:
                best_path, best_score = new_path, new_score
        return best_path, best_score
    
    best_path, best_score = dfs(AIG.replace('.aig', ''), 0, [])
    return best_path


# A*
def a_star_search(AIG, output_dir, libFile, synthesisOpToPosDic, evaluate_2, max_depth=10):
    def heuristic(state):
        return evaluate_2(state.replace('.aig', ''))
    
    open_set = []
    heapq.heappush(open_set, (0, [AIG.replace('.aig', '')]))
    best_score = -float('inf')
    best_path = []
    
    while open_set:
        current_score, path = heapq.heappop(open_set)
        current_state = path[-1]
        
        if len(path) == max_depth + 1:
            score = heuristic(current_state)
            if score > best_score:
                best_score = score
                best_path = path
            continue
        
        for child in range(7):
            childFile = os.path.join(output_dir, str(current_state) + str(child) + '.aig')
            abcRunCmd = f"/home/yuan/桌面/yosys/yosys-abc -c 'read {AIG}; {synthesisOpToPosDic[child]}; read_lib {libFile}; write {childFile}; print_stats' > {logFile}"
            os.system(abcRunCmd)
            new_state = childFile.replace(output_dir, "").replace('.aig', '')
            new_path = path + [new_state]
            heapq.heappush(open_set, (current_score + heuristic(new_state), new_path))
    
    return best_path


# NCTS
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == 7

    def best_child(self, c_param=1.4):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

def mcts(AIG, output_dir, libFile, synthesisOpToPosDic, evaluate_2, n_iter=1000):
    def select(node):
        while not node.is_fully_expanded():
            if len(node.children) < 7:
                return expand(node)
            else:
                node = node.best_child()
        return node

    def expand(node):
        child_index = len(node.children)
        child_state = os.path.join(output_dir, str(node.state) + str(child_index) + '.aig')
        abcRunCmd = f"/home/yuan/桌面/yosys/yosys-abc -c 'read {AIG}; {synthesisOpToPosDic[child_index]}; read_lib {libFile}; write {child_state}; print_stats' > {logFile}"
        os.system(abcRunCmd)
        child_state = child_state.replace(output_dir, "").replace('.aig', '')
        child_node = MCTSNode(child_state, parent=node)
        node.children.append(child_node)
        return child_node

    def simulate(node):
        current_state = node.state
        for _ in range(10):  # Simulate for a fixed depth
            child_index = random.randint(0, 6)
            child_state = os.path.join(output_dir, str(current_state) + str(child_index) + '.aig')
            abcRunCmd = f"/home/yuan/桌面/yosys/yosys-abc -c 'read {AIG}; {synthesisOpToPosDic[child_index]}; read_lib {libFile}; write {child_state}; print_stats' > {logFile}"
            os.system(abcRunCmd)
            current_state = child_state.replace(output_dir, "").replace('.aig', '')
        return evaluate_2(current_state.replace('.aig', ''))

    def backpropagate(node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    root = MCTSNode(AIG.replace('.aig', ''))
    for _ in range(n_iter):
        leaf = select(root)
        reward = simulate(leaf)
        backpropagate(leaf, reward)

    best_path = []
    node = root
    while node.children:
        node = node.best_child(c_param=0)
        best_path.append(node.state)
    return best_path


# Example usage
best_sequence = depth_first_search(AIG="alu4_", output_dir="/home/yuan/桌面/alu4/", libFile="path/to/libFile", synthesisOpToPosDic={}, evaluate_2=evaluate_2)
print(best_sequence)

best_sequence = a_star_search(AIG="alu4_", output_dir="/home/yuan/桌面/alu4/", libFile="path/to/libFile", synthesisOpToPosDic={}, evaluate_2=evaluate_2)
print(best_sequence)

best_sequence = mcts(AIG="alu4_", output_dir="/home/yuan/桌面/alu4/", libFile="path/to/libFile", synthesisOpToPosDic={}, evaluate_2=evaluate_2, n_iter=100)
print(best_sequence)



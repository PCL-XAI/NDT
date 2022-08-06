#依靠sklearn_json转化词典
#输入cstree
#输出decision_tree
import sklearn_json as skljson

def to_skl_tree(cstree):
    tree_dict = {}
    tree_dict['meta'] = 'decision-tree'
    tree_dict['n_features_'] = cstree._n_features
    tree_dict['feature_importances_'] = cstree.feature_importance
    tree_dict['max_features_'] = cstree._n_features
    tree_dict['n_classes_'] = len(cstree.classes_)
    tree_dict['n_outputs_'] = 1
    tree_dict['tree_'] = {}
    tree_dict['tree_']['max_depth'] = get_depth(cstree.root)-1
    tree_dict['tree_']['nodes'],tree_dict['tree_']['values'] = create_nodes_values(cstree)
    tree_dict['tree_']['node_count'] = len(tree_dict['tree_']['nodes'])
    tree_dict['tree_']['nodes_dtype'] = ['<i8', '<i8', '<i8', '<f8', '<f8', '<i8', '<f8'] #目测是固定
    tree_dict['classes_'] = list(cstree.classes_.keys())
    tree_dict['params'] = {
        'ccp_alpha': 0.0,
        'class_weight': None,
        'criterion': cstree.criterion,
        'max_depth': None,
        'max_features': cstree.max_features,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'min_samples_leaf': 2,
        'min_samples_split': 3,
        'min_weight_fraction_leaf': 0.0,
        'random_state': cstree.random_state,
        'splitter': cstree.splitter
    }
    return tree_dict
    
#先序计算，返回先序列
def preorder_dfs(node, index):
    if node:
        index.append(node)
        preorder_dfs(node.left_child, index)
        preorder_dfs(node.right_child, index)
    
#中序计算
def inorder_dfs(node, index):
    if node:
        inorder_dfs(node.left_child, index)
        index.append(node)
        inorder_dfs(node.right_child, index)
    
#获取深度
def get_depth(node):
    if node:
        return max(get_depth(node.left_child), get_depth(node.right_child))+1
    return 0
    
#先序遍历树，同时计算 
#1.序号 2.前驱节点 3.置0 4.threshold 5.impurity 6.n_samples 7.n_samples
#以及values
def create_nodes_values(cstree):
    preorder_nodes = []
    preorder_dfs(cstree.root, preorder_nodes) #获取了先序遍历下的node序列
    #对先序遍历进行下标
    for i in range(len(preorder_nodes)):
        preorder_nodes[i].content['preorder_index'] = i #把前序的id放置到nodes中
    #创建nodes
    nodes = [[0]*7 for i in range(len(preorder_nodes))]
    #填充第一、二列
    for i in range(len(preorder_nodes)):
        if preorder_nodes[i].left_child:
            nodes[i][0] = preorder_nodes[i].left_child.content['preorder_index']
        else:
            nodes[i][0] = -1
        if preorder_nodes[i].right_child:
            nodes[i][1] = preorder_nodes[i].right_child.content['preorder_index']
        else:
            nodes[i][1] = -1
    #填充第三,四，五，六，七列 【第三列统一填充为0】
    for i in range(len(preorder_nodes)):
        nodes[i][2] = preorder_nodes[i].content['feature']
        nodes[i][3] = preorder_nodes[i].content['threshold']
        nodes[i][4] = preorder_nodes[i].content['impurity']
        nodes[i][5] = preorder_nodes[i].content['n_sample']
        nodes[i][6] = float(preorder_nodes[i].content['n_sample'])
    #创建values
    values = []
    for i in range(len(preorder_nodes)):
        values.append([preorder_nodes[i].content['value']])
    return nodes, values

# transfer cstree to sklearn tree
def to_sklearn_tree(cstree):
    tree_dict = skljson.to_dict(cstree)
    return skljson.from_dict(tree_dict)

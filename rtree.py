import sys
import uuid
import queue
import numpy as np
from tool import preorder_dfs
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans,DBSCAN
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from strategy import Strategy,build_strategy_threshold,build_strategy_topk,get_data_matching,find_strategy_by_data

MAX_DEPTH_LIMIT=1000

#统计标签
def unique_label(labels, sort=False):
    label_set = set()
    if labels is not None:
        for l in labels:
            label_set.add(l)
    if sort:
        label_set = list(label_set)
        label_set.sort(reverse=False)
    return label_set

#统计Y的种类 Y['apple', 'orange', 'apple', ...]
def unique_count(Y, category=None):
    results = {}
    if category is not None:
        for c in category:
            results[c] = 0
    for y in Y:
        if y not in results:results[y] = 0
        results[y]+=1
    return results # 返回一个字典

# 在缓存中搜索合适的位置，大幅度提升
def search_in_cache(cache_table, feature, value):
    if feature in cache_table:
        res,index = binary_search(cache_table[feature]['value'], 0, len(cache_table[feature]['value'])-1, value)
        if res > 0:
            return cache_table[feature]['gain'][index]
    return None
    
# 二分搜索
def binary_search(arr, l, r, x):
    if r >= l:
        mid = int(l + (r - l)/2)
        if arr[mid] == x: 
            return 1,mid 
        elif arr[mid] > x: 
            return binary_search(arr, l, mid-1, x) 
        else: 
            return binary_search(arr, mid+1, r, x) 
    else: 
        mid = int((l+r)/2)
        return -1,mid

# 获取解决方法
# 输入：tree, [data]
def classify(x, node):
    if node.content['is_leave']: #叶子结点
        return node
    else:
        value = x[node.content['feature']]
        next_child = None
        if isinstance(value,int) or isinstance(value,float):
            if value < node.content['threshold']: 
                next_child = node.left_child
            else: 
                next_child = node.right_child
        else:
            if value==node.content['threshold']: 
                next_child = node.left_child
            else: 
                next_child = node.right_child
        return classify(x, next_child)
    
def solve(tree, X):
    assert len(X) > 0 and len(X[0]) == tree._n_features
    leaves = []
    for i in range(len(X)):
        leaves.append(classify(X[i], tree.root))
    paths = [] #记录从root到leave上的节点
    for i in range(len(leaves)):
        path = []
        tree._find_leave(tree.root, leaves[i], path) #获取路径
        path.reverse()
        paths.append(path)
    return paths

def get_solutions(tree):
    paths = []
    solutions = []
    leaves = tree._get_leave(tree.root)
    for l in leaves:
        path = []
        tree._find_leave(tree.root, l, path) #获取路径
        path.reverse()
        paths.append(path)
    for i in range(len(paths)):
        solution = set()
        for j in range(len(paths[i])-1):
            solution.add(str(paths[i][j].content['feature'])+':'+str(np.round(paths[i][j].content['threshold'], 1)))
        solutions.append(solution)
    return solutions

def print_solutions(tree, ibe, feature_names):
    paths = solve(tree, ibe)
    for i in range(len(paths)):
        startegy_path = []
        for j in range(len(paths[i])):
            if j < len(paths[i])-1:
                sign = ''
                if paths[i][j+1] == paths[i][j].left_child:
                    sign = '<'
                else:
                    sign = '>'
                startegy_path.append(feature_names[paths[i][j].content['feature']] + sign + str(np.round(paths[i][j].content['threshold'],3)))
            else:
                startegy_path.append(str(paths[i][j].content['output_label'][0]))
        s = ''.join([str(int(ibe[i][j])) for j in range(len(ibe[i]))]) + ': '
        for j in range(len(startegy_path)):
            if j < len(startegy_path)-1:
                s += startegy_path[j] + ' -> '
            else:
                s += startegy_path[j]
        print(s)
    return paths

#集合划分系列
def divideset(X, Y, feature, threshold, 
              threshold_indice=None, median_indice=None, 
              threshold_category=None, threshold_keys=None, use_index=False):
    #定义一个函数，判断当前数据行属于第一组还是第二组
    split_function = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_function = lambda x:x[feature] < threshold
    else:
        split_function = lambda x:x[feature]==threshold #如果特征不是数值，那么将使用是否等同进行判断
    # 将数据集拆分成两个集合，并返回
    x1,y1,x2,y2 = [],[],[],[]
    if use_index:
        if threshold_indice is None or median_indice is None: #没有索引
            for i in range(len(X)):
                x = X[i]
                if split_function(x):
                    x1.append(i)
                else:
                    x2.append(i)
            return x1,x2
        else: #存在索引
            divide_index = median_indice[threshold] #divide_index >= 1
            x1 = threshold_indice[:divide_index]
            x2 = threshold_indice[divide_index:]
            if threshold_category is None or threshold_keys is None:
                return x1,x2 #局部的 
            else:
                c1,c2 = {k:0 for k in threshold_keys},{k:0 for k in threshold_keys}
                c1_ = threshold_category[divide_index-1]
                c2_ = threshold_category[-1] - c1_
                for i in range(len(threshold_keys)):
                    c1[threshold_keys[i]] = c1_[i]
                    c2[threshold_keys[i]] = c2_[i]
                return x1,x2,c1,c2
    else:
        if threshold_indice is None or median_indice is None: #没有索引
            for x,y in zip(X,Y):
                if split_function(x):
                    x1.append(x)
                    y1.append(y)
                else:
                    x2.append(x)
                    y2.append(y)
            return x1,x2,y1,y2 #局部的 
        else:
            divide_index = median_indice[threshold] #divide_index >= 1
            x1 = X[threshold_indice[:divide_index]]
            x2 = X[threshold_indice[divide_index:]]
            y1 = Y[threshold_indice[:divide_index]]
            y2 = Y[threshold_indice[divide_index:]]
            if threshold_category is None or threshold_keys is None:
                return x1,x2,y1,y2 #局部的 
            else:
                c1,c2 = {k:0 for k in threshold_keys},{k:0 for k in threshold_keys}
                c1_ = threshold_category[divide_index-1]
                c2_ = threshold_category[-1] - c1_
                for i in range(len(threshold_keys)):
                    c1[threshold_keys[i]] = c1_[i]
                    c2[threshold_keys[i]] = c2_[i]
                return x1,x2,y1,y2,c1,c2

# 熵计算
# E(y) = -(y1logy1+y2logy2+...+ynlogyn)
# 输入：经过unique_count处理之后的之后的字典{'apple':3,'orange':2}
def entropy(unique_count):
    ent = 0.0
    unique_sum = np.sum([unique_count[c] for c in unique_count])
    if unique_sum > 0:
        for r in unique_count.keys():
            p = float(unique_count[r])/unique_sum
            if p > 0:
                ent = ent - p*np.log2(p)
    return ent

# 基尼计算
# gini = 1-(p1)^2-(p2)^2...
def gini(unique_count):
    unique_sum = np.sum([unique_count[c] for c in unique_count])
    gini_ = 0
    if unique_sum > 0:
        for r in unique_count.keys():
            p = float(unique_count[r])/unique_sum
            gini_ += p**2
    return 1 - gini_

# 用于缓存，加速计算
def caching_feature(X, y, y_keys, feature):
    medians = [] #前后平均数
    threshold_indice = [] #[特征值：下标] 例如[[low,1],[low,20],[median,12],[high,55]]
    threshold_category = [] #[特征值：{category}] 例如[low,{y1:2,y2:3}]
    unique_threshold_indice = {} #直接索引{median: 最末位下标} 索引最低为1
    for i in range(len(X)):
        ti = [X[i][feature], i]
        threshold_indice.append(ti)
    threshold_indice = sorted(threshold_indice, key=lambda x:x[0]) #按照特征值进行低到高排序
    last_ci = np.zeros(len(y_keys), dtype=np.int)
    last_ci[y_keys.index(y[threshold_indice[0][1]])] = 1
    threshold_category.append(last_ci) #追加第一个分类索引
    last_threshold = threshold_indice[0][0]
    for i in range(1, len(threshold_indice)): #建立median和快速键值
        current_threshold = threshold_indice[i][0]
        if current_threshold != last_threshold:
            median = (last_threshold + current_threshold)/2
            unique_threshold_indice[median] = i
            last_threshold = current_threshold
            medians.append(median)
        ci = np.zeros(len(y_keys), dtype=np.int)
        ci[y_keys.index(y[threshold_indice[i][1]])] = 1
        ci += last_ci
        threshold_category.append(ci)
        last_ci = ci
    threshold_indice = [threshold_indice[i][1] for i in range(len(threshold_indice))] #去除特征列
    return medians, threshold_indice, unique_threshold_indice, threshold_category

# 计算ID3，返回：根据ID3选择最好的特征下标，分割值，最大熵降，最佳划分
def ID3(X, y, feature=-1, threshold=None):
    candidate_gain = []
    candidate_feature = []
    candidate_theshold = []
    candidate_indice = [] #[(indice_x1, indice_x2)]
    feature_count = len(X[0]) #特征数
    y_categories = unique_count(y)
    y_keys = sorted(list(y_categories.keys()))
    information_gain = entropy(y_categories)
    if feature >= 0 and threshold is not None: #给定特征与阈值，直接计算分裂
        index1,index2 = divideset(X, y, feature, threshold, use_index=True) #根据该第feature个特征的阈值进行划分
        y1,y2 = y[index1],y[index2]
        p = float(len(index1))/(len(index1) + len(index2))
        category1,category2 = unique_count(y1),unique_count(y2)
        gain = information_gain - p*entropy(category1) - (1-p)*entropy(category2)
        gain = max(1-np.tanh(gain), 0) #使用tah进行统一使用
        return gain, (index1,index2)
    else:
        cache_table = {} #{feature_index:{v1:gain,v2:gain,...,vn:gain}
        feature_indice = list(range(0, feature_count))
        np.random.shuffle(feature_indice)
        for feature in range(0, feature_count): #遍历特征
            cache_feature = {'value':[],'gain':[]} #已经排好顺序
            medians, threshold_indice, unique_threshold_indice, threshold_category =\
                caching_feature(X, y, y_keys, feature)
            #根据这一列中的每个值，尝试对数据集进行拆分
            feature_candidate_gain = sys.float_info.max if len(y_categories) > 1 else 0.0
            feature_candidate_theshold = None
            feature_candidate_indice = ([],[])
            for i in range(len(medians)): #遍历特征中位数值
                value = medians[i]
                index1,index2,category1,category2 = divideset(X, y, feature, value, 
                    threshold_indice=threshold_indice, median_indice=unique_threshold_indice,
                    threshold_category=threshold_category, threshold_keys=y_keys, use_index=True) #根据该第feature个特征的阈值进行划分
                p = float(len(index1))/(len(index1) + len(index2))
                gain = information_gain - p*entropy(category1) - (1-p)*entropy(category2)
                gain = max(1-np.tanh(gain), 0) #使用tah进行统一使用
                cache_feature['value'].append(value)
                cache_feature['gain'].append(gain)
                if gain < feature_candidate_gain: # 加入到候选
                    feature_candidate_gain = gain
                    feature_candidate_indice = (index1,index2)
                    feature_candidate_theshold = value
            candidate_feature.append(feature)
            candidate_gain.append(feature_candidate_gain)
            candidate_indice.append(feature_candidate_indice)
            candidate_theshold.append(feature_candidate_theshold)
            cache_table[feature] = cache_feature
        return candidate_feature, candidate_theshold, candidate_gain, candidate_indice, cache_table
    
# 计算基尼系数，返回：根据基尼选择最好的特征下标，分割值，最大熵降，最佳划分
def CART(X, y, feature=-1, threshold=None):
    candidate_gini = []
    candidate_feature = []
    candidate_theshold = []
    candidate_indice = [] #[(indice_x1, indice_x2)]
    feature_count = len(X[0]) #特征数
    y_categories = unique_count(y)
    y_keys = sorted(list(y_categories.keys()))
    if feature >= 0 and threshold is not None: #给定特征与阈值，直接计算分裂
        index1,index2 = divideset(X, y, feature, threshold, use_index=True) #根据给定的特征和阈值进行划分
        y1,y2 = y[index1],y[index2]
        p = float(len(index1))/(len(index1) + len(index2))
        category1,category2 = unique_count(y1),unique_count(y2)
        feature_value_gini = p*gini(category1)+(1-p)*gini(category2)
        return feature_value_gini, (index1,index2)
    else:
        cache_table = {} #{feature_index:{v1:gain,v2:gain,...,vn:gain}
        feature_indice = list(range(0, feature_count))
        np.random.shuffle(feature_indice)
        for feature in feature_indice: #遍历特征
            cache_feature = {'value':[],'gain':[]} #已经排好顺序
            medians, threshold_indice, unique_threshold_indice, threshold_category =\
                caching_feature(X, y, y_keys, feature)
            #根据这一列中的每个值，尝试对数据集进行拆分
            feature_candidate_gini = sys.float_info.max if len(y_categories) > 1 else 0.0
            feature_candidate_theshold = -1
            feature_candidate_indice = ([],[])
            for i in range(len(medians)): #遍历特征中位数值
                value = medians[i]
                index1,index2,category1,category2 = divideset(X, y, feature, value, 
                    threshold_indice=threshold_indice, median_indice=unique_threshold_indice,
                    threshold_category=threshold_category, threshold_keys=y_keys, use_index=True) #根据该第feature个特征的阈值进行划分
                p = float(len(index1))/(len(index1) + len(index2))
                feature_value_gini = p*gini(category1)+(1-p)*gini(category2)
                cache_feature['value'].append(value)
                cache_feature['gain'].append(feature_value_gini)
                if feature_value_gini < feature_candidate_gini:
                    feature_candidate_gini = feature_value_gini
                    feature_candidate_indice = (index1,index2)
                    feature_candidate_theshold = value
            candidate_feature.append(feature)
            candidate_gini.append(feature_candidate_gini)
            candidate_indice.append(feature_candidate_indice)
            candidate_theshold.append(feature_candidate_theshold)
            cache_table[feature] = cache_feature
        return candidate_feature,candidate_theshold,candidate_gini,candidate_indice,cache_table
    
class Node:
    def __init__(self,
                 left_child=None,
                 right_child=None,
                 content={}
                ):
        self.tag = str(uuid.uuid4()) #唯一的标记
        self.left_child = left_child
        self.right_child = right_child
        self.content = {}
        self.content['is_leave'] = False
        self.content['feature'] = -2 #Feature used for splitting the node
        self.content['threshold'] = -1 #Threshold value at the node
        self.content['impurity'] = 0 #Impurity of the node (i.e., the value of the criterion)
        self.content['n_sample'] = 0 #number of support sample
        self.content['value'] = [] #value for all target
        self.content['output_label'] = [] #if is_leave [possitive, negative]
        self.content['output_prob'] = [] #if is_leave [0.7, 0.3]
        self.content['local_impurity'] = 0 #Rashomon Effect for local impurity
        self.content['local_value'] = [] #Rashomon Effect for local values
        self.content['rashomon'] = None #Whether Rashomon Effect is existed T/F/None(leave)
        if content:
            for k in content:
                self.content[k] = content[k]
                
class DecisionTreeClassifier(BaseEstimator):
    def __init__(self,
                 criterion='id3', #only 'id3','c45','gini' are supported.
                 splitter='best', #only 'best','random'
                 build_method='dfs', #only 'bfs','dfs'
                 max_features=None, #int, float or {“sqrt”, “log2”}, default=None work when splitter is 'random'
                 min_impurity_decrease=0.0, #允许分裂最低增益/熵降
                 max_depth=-1, #允许全局树最深度，与min_impurity_decrease是or关系，出现一者即停止分裂
                 min_samples_split=2, #The minimum number of samples required to split an internal node
                 feature_name=None,
                 class_name=None,
                 random_state=0,
                 logs=True):
        self.criterion = criterion
        self.splitter = splitter
        self.build_method = build_method
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.feature_name = feature_name
        self.feature_importance = None
        self.class_name = class_name
        self.classes_ = None
        self._n_features = 0
        self.root = None #if none mean not fit
        self.logs = logs
        np.random.seed(random_state)
    
    # 建树的过程
    # 以递归方式构造树
    # 每个递归计算一次熵降，然后再进入分支继续进行递归
    # 结束条件：1.数据耗尽 2.熵不再下降
    def _build_dfs(self, X, y, current_depth=-1):
        if len(X)==0 : return None
        reach_max_depth = False
        if current_depth >= self.max_depth and current_depth > 0 and self.max_depth > 0:
            reach_max_depth = True
        is_leave,res = self._split(X, y, min_impurity_decrease=self.min_impurity_decrease, 
                                   min_samples_split=self.min_samples_split, is_leave=reach_max_depth)
        if is_leave: #基尼系数为0或者所有特征收益相近，表示没有显著的区分特征或者已经完全划分
            return Node(content=res)
        else:
            c,data1,data2 = res[0],res[1],res[2]
            if data1 is not None:
                left_child = self._build_dfs(data1[0], data1[1], current_depth+1)
            if data2 is not None:
                right_child = self._build_dfs(data2[0], data2[1], current_depth+1)
            return Node(left_child=left_child, right_child=right_child, content=c)

    # 建树的过程
    # 以BFS方式构造树
    # 结束条件：1.数据耗尽 2.没有节点的返回
    def _build_bfs(self, X, y):
        if len(X)==0 : return None
        q = queue.Queue()
        root = Node()
        q.put((root, X, y, 1))
        while(not q.empty()):
            (node, x_, y_, current_depth) = q.get()
            reach_max_depth = False
            if current_depth >= self.max_depth and current_depth > 0 and self.max_depth > 0:
                reach_max_depth = True
            is_leave,res = self._split(x_, y_, min_impurity_decrease=self.min_impurity_decrease, 
                                   min_samples_split=self.min_samples_split, is_leave=reach_max_depth)
            if is_leave:
                node.content = {**node.content, **res}
            else:
                c,data1,data2 = res[0],res[1],res[2]
                node.content = c
                if data1 is not None:
                    left_child = Node()
                    node.left_child = left_child
                    q.put((left_child, data1[0], data1[1], current_depth+1))
                if data2 is not None:
                    right_child = Node()
                    node.right_child = right_child
                    q.put((right_child, data2[0], data2[1], current_depth+1))
        return root
            
    # 以熵降的过程计算分裂点
    # 输入：待观察数据与标签，is_leave：是否强制转化为叶子
    # 注意：XY为全局标签（用于CSTree），xy为局部标签配合罗生门系数[0,1]（全局是指最终结果，局部是指某层的分类策略）
    # 当xy为空则只考虑全局优化
    # 输出1：False, 节点内容，数据1，数据2
    # 输出2：True, 节点内容
    def _split(self, X, Y, min_impurity_decrease=0.0, min_samples_split=2, rashomon=1.0, rashomon_splitter='best', x=None, y=None, is_leave=False):
        if len(X)==0 : return True,None
        criterion = ID3 if self.criterion.lower() == 'id3' else CART
        candidate_feature,candidate_theshold,candidate_gain,candidate_indice,cache_table = criterion(X, Y)
        #按照gain对所有的candidate进行排序
        candidate = zip(candidate_gain, candidate_feature, candidate_theshold, candidate_indice)
        candidate_sort = sorted(candidate, key = lambda x:x[0], reverse=False) #loss采用最大的，越接近0越好
        candidate_gain,candidate_feature,candidate_theshold,candidate_indice = zip(*candidate_sort) #解压排序后的数组们
        #最终根据best_****进行分裂树，这个时候有多种条件，有局部标签与无局部标签，记录在标记is_rashomon
        best_feature, best_theshold, best_gain, best_split = \
            candidate_feature[0],candidate_theshold[0],candidate_gain[0],candidate_indice[0]#全局四元组
        is_rashomon = False #罗生门标记
        best_local_feature,best_local_theshold,best_local_gain,best_local_split = -1,-1,0,None #局部四元组
        if not is_leave:
            if x is not None and y is not None and best_gain != sys.float_info.max: #存在局部约束
                assert rashomon > 0.0 and rashomon <= 1.0
                rashomon_gain = best_gain / rashomon #罗生门阀值
                local_feature,local_theshold,local_gain,local_split,_ = criterion(x, y) #局部的特征与最佳分裂值
                #在罗生门阀值上的最佳局部的特征与最佳分裂值，标记为-1表示不存在
                best_local_global_gain = sys.float_info.max
                best_local_global_index = None
                #罗生门中的罗生门效应，拟合策略时候出现罗生门
                local_candidate = zip(local_gain,local_feature,local_theshold,local_split)
                local_candidate_sort = sorted(local_candidate, reverse=False) #loss采用最大的，越接近0越好，reverse=False为递增
                local_gain,local_feature,local_theshold,local_split = zip(*local_candidate_sort) #解压排序后的数组们
                local_index_vaild,local_gain_vaild = [],[] #计算能够有多少个局部分裂的增益能够在罗生门增益之内，有必然不为空 {index:gain}
                for lf,lth,lg,ls,i in zip(local_feature,local_theshold,local_gain,local_split,list(range(len(local_gain)))): #计算局部特征在全局下的增益/基尼
                    global_gain = search_in_cache(cache_table, lf, lth) #在缓存中寻找当前特征与分裂值下的全局增益
                    if global_gain is None: #没有击中缓存
                        global_gain,_ = criterion(X,Y,lf,lth)
                    if global_gain <= rashomon_gain: #寻找在罗生门阀值上的最佳局部的特征与最佳分裂值 （条件1）
                        local_index_vaild.append(i)
                        local_gain_vaild.append(global_gain)
                if len(local_index_vaild) > 0: #存在符合罗生门增益的分裂
                    rashomon_index,rashomon_gain = None,None
                    if rashomon_splitter == 'best': #取对全局影响最高的
                        rashomon_index,rashomon_gain = local_index_vaild[0],local_gain_vaild[0]
                    else:
                        local_topk = np.random.randint(0, len(local_index_vaild), size=1)[0]
                        rashomon_index,rashomon_gain = local_index_vaild[local_topk],local_gain_vaild[local_topk]
                    global_index1,global_index2 = divideset(X, Y, local_feature[rashomon_index], local_theshold[rashomon_index], use_index=True) #在满足全局条件下更新全局特征与最佳分裂值（需要重新计算，因为index适用的x不同）
                    best_local_feature,best_local_theshold,best_local_gain,best_local_split = \
                        local_feature[rashomon_index],local_theshold[rashomon_index],rashomon_gain,(global_index1,global_index2)
                if best_local_feature > -1 and best_local_split is not None:
                    is_rashomon = True #不存在罗生门阀值上的最佳局部的特征与最佳分裂值，改为全局模式
                    best_feature,best_theshold,best_gain,best_split = \
                        best_local_feature,best_local_theshold,best_local_gain,best_local_split
            if is_rashomon==False or x is None or y is None: #罗生门现象不明显或者没有约束，转为全局模式
                if self.splitter == 'best':
                    best_feature,best_theshold,best_gain,best_split = \
                        candidate_feature[0],candidate_theshold[0],candidate_gain[0],candidate_indice[0]
                else:
                    topk = 1
                    if isinstance(self.max_features, int): #取前max_features名
                        topk = min(self.max_features, self._n_features)
                    elif isinstance(self.max_features, float): #取前max_features% 【0，1】
                        topk = max(int(self.max_features * self._n_features), 1)
                    elif self.max_features == 'sqrt':
                        topk = max(int(self._n_features ** 0.5), 1)
                    elif self.max_features == 'log':
                        topk = max(int(np.log(self._n_features)), 1)
                    if topk > 1:
                        topk = np.random.randint(0, topk, size=1)[0]
                    best_feature,best_theshold,best_gain,best_split = \
                            candidate_feature[topk],candidate_theshold[topk],candidate_gain[topk],candidate_indice[topk]
        global_value = self._split_get_value(Y, (self.class_name if self.class_name is not None else list(self.classes_)))
        local_value = []
        if 'clusters' in self.__dict__ and 'strategies' in self.__dict__ and y is not None:
            local_value = self._split_get_value(y, unique_label(self.clusters[self._current_strategy_layer+1].labels_))
        condition1 = best_gain >= min_impurity_decrease and best_gain != sys.float_info.max
        condition2 = not (all(gain <= 0.0 for gain in candidate_gain) or all(gain == sys.float_info.max for gain in candidate_gain)) #当存在有gain>threshold而且<max时候，依旧需要分裂
        condition3 = min(len(best_split[0]), len(best_split[1])) >= min_samples_split
        condition4 = is_leave==False
        if condition1 and condition2 and condition3 and condition4:
            self.feature_importance[best_feature] += best_gain
            content={'is_leave':False, 'feature':best_feature, 'threshold':best_theshold, 'impurity':best_gain,
                     'n_sample':len(best_split[0])+len(best_split[1]), 'value':global_value, 
                     'local_value':local_value, 'local_impurity':best_local_gain, 'rashomon':is_rashomon}
            if len(best_split[0]) > 0 and len(best_split[1]) > 0:
                if is_rashomon:
                    return False,(content, 
                                 (X[best_split[0]], Y[best_split[0]], x[best_local_split[0]], y[best_local_split[0]]), #左节点
                                 (X[best_split[1]], Y[best_split[1]], x[best_local_split[1]], y[best_local_split[1]])) #右节点
                else:
                    return False,(content, 
                                 (X[best_split[0]], Y[best_split[0]], None, None),
                                 (X[best_split[1]], Y[best_split[1]], None, None))
            elif len(best_split[0]) > 0: #只存在左节点
                if is_rashomon:
                    return False,(content, (X[best_split[0]], Y[best_split[0]], x[best_local_split[0]], y[best_local_split[0]]), None)
                else:
                    return False,(content, (X[best_split[0]], Y[best_split[0]], None, None), None)
            elif len(best_split[1]) > 0: #只存在右节点
                if is_rashomon:
                    return False,(content, None, (X[best_split[1]], Y[best_split[1]], x[best_local_split[1]], y[best_local_split[1]]))
                else:
                    return False,(content, None, (X[best_split[1]], Y[best_split[1]], None, None))
        else: #基尼系数为0或者所有特征收益相近，表示没有显著的区分特征或者已经完全划分；叶子结点没有罗生门效应
            y_categories = unique_count(Y, category=list(self.classes_))
            counts = np.sum([y_categories[y_] for y_ in y_categories])
            if len(y_categories)==1: #完全划分，该节点只有一个种类
                return True,{'is_leave':True, 'value':global_value, 'local_value':local_value, 
                'output_label':list(y_categories.keys()), 'output_prob':[1.0], 
                'n_sample':counts, 'impurity':best_gain, 'local_impurity':best_local_gain, 
                'threshold':-2, 'rashomon':None}
            else: #没能完全划分，特征不足
                labels,probs = [],[]
                for y_,count in sorted(y_categories.items(), key=lambda item:item[1], reverse=True):
                    labels.append(y_)
                    probs.append(count/counts)
                return True,{'is_leave':True, 'value':global_value, 'local_value':local_value, 
                'output_label':labels, 'output_prob':probs, 
                'n_sample':counts, 'impurity':best_gain, 'local_impurity':best_local_gain, 
                'threshold':-2, 'rashomon':None}
    
    # 配合split
    def _split_get_value(self, y, class_name):
        value_ = [] #value的定义
        current_class = unique_count(y) #原有的普通决策树分类
        current_label = unique_label(y, sort=True) 
        for i in range(len(class_name)):
            if i in current_label: #当前数据流不一定含有所有的标签
                value_.append(float(current_class[i]))
            else:
                value_.append(0.0)
        return value_
    
    # 获取深度
    def get_depth(self, node):
        if node is None:
            return 0
        else:
            return max(self.get_depth(node.left_child), self.get_depth(node.right_child))+1
            
    # 根据给定的节点node获取对应的子节点
    def _get_leave(self, root):
        leave = []
        q = queue.Queue() #采用bfs
        q.put(root)
        while(not q.empty()):
            node = q.get()
            if node.left_child is None and node.right_child is None:
                 leave.append(node)
            else:
                if node.left_child is not None:
                    q.put(node.left_child)
                if node.right_child is not None:
                    q.put(node.right_child)
        return leave
    
    # 根据id寻找相应的节点
    def _find_node_by_tag(self, tag):
        nodes = []
        preorder_dfs(self.root, nodes)
        for i in range(len(nodes)):
            if nodes[i].tag == tag:
                return nodes[i]
        return None

    # 根据根节点到叶子的路径，对数据进行过滤
    # 输入：X全局，INDEX下标
    # 返回下标（全局）
    def _filter_by_node(self, root, leave, X, INDEX):
        paths = [] #记录从root到leave上的节点
        self._find_leave(root, leave, paths) #获取路径
        index = INDEX #输出（全局下标）
        if len(paths)==0:
            return None
        else:
            paths.reverse() #从根高层到低层
            #print([p.content for p in paths])
            for i in range(len(paths)):
                if i+1 < len(paths):
                    index1,index2 = divideset(X[index], None, paths[i].content['feature'], paths[i].content['threshold'], use_index=True)
                    assert paths[i+1] == paths[i].left_child or paths[i+1] == paths[i].right_child
                    if paths[i+1] == paths[i].left_child:
                        index = index[index1]
                    else:
                        index = index[index2]
                else:
                    return index
    
    # 配合_filter_by_node工作
    def _find_leave(self, node, leave, paths):
        if node is None or leave is None: return False
        if node.tag != leave.tag:
            if node.left_child is None and node.right_child is None:
                return False
            else:
                if self._find_leave(node.left_child, leave, paths):
                    paths.append(node)
                    return True
                elif self._find_leave(node.right_child, leave, paths):
                    paths.append(node)
                    return True
                else:
                    return False
        else:
            paths.append(node)
            return True
    
    #预测函数，根绝给定的x预测，从node开始
    def _classify(self, x, node):
        if node.content['is_leave']: #叶子结点
            return node.content['output_label'],node.content['output_prob']
        else:
            value = x[node.content['feature']]
            next_child = None
            if isinstance(value,int) or isinstance(value,float):
                if value < node.content['threshold']: 
                    next_child = node.left_child
                else: 
                    next_child = node.right_child
            else:
                if value==node.content['threshold']: 
                    next_child = node.left_child
                else: 
                    next_child = node.right_child
            return self._classify(x, next_child)
    
    #打印函数
    def _print_node(self, node, indent, feature_name=None):
        if node.content['is_leave']:
            print('*' + ''.join(indent) + ' : ' + str(node.content['n_sample']) + ' '
                  + str({l:np.round(p,3) for l,p in zip(node.content['output_label'], node.content['output_prob'])}) 
                  + ' ' + self.criterion + ' : ' + str(np.round(node.content['impurity'],3)) + ' value: ' + str(node.content['value']))
        else:
            if isinstance(node.content['threshold'], int) and isinstance(node.content['threshold'], float):
                if feature_name:
                    print(' ' + ''.join(indent) + ' : ' + feature_name[node.content['feature']] +
                          ' < ' + str(np.round(node.content['threshold'], 3)) + ' ' + self.criterion + ' : ' + str(np.round(node.content['impurity'],3)) + ' value: ' + str(node.content['value']))
                else:
                    print(' ' + ''.join(indent) + ' : ' + 'feature_' + str(node.content['feature']) + 
                          ' < '  + str(np.round(node.content['threshold'], 3)) + ' ' + self.criterion + ' : ' + str(np.round(node.content['impurity'],3)) + ' value: ' + str(node.content['value']))
            else:
                if feature_name:
                    print(' ' + ''.join(indent) + ' : ' + feature_name[node.content['feature']] + 
                          ' < ' + str(np.round(node.content['threshold'], 3)) + ' ' + self.criterion + ' : ' + str(np.round(node.content['impurity'],3)) + ' value: ' + str(node.content['value']))
                else:
                    print(' ' + ''.join(indent) + ' : ' + 'feature_' + str(node.content['feature']) + 
                          ' < '  + str(np.round(node.content['threshold'], 3)) + ' ' + self.criterion + ' : ' + str(np.round(node.content['impurity'],3)) + ' value: ' + str(node.content['value']))
        
    def fit(self, X, y): 
        X, y = check_X_y(X, y)
        self.classes_ = unique_count(y)
        if len(X) == 0:
            self._n_features = 0
            self.feature_importance = []
            self.root = Node()
        else:
            self._n_features = len(X[0])
            self.feature_importance = [0] * len(X[0])
            assert self.build_method.lower() == 'bfs' or self.build_method.lower() == 'dfs'
            if self.build_method == 'bfs':
                self.root = self._build_bfs(np.array(X), np.array(y))
            else:
                self.root = self._build_dfs(np.array(X), np.array(y), 1)
        return self
        
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        assert len(X) > 0 and len(X[0]) == self._n_features
        result = []
        for i in range(len(X)):
            result.append(self._classify(X[i], self.root)[0][0])
        return result
        
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        assert len(X) > 0 and len(X[0]) == self._n_features
        result = []
        for i in range(len(X)):
            label,prob = self._classify(X[i], self.root)
            label_prob = zip(label, prob)
            label_prob_sort = sorted(label_prob) #ascend according to the label
            label_,prob_ = zip(*label_prob_sort)
            result.append(list(prob_))
        return result
        
    def score(self, X, y):
        assert len(X) == len(y)
        res = self.predict(X)
        return np.sum(np.array(np.array(res)==np.array(y), dtype=np.int))/len(y)
    
    def export_text(self, from_node=None,feature_name=None):
        depth = 0
        q = queue.Queue()
        if from_node is None:
            q.put((self.root, depth))
        else:
            q.put((from_node, depth))
        indent = []
        if feature_name:
            assert len(feature_name) == self._n_features
        while(not q.empty()):
            node,depth = q.get()
            indent = ' |' * depth
            self._print_node(node, indent, feature_name)
            if node.left_child: q.put((node.left_child, depth+1))
            if node.right_child: q.put((node.right_child, depth+1))
                      
class CSNode(Node):
    def __init__(self,
                 left_child=None,
                 right_child=None,
                 strategy=None, #在该层下的策略，该节点如果为空，则没有被分配到合理的策略（对应-1）
                 strategy_layer=-1, #策略所属于的层次 
                 content={}, #实际内容
                ):
        super(CSNode, self).__init__(left_child, right_child, content)
        self.strategy = strategy
        self.strategy_layer = strategy_layer
        self.strategy_suspend = False #记录是否策略中断
        self.debug = {} #debug

class CSTreeClassifier(DecisionTreeClassifier):
    def __init__(self,
                criterion='id3', #only 'id3','c45','gini' are supported.
                splitter='best', #only 'best','random'
                build_method='cstree', #only 'cstree'
                max_features=None, #int, float or {“sqrt”, “log2”}, default=None work when splitter is 'random'
                max_depth=-1, #允许的最大深度（全局）
                min_impurity_decrease=0.0, #允许的最少熵减（用于建立树）【float或者其长度跟X一致数组】
                min_samples_split=2, #允许的最少分裂点数（用于建立树）【float或者其长度跟X一致数组】
                max_substree_depth=2, #允许建立策略子树时候的最大深度【int或者其长度跟X一致数组】
                min_rest_depth=2, #允许最小保真层数：max_depth-策略所占层数 
                min_similarity=0.6, #允许策略匹配中节点系数=D&G/D【float或者其长度跟X一致数组】
                min_jaccob=0.01, #允许策略进行的最少流入样本数=D&G/D|G【float或者其长度跟X一致数组】
                rashomon=0.8, #罗生门指数（候选特征的全局熵降不能低于最优特征的熵降的占比【float或者其长度跟X一致数组】
                rashomon_splitter='best', #only 'best','random' for local x,y
                clusters=None, #层次聚类【长度跟X一致数组，内容是聚类】#必须与策略层同时提供，如果提供，则不尽兴build_strategy
                strategies=None, #策略层【长度跟X一致数组，内容是策略】#
                strategy_type='topk', #only 'threshold' and 'topk' is allow
                strategy_param={'k1':2, 'k2':2, 'clustering':DBSCAN, 'cluster_parmas':{'n_cluster':3}}, #策略采用的聚类器的参数以及输出阈值
                is_strategy_continuous=False, #如果遇到策略断层，后续若果匹配成功后是否还继续进行策略划分
                feature_name=None,
                class_name=None,
                random_state=0,
                logs=1):
        super(CSTreeClassifier, self).__init__(criterion, splitter, build_method, max_features, min_impurity_decrease, 
                                    max_depth, min_samples_split, feature_name, class_name, random_state, logs)
        assert type(min_impurity_decrease)==float or type(min_impurity_decrease)==list
        assert type(max_substree_depth)==int or type(max_substree_depth)==list
        assert type(min_similarity)==float or type(min_similarity)==list
        assert type(min_jaccob)==float or type(min_jaccob)==list
        assert type(rashomon)==float or type(rashomon)==list
        assert max_substree_depth >= 1
        self.strategy_type = strategy_type
        self.strategy_param = strategy_param
        self.max_substree_depth = max_substree_depth
        self.min_rest_depth = min_rest_depth
        self.min_similarity = min_similarity
        self.min_jaccob = min_jaccob
        self.strategies = strategies
        self.clusters = clusters
        self.rashomon = rashomon
        self.rashomon_splitter = rashomon_splitter
        self._current_strategy_layer=None #辅助使用
        self.is_strategy_continuous = is_strategy_continuous
            
    # 建树的过程
    # 以BFS+strategy方式构造树
    # 输入：X数据，Y标签
    # 结束条件：1.数据耗尽 2.没有节点的返回
    def _build(self, X, Y, clusters=None, strategies=None, logs=1):
        if len(X)==0 : return None
        # 建立Strategy
        self.strategy_param['X'] = X
        if clusters is None or strategies is None:
            clusters,strategies = self._build_strategy(X, logs)
        self.clusters,self.strategies = clusters,strategies #缓存簇与策略
        groups = self._build_groups(X, clusters) #获取所有层数据聚类分布
        if type(self.min_impurity_decrease) != list:
            self.min_impurity_decrease = [self.min_impurity_decrease for i in range(len(strategies)+1)]
        if type(self.min_samples_split) != list:
            self.min_samples_split = [self.min_samples_split for i in range(len(strategies)+1)]
        current_group = groups[0] #获取当前数据聚类分布如簇1[1,3,4],簇2[2,5]
        # BFS建立树
        q = queue.Queue()
        root = CSNode() #根节点（只有一个聚类）
        #三元组<node（仅分配了根节点聚类，输入不设置聚类）, 数据输入流下标, 当前所属策略层>
        q.put((root, np.arange(len(X[0]), dtype=np.int), 0)) 
        last_strategy_layer = 0
        slibling_num = 0 #记录位于该层的下标
        while(not q.empty()):
            (node, data_index, current_strategy_layer) = q.get()
            assert len(data_index) > 0 #如果没有数据流经该节点，将失去意义
            #根据流经本节点的数据流匹配最适合的策略
            self._current_strategy_layer = current_strategy_layer
            if last_strategy_layer != current_strategy_layer:
                current_group = groups[current_strategy_layer] #必须为全局数据，获取当前数据聚类分布如簇1[1,3,4],簇2[2,5]
                last_strategy_layer = current_strategy_layer
                slibling_num = 0
            #根据当前数据流data_index和current_groups如[1,3,4], [2,5] #代表簇0、簇1，（全局）匹配最适合的s（root）
            slibling_num += 1
            current_depth = -1 #建树之前必须满足深度
            rest_depth = MAX_DEPTH_LIMIT #系统最大限制深度
            if self.max_depth > 0:
                paths = [] 
                self._find_leave(root, node, paths)
                current_depth = len(paths) #路径长度即为深度(root to node)
                rest_depth = self.max_depth - current_depth
            if rest_depth > 0: #有充足的允许深度
                #策略层比截面层少1，并且策略层还在允许范围中，而且不在策略断层
                if current_strategy_layer < len(X)-1 and rest_depth > self.min_rest_depth:
                    if not node.strategy_suspend or self.is_strategy_continuous:
                        matching_index,matching_s1,matching_s2,s1_,s2_ = find_strategy_by_data(strategies[current_strategy_layer], 
                                                            data_index, current_group, self.min_similarity, self.min_jaccob)
                        if matching_index > -1:#存在匹配策略的下标
                            if current_strategy_layer < len(strategies) and current_strategy_layer < len(clusters)-1: #最后一层判断
                                if logs > 0:
                                    print('-------------------------------------------')
                                    print('current_strategy_layer: ' + str(current_strategy_layer) + ' slibling_num: ' + str(slibling_num))
                                    print('matching_index: ' + str(matching_index) +
                                          ' matching_s1: ' + str(matching_s1) + ' matching_s2: ' + str(matching_s2) + ' ' + 
                                          str(s1_) + ' ' + str(s2_))
                                    print('matching strategy: ' + str(strategies[current_strategy_layer][matching_index].__dict__))
                                current_cluster = clusters[current_strategy_layer+1]
                                node.strategy_layer = current_strategy_layer
                                node.strategy = strategies[current_strategy_layer][matching_index]
                                next_group = groups[current_strategy_layer+1]
                                agg_index,agg_label = self._aggregate(current_group, next_group, data_index) #获取所有涉及到的簇的数据点
                                x,y = X[0][data_index], Y[data_index]
                                x_c,y_c = X[0][agg_index], agg_label
                                if logs > 1:
                                    print('building subtree ' + 'for ' + str(len(x)) \
                                            + ' data for ' + str(len(current_cluster.cluster_centers_)) + ' categories.')
                                self._build_subtree(node, X=x, Y=y, 
                                                    min_impurity_decrease = self.min_impurity_decrease[current_strategy_layer],
                                                    min_samples_split = self.min_samples_split[current_strategy_layer],
                                                    max_depth = min(self.max_substree_depth, rest_depth-self.min_rest_depth),
                                                    rashomon = self.rashomon, rashomon_splitter=self.rashomon_splitter, x=x_c, y=y_c) #根据匹配提供的策略聚类进行输出，并且分配了聚类
                                # 设置策略标记
                                self.set_strategy_info(node, strategies[current_strategy_layer][matching_index], current_strategy_layer)
                                leaves = self._get_leave(node)
                                for leaf in leaves:
                                    leaf_index = self._filter_by_node(node, leaf, X[0], data_index) #过滤出流从节点node到leaf的数据下标
                                    assert len(leaf_index) > 0 #所有节点都应该有数据流经过，否则建树没有意义
                                    q.put((leaf, leaf_index, current_strategy_layer+1)) #层次再+1
                                if self.logs > 1:
                                    self.export_text(from_node=node, feature_name=self.feature_name) #打印当前策略生成子树（node）
                                continue
                        else:
                            if logs > 1:
                                print('current_strategy_layer: ' + str(current_strategy_layer) + ' slibling_num: ' + str(slibling_num))
                                print('no matching: ' + str(matching_index) +
                                      ' matching_s1: ' + str(matching_s1) + ' matching_s2: ' + str(matching_s2) + 
                                      ' s1 ' + str(s1_) + ' s2 ' + str(s2_))
                #不存在匹配，直接输出，并且不再加入队列？
                #print('no matching_index...' + ' matching_s1: ' + str(matching_s1) + ' matching_s2: ' + str(matching_s2))
                node.strategy, node.strategy_layer, node.content['rashomon'] = None,-1,False #清除策略标记
                if rest_depth == MAX_DEPTH_LIMIT: #无限制
                    self._build_subtree(node, X[0][data_index], Y[data_index], self.min_impurity_decrease[-1], 
                                        self.min_samples_split[-1], max_depth=MAX_DEPTH_LIMIT-current_depth) 
                else:
                    self._build_subtree(node, X[0][data_index], Y[data_index], self.min_impurity_decrease[-1], 
                                        self.min_samples_split[-1], max_depth=rest_depth)
        return root    
    
    # 建立策略
    def _build_strategy(self, X, logs=True):
        # 建立Strategy
        if logs > 1:
            print('-------------------------------------------')
            print('building strategy ...')
            print('by param:')
        assert self.strategy_type == 'threshold' or self.strategy_type == 'topk'
        if self.strategy_type == 'threshold':
            clusters,strategies = build_strategy_threshold(**self.strategy_param)
        else:
            clusters,strategies = build_strategy_topk(**self.strategy_param)
        if logs > 1:
            if len(strategies) > 0:
                print('print strategies built ...')
                for c,s in zip(clusters[1:], strategies):
                    print(c)
                    for j in range(len(s)):
                        print(s[j].__dict__)
        return clusters,strategies
           
    #获取不同层的数据聚类分布如第2层：簇1[1,3,4],簇2[2,5]
    def _build_groups(self, X, clusters):
        groups = []
        for i in range(len(clusters)):
            group = get_data_matching(X[i], clusters[i])
            groups.append(group)
        return groups
           
    # 聚合计算，根据当前数据流包含的簇，返回属于簇的数据点(x,y)
    # 例如：当前数据流包含簇[1,2,3]，那么我们应该返回（x[c1]+x[c2]+x[c3], y[c1]+y[c2]+y[c3]）
    def _aggregate(self, current_groups, next_groups, data_index):
        index = [] #返回属于簇的聚类下标
        label = [] #返回属于簇的类别
        contain_cluster = set()
        for i in range(len(data_index)):
            for j in range(len(current_groups)):
                if data_index[i] in current_groups[j]:
                    contain_cluster.add(j) #记录涉及到簇
                    break
        contain_cluster = sorted(list(contain_cluster)) #囊括所属簇的所有下标（顺序已经被打乱，因此需要排序）
        for i in range(len(contain_cluster)): 
            index += current_groups[contain_cluster[i]]
        for i in range(len(index)):
            for j in range(len(next_groups)):
                if index[i] in next_groups[j]:
                    label.append(j)
                    break
        assert len(index)==len(label)
        return np.array(index, dtype=np.int),np.array(label, dtype=np.int)
    
    # 根据给定的节点node、数据X和标签y，按照罗生门比率进行创建，dfs创建即可
    # Y为全局标签，y为局部簇标签，优先根据罗生门比率计算全局最低熵降阈值，在阈值的基础上匹配y
    def _build_subtree(self, node, X, Y, min_impurity_decrease=0, min_samples_split=2, max_depth=3, \
                        rashomon=1.0, rashomon_splitter='best',x=None, y=None):
        if len(X)==0 : return None
        dfs_tree = self._build_dfs(X, Y, min_impurity_decrease, min_samples_split, max_depth, rashomon, rashomon_splitter, x, y, current_depth=1)
        if dfs_tree is not None:
            node.left_child = dfs_tree.left_child
            node.right_child = dfs_tree.right_child
            node.content = dfs_tree.content
    
    # 递归方式建立分叉，属于build_subtree，参考_build_subtree
    def _build_dfs(self, X, Y, min_impurity_decrease=0, min_samples_split=2, max_depth=5, \
                    rashomon=1.0, rashomon_splitter='best', x=None, y=None, current_depth=-1):
        if len(X)==0 : return None
        if rashomon < 1.0 and (x is None or y is None): return None #罗生门不为1时候属于策略过程中，但由于x和y已经划分完毕，所以停止建立子树
        reach_max_depth = False
        if current_depth > max_depth and max_depth > 0: #最大深度抑制
            reach_max_depth = True
        is_leave,res = self._split(X, Y, min_impurity_decrease, min_samples_split, rashomon, rashomon_splitter, x, y, is_leave=reach_max_depth) 
        if is_leave: #基尼系数为0或者所有特征收益相近，表示没有显著的区分特征或者已经完全划分
            return CSNode(content=res)
        else:
            c,data1,data2 = res[0],res[1],res[2]
            left_child,right_child = None,None
            if data1 is not None:
                left_child = self._build_dfs(data1[0], data1[1], min_impurity_decrease, min_samples_split, max_depth, \
                                                rashomon, rashomon_splitter, data1[2], data1[3], current_depth+1)
            if data2 is not None:
                right_child = self._build_dfs(data2[0], data2[1], min_impurity_decrease, min_samples_split, max_depth, \
                                                rashomon, rashomon_splitter, data2[2], data2[3], current_depth+1)
            return CSNode(left_child=left_child, right_child=right_child, content=c)
    
    # 为node以及其所有孩子节点都标记
    def set_strategy_info(self, node, strategy, strategy_layer):
        nodes = []
        preorder_dfs(node, nodes)
        for i in range(len(nodes)):
            nodes[i].strategy = strategy
            nodes[i].strategy_layer = strategy_layer
            if nodes[i].content['rashomon']==False: #不存在罗生门效应
                nodes[i].strategy_suspend = True #产生策略断层                
    
    def fit(self, X, y):
        X_check = []
        for i in range(len(X)):
            x, y = check_X_y(X[i], y)
            X_check.append(x)
        X = X_check
        self.classes_ = unique_count(y)
        if type(self.min_impurity_decrease) == list:
            assert len(self.min_impurity_decrease)==len(X)
        if type(self.min_samples_split) == list:
            assert len(self.min_samples_split)==len(X)
        if type(self.max_substree_depth) == list:
            assert len(self.max_substree_depth)==len(X)
        if type(self.rashomon) == list:
            assert len(self.rashomon)==len(X)
        if type(self.min_similarity) == list:
            assert len(self.min_similarity)==len(X)
        if type(self.min_jaccob) == list:
            assert len(self.min_jaccob)==len(X)
        if len(X) == 0:
            self._n_features = 0
            self.feature_importance = []
            self.root = Node()
        else:
            self._n_features = len(X[0][0])
            self.feature_importance = [0] * len(X[0][0])
            self.root = self._build(X, np.array(y), self.clusters, self.strategies, self.logs)
        return self


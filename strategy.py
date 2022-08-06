import sys
import numpy as np
from sklearn.cluster import KMeans,DBSCAN

#### 定义多功能聚类
MAX_CLUSTER_IERTATIONS = 5 #最大聚类实验次数

#### 定义Strategy
class Strategy:
    def __init__(self, G_root):
        self.G_root = G_root #group ID
        self.G_child = [] #group ID
        self.S_root = 0 #策略总分
        self.S_child = [] #分裂分数
        self.S_conn = [] #链接分数，与child下标一致

def unique_label(labels, sort=False):
    label_set = set()
    if labels is not None:
        for l in labels:
            label_set.add(l)
    if sort:
        label_set = list(label_set)
        label_set.sort(reverse=False)
    return label_set

def try_ncluster(X, n_cluster, other_param={}, init_eps='mean', max_cluster_iterations=MAX_CLUSTER_IERTATIONS, logs=False):
    eps = 0
    last_eps = 0
    best_eps = 0
    try_labels = -1
    best_labels = -1
    test_cluster = None
    best_cluster = None
    excepted_cluster = 0 #-1,0,1
    if init_eps == 'mean':
        eps = mean_point_distance(X) #用均值
    else:
        assert type(init_eps)==float
        eps = init_eps
    for i in range(max_cluster_iterations):
        other_param['eps'] = eps
        test_cluster = DBSCAN(**other_param).fit(X)
        labels = unique_label(test_cluster.labels_, sort=False)
        if logs:
            print('last_eps: ' + str(last_eps) + ' eps: ' + str(eps) + ' u_label: ' + str(len(labels)))
        if (excepted_cluster > 0 and len(labels) < try_labels) or (excepted_cluster < 0 and len(labels) > try_labels):
            if logs:
                print('unexcepted cluster: ' + str(len(labels)) + ' which is not: ' + str(excepted_cluster))
            break
        if abs(len(labels) - n_cluster) <= abs(best_labels - n_cluster) and len(labels) > best_labels: #仅当有更新时候，当有两者距离相等时，取较大者
            best_cluster = test_cluster
            best_labels = len(labels)
            best_eps = eps
        try_labels = len(labels)
        tmp = last_eps
        last_eps = eps    
        if try_labels > n_cluster: #半径太小
            if tmp > eps:
                eps = (eps + tmp)/2
            else:
                eps *= 2
            excepted_cluster = -1 #期待try_labels变小
        elif try_labels < n_cluster: #半径太大
            if tmp > eps:
                eps /= 2
            else:
                eps = (eps + tmp)/2
            excepted_cluster = 1 #期待try_labels变大
        else:
            break 
    return best_eps,best_labels,best_cluster

#### 定义多功能聚类
class CLUSTER:
    def __init__(self, cluster, params, logs=False):
        assert cluster==KMeans or cluster==DBSCAN
        self.cluster = cluster
        self.params = params
        self.cluster_centers_ = []
        self.labels_ = None
        self.datas_ = None
        self.logs = logs
        
    def fit(self, X):
        assert len(X) > 1
        if type(X) != np.ndarray:
            self.datas_ = np.array(X) #[n, d]
        else:
            self.datas_ = X
        if self.cluster == KMeans:
            indice = list(range(0,len(X)))
            if 'sampling' in self.params: #仅在kmeans上可使用sampling
                assert type(self.params['sampling'])==float or type(self.params['sampling'])==int
                if type(self.params['sampling'])==float:
                    indice = np.random.choice((indice), int(len(X)*self.params['sampling']+1), replace=False)
                else:
                    indice = np.random.choice((indice), self.params['sampling'], replace=False)
                del self.params['sampling']
            self.params['init'] = 'k-means++'
            if 'n_cluster' in self.params:
                self.params['n_clusters'] = self.params['n_cluster']
                del self.params['n_cluster']
            print('trying ' + str(self.params['n_clusters']) + ' clustering in KMeans')
            self.cluster = self.cluster(**self.params).fit(X[indice])
            self.cluster_centers_ = self.cluster.cluster_centers_
            self.labels_ = self.cluster.predict(X)
        else:
            eps = -1
            if 'n_clusters' in self.params: #尽量的自动化调参数
                print('trying ' + str(self.params['n_clusters']) + ' clustering in DBSCAN')
                eps,try_labels,test_cluster = try_ncluster(X, self.params['n_clusters'], logs=self.logs)
                self.cluster = test_cluster
                self.params['eps'] = eps
                #print('try_result: ' + str(try_labels))
            else:
                if 'epsr' in self.params:
                    eps = self.params['epsr'] * max_point_distance(X) #用比例
                else:
                    eps = self.params['eps']
                params_ = {}
                for k in self.params:
                    if k == 'epsr':
                        params_['eps'] = eps
                    else:
                        params_[k] = self.params[k]
                self.params = params_
                self.cluster = self.cluster(**self.params).fit(X)
            labels = self.cluster.labels_
            if -1 in labels: #不使用-1作为计数
                labels += 1
            self.labels_ = labels
            label_points = {}
            for i in range(len(labels)):
                if labels[i] not in label_points:
                    label_points[labels[i]] = []
                label_points[labels[i]].append(X[i])
            u_labels = unique_label(labels, sort=True)
            centorids = []
            for i in range(len(u_labels)):
                centorids.append(np.mean(label_points[u_labels[i]], axis=0))
            self.cluster_centers_ = np.stack(centorids)
            print('dbscan_result: ' + str(len(unique_label(self.labels_))))
        return self
    
    # 采用self.label
    # 用于predict的sampling
    # 优点：快速
    # 缺点：不稳定
    def sampling(self, sample_rate=None):
        labels = unique_label(self.labels_, sort=True)
        categories = [[] for i in range(len(labels))]
        sample_categories = [[] for i in range(len(labels))]
        for i in range(len(self.labels_)):
            categories[self.labels_[i]].append(i)
        for i in range(len(categories)):
            sample_count = 0
            if sample_rate == None:
                sample_count = int(len(categories[i]) ** 0.5)+1 #没有参数，则采样sample_count**0.5
            elif sample_rate == 1.0:
            	sample_count = len(categories[i])
            else:
                sample_count = int(len(categories[i]) * sample_rate)+1
            sub_categories = list(range(len(categories[i])))
            indice = np.random.choice(sub_categories, min(sample_count, len(sub_categories)), replace=False) #对于每一个分类进行采样
            for j in range(len(indice)): 
                sample_categories[i].append(categories[i][indice[j]])
        datas_,indice = [],[]
        for i in range(len(sample_categories)):
            for j in range(len(sample_categories[i])):
                datas_.append(self.datas_[sample_categories[i][j]])
                indice.append(sample_categories[i][j])
        return datas_,indice

    def predict(self, X):
        if 'predict' in dir(self.cluster):
            return self.cluster.predict(X)
        else:
            #不存在预测函数，寻找最近值
            label = []
            for i in range(len(X)):
                index = -1
                distance = sys.float_info.max
                #anchors = self.cluster_centers_
                anchors = self.datas_
                for j in range(len(anchors)):
                    d = np.linalg.norm(X[i]-anchors[j])
                    if d < distance:
                        distance = d
                        index = j
                        # if d <= self.params['eps']: #在半径内则直接判定（存在bug，不稳定）
                        #     break 
                assert index >= 0
                label.append(self.labels_[index])
            label = np.stack(label)
            return label 

'''
根据阈值建立策略函数（目前只允许DNSCAN）
输入：
X: X= X0,X1,...,Xn #待模仿的神经网络按照高到低的输出分布，可以跳层
cluster_parmas: cluster的参数
B: 能形成分裂的最低成绩
beta: 孩子聚类轮廓成绩的次方 [0-1]越小，轮廓系数越重要 (必须有，推荐0.5)
A: 能形成策略的最低成绩
alpha: 根聚类轮廓成绩的次方 [0-1]越小，轮廓系数越重要 (必须有，推荐0.5)
is_input_clustering: 第一层是否使用聚类，否则n_clusters=1
输出：
Strategy二维数组，第一维度=n，第二维度等于各级策略数
Cluster一维度数组，每一个维度是一个聚类（包括输入聚类，因此比Strategy要长）
'''
def build_strategy_threshold(X, clustering, A=1e-1, B=1e-2, alpha=0.5, beta=0.5, gamma=2,
    cluster_parmas={}, is_input_clustering=False, logs=False):
    C = [] #聚类数据
    S = [] #策略数组
    assert len(X) > 1
    if type(A) != list:
        A = [A] * len(X)
    if type(B) != list:
        B = [B] * len(X)
    if type(alpha) != list:
        alpha = [alpha] * len(X)
    if type(beta) != list:
        beta = [beta] * len(X)
    if type(gamma) != list:
        gamma = [gamma] * len(X)
    if type(clustering) != list:
        clustering = [clustering] * len(X)
    if len(k1) == len(X)-1:
        k1.insert(0, 0)
    if len(k2) == len(X)-1:
        k2.insert(0, 0)
    if len(alpha) == len(X)-1:
        alpha.insert(0, 0)
    if len(beta) == len(X)-1:
        beta.insert(0, 0)
    if len(gamma) == len(X)-1:
        gamma.insert(0, 0)    
    if type(cluster_parmas) == dict:
        cluster_parmas = [cluster_parmas] * len(X)
    if len(cluster_parmas) == len(X)-1:
        cluster_parmas.insert(0, {})
    if is_input_clustering:
        cluster_ = CLUSTER(clustering[0], cluster_parmas[0], logs=logs).fit(X[0]) #cluster为聚类器，带_为输入，没有则为输出
    else:
        cluster_ = KMeans(n_clusters=1).fit(X[0])
        clustering.insert(0, cluster_)
    print('input ' + ' : ' + str(cluster_)) 
    C.append(cluster_)
    for i in range(1, len(X)):
        Si = []
        cluster = CLUSTER(clustering[i], cluster_parmas[i], logs=logs).fit(X[i])
        group_,group = unique_label(cluster_.labels_, sort=True),unique_label(cluster.labels_, sort=True)  #G_为输入聚类，G为输出聚类
        print('layer ' + str(i) + ' c: ' + str(cluster.cluster) + ' g: ' + str(len(group)) + ' A: ' + str(A[i]) + ' B: ' + str(B[i])
            + ' alpha: ' + str(alpha[i]) + ' beta: ' + str(beta[i]))
        #计算单个类的轮廓系数
        silhouette_score_ = np.array([silhouette(cluster_, g_, X[i]) for g_ in group_])
        silhouette_score = np.array([silhouette(cluster, g, X[i]) for g in group])
        for j in range(len(group_)): #对每一个输入进行策略计算
            s_ = Strategy(group_[j])
            conenction_score = np.array([connection(cluster_, cluster, group_[j], g, gamma[i]) for g in group]) #计算给定输入下所有输出的连接强度
            split_score = (silhouette_score**beta[i])*conenction_score #候选成绩数组
            for k in range(len(split_score)):
                if split_score[k] > B[i]: #大于B阈值才能定为一个分裂
                    s_.G_child.append(group[k]) 
                    s_.S_child.append(np.round(split_score[k]), 4) #孩子成绩是指分裂成绩：由轮廓系数与链接系数的组合
                    s_.S_conn.append(np.round(conenction_score[k], 4))
            if len(s_.G_child) > 0:
                s_.S_root = np.round((silhouette_score_[j]**alpha[i])*np.mean(s_.S_child), 4)
                if s_.S_root > A[i]:
                    Si.append(s_)
        cluster_ = cluster
        C.append(cluster)
        S.append(Si)
    return C,S #S比C要短一层


'''
根据topk建立策略函数（目前只允许DNSCAN）
输入：
X: X= X0,X1,...,Xn #待模仿的神经网络按照高到低的输出分布，可以跳层
cluster_parmas: cluster的参数
gamma: 确定数量的重要性(A & B) [1, +) 越大越重要，默认为2
k1: 根据分裂成绩由高到低取top k1 #内部分支
beta: 孩子聚类轮廓成绩的次方 [0-1]越小，轮廓系数越重要 (必须有，推荐0.5)
k2: 根据策略成绩由高到低取top #外部分支
alpha: 根聚类轮廓成绩的次方 [0-1]越小，轮廓系数越重要 (必须有，推荐0.5)
is_input_clustering: 第一层是否使用聚类，否则n_clusters=1
输出：
Strategy二维数组，第一维度=n，第二维度等于各级策略数
Cluster一维度数组，每一个维度是一个聚类（包括输入聚类，因此比Strategy要长）
'''
def build_strategy_topk(X, clustering, k1=2, k2=2, alpha=0.5, beta=0.5, gamma=2,
    cluster_parmas={}, is_input_clustering=False, logs=False):
    C = [] #聚类数据
    S = [] #策略数组
    assert len(X) > 1 #少于两层的分析没有意义
    if type(k1) != list:
        k1 = [k1] * len(X)
    if type(k2) != list:
        k2 = [k2] * len(X)
    if type(alpha) != list:
        alpha = [alpha] * len(X)
    if type(beta) != list:
        beta = [beta] * len(X)
    if type(gamma) != list:
        gamma = [gamma] * len(X)
    if type(clustering) != list:
        clustering = [clustering] * len(X)
    if type(cluster_parmas) == dict:
        cluster_parmas = [cluster_parmas] * len(X)
    if len(k1) == len(X)-1:
        k1.insert(0, 0)
    if len(k2) == len(X)-1:
        k2.insert(0, 0)
    if len(alpha) == len(X)-1:
        alpha.insert(0, 0)
    if len(beta) == len(X)-1:
        beta.insert(0, 0)
    if len(gamma) == len(X)-1:
        gamma.insert(0, 0)
    if len(cluster_parmas) == len(X)-1:
        cluster_parmas.insert(0, {}) 
    if is_input_clustering:
        cluster_ = CLUSTER(clustering[0], cluster_parmas[0], logs=logs).fit(X[0]) #cluster为聚类器，带_为输入，没有则为输出
    else:
        cluster_ = KMeans(n_clusters=1).fit(X[0])
        clustering.insert(0, cluster_)
    print('input ' + ' : ' + str(cluster_)) 
    C.append(cluster_)
    for i in range(1, len(X)): #每一层 O(L)
        Si = []
        cluster = CLUSTER(clustering[i], cluster_parmas[i], logs=logs).fit(X[i])
        group_,group = unique_label(cluster_.labels_, sort=True),unique_label(cluster.labels_, sort=True)  #G_为输入聚类，G为输出聚类
        print('layer ' + str(i) + ' c: ' + str(cluster.cluster) + ' g: ' + str(len(group))  + ' k1: ' + str(k1[i]) + ' k2: ' + str(k2[i]) 
            + ' alpha: ' + str(alpha[i]) + ' beta: ' + str(beta[i]))
        #计算单个类的轮廓系数
        silhouette_score_ = np.array([silhouette(cluster_, g_, X[i]) for g_ in group_])
        silhouette_score = np.array([silhouette(cluster, g, X[i]) for g in group])
        for j in range(len(group_)): #对每一个输入进行策略计算 O(C)
            s_ = Strategy(group_[j])
            conenction_score = np.array([connection(cluster_, cluster, group_[j], g, gamma[i]) for g in group]) #计算给定输入下所有输出的连接强度
            split_score = (silhouette_score**beta[i])*conenction_score #候选成绩数组
            sort_index = sorted(range(len(split_score)), key=lambda k: split_score[k], reverse=True) #获取候选成绩的排序下标
            for k in range(min(k1[i], len(sort_index))):
                if split_score[sort_index[k]] == 0:
                    break
                s_.G_child.append(group[sort_index[k]]) 
                s_.S_child.append(np.round(split_score[sort_index[k]], 4)) #孩子成绩是指分裂成绩：由轮廓系数与链接系数的组合
                s_.S_conn.append(np.round(conenction_score[sort_index[k]], 4))
            if len(s_.G_child) > 0:
                s_.S_root = np.round((silhouette_score_[j]**alpha[i])*np.mean(s_.S_child), 4)
                Si.append(s_)
        cluster_ = cluster
        C.append(cluster)
        Si_ = [] #筛选topk
        if len(Si) > 0: #每一层的策略加入
            sort_index = sorted(range(len(Si)), key=lambda k: Si[k].S_root, reverse=True)
            for k in range(min(k2[i], len(sort_index))):
                if Si[k].S_root == 0:
                    break
                Si_.append(Si[sort_index[k]])
        S.append(Si_)
    return C,S #S比C要短一层


'''
输入：
当前聚类cluster
当前簇下标==簇label（group）
所有的数据点（point）
输出：
当前簇的轮廓系数
'''
def silhouette(cluster, group, point):
    if len(cluster.cluster_centers_) == 1: #只有一个簇的时候则认为该簇是完美的
        return 1
    index,in_distance,out_distance = [],[],[] #index:点在point上的下标,in_distance:对应的内部
    for i in range(len(cluster.labels_)):
        if cluster.labels_[i] == group:
            out_distance.append(0)
            in_distance.append(0)
            index.append(i)
    assert len(index) > 0
    assert len(cluster.labels_) == len(point)
    closet_group = -1
    closet_distance = sys.float_info.max
    for i in range(len(cluster.cluster_centers_)):
        if group != i:
            if np.linalg.norm(cluster.cluster_centers_[i]-cluster.cluster_centers_[group]) < closet_distance:
                closet_distance = np.linalg.norm(cluster.cluster_centers_[i]-cluster.cluster_centers_[group])
                closet_group = i
    assert closet_group >= 0
    #print(closet_group)
    closet_group_count = 0
    for i in range(len(cluster.labels_)): #最近簇元素数量
        if cluster.labels_[i] == closet_group:
            closet_group_count += 1
    for i in range(len(index)):
        for j in range(len(cluster.labels_)):
            if cluster.labels_[j] == group: #簇内
                in_distance[i] += np.linalg.norm(point[index[i]]-point[j])
            elif cluster.labels_[j] == closet_group: #簇外
                out_distance[i] += np.linalg.norm(point[index[i]]-point[j])
    in_distance,out_distance = np.array(in_distance, dtype=np.float),np.array(out_distance, dtype=np.float)
    in_distance /= len(index)
    out_distance /= closet_group_count
    #计算轮廓系数
    s = np.zeros_like(index, dtype=np.float)
    for i in range(len(s)):
        if out_distance[i] <= in_distance[i]:
            s[i] = 0
        else:
            s[i] = (out_distance[i]-in_distance[i])/np.max([out_distance[i], in_distance[i]])
    return np.mean(s)
    
'''
输入：
当前出发聚类cluster_
当前终点聚类cluster
当前出发簇下标==簇label（group_）
当前终点簇下标==簇label（group）
gamma：参考build_strategy_系列
输出：
当前出发簇-终点簇的连接强度 (出发&终点/出发)
'''    
def connection(cluster_, cluster, group_, group, gamma=2):
    index_ = set()
    index = set()
    assert len(cluster_.labels_) == len(cluster.labels_)
    for i in range(len(cluster_.labels_)):
        if cluster_.labels_[i] == group_:
            index_.add(i)
        if cluster.labels_[i] == group:
            index.add(i)
    return (len(index_&index)**gamma)/len(index_)

'''
点之间的平均距离
'''
def mean_point_distance(points):
    assert len(points) > 1
    mean_distance = 0
    for i in range(len(points)):
        for j in range(len(points)):
            mean_distance += np.linalg.norm(points[i]-points[j])
    return mean_distance/(len(points)*(len(points)-1))

'''
点之间的最大距离
'''
def max_point_distance(points):
    assert len(points) > 1
    max_distance = 0
    for i in range(len(points)):
        for j in range(len(points)):
            d = np.linalg.norm(points[i]-points[j])
            if d > max_distance:
                max_distance = d
    return max_distance

'''
簇之间的最大距离
'''
def max_cluster_center(cluster):
    max_distance = 0
    assert len(cluster.cluster_centers_) > 1
    for i in range(len(cluster.cluster_centers_)):
        for j in range(len(cluster.cluster_centers_)):
            d = np.linalg.norm(cluster.cluster_centers_[i]-cluster.cluster_centers_[j])
            if d > max_distance:
                max_distance = d
    return max_distance
            
'''
显著系数，衡量策略中的分裂距离（簇中心平均两两距离/簇中心最大两两距离）
delta = mean(si)/max(si)
输入：Strategy，该层的聚类（child）
输出：显著系数[0,1]
'''
def distinguish(strategy, cluster):
    if len(strategy.G_child) == 1:
        return 1
    max_distance = max_cluster_center(cluster)
    #print('--------------start testing--------------')
    distance = [max_distance]*len(strategy.G_child)
    for i in range(len(strategy.G_child)):
        for j in range(len(strategy.G_child)):
            if i != j:
                if strategy.G_child[i] != strategy.G_child[j]:
                    d = np.linalg.norm(cluster.cluster_centers_[strategy.G_child[i]]-cluster.cluster_centers_[strategy.G_child[j]])
                    if d < distance[i]:
                        distance[i] = d
    #print(np.mean(distance), max_distance)
    #print('--------------end testing--------------')
    return np.mean(distance)/max_distance

'''
#获取当前数据聚类分布如簇1[1,3,4],簇2[2,5]
#输入：X, 聚类：包括中心0，1
#输出：[1,3,4], [2,5] #代表簇0、簇1
'''
def get_data_matching(data, cluster):
    labels = cluster.predict(data) #labels可能不能覆盖从0到最后
    unique_count = len(unique_label(cluster.labels_, sort=True))
    c = [[] for i in range(unique_count)]
    for i in range(len(labels)):
        c[labels[i]].append(i)
    return c

'''
根据当前数据流data以及data_matching待匹配的策略的数据分布，匹配最合适的策略
条件：数据分布的相似度
输入：stregies当前策略层，data数据流（某个节点的输入）的下标，待匹配的策略的数据分布（data_matching）#与策略下标一致
输出：最合适的策略的下标，参考相似度
'''
def find_strategy_by_data(stregies, data, data_matching, min_simi=0.75, min_jaccob=0.1):
    simi = 0
    index = -1
    jaccob = 0
    s1_,s2_ = [],[]
    for i in range(len(stregies)):
        assert stregies[i].G_root < len(data_matching) #聚类id必须在全局内
        d = data_matching[stregies[i].G_root]
        s1 = len(set(data) & set(d))/len(set(data))
        s2 = len(set(data) & set(d))/len(set(data) | set(d))
        s1_.append(np.round(s1, 4))
        s2_.append(np.round(s2, 4))
        if s1 > simi and s2 > min_jaccob:
            simi = np.round(s1, 4)
            jaccob = np.round(s2, 4)
            index = i
    if simi > min_simi:
        return index,simi,jaccob,s1_,s2_
    else:
        return -1,simi,jaccob,s1_,s2_

                        
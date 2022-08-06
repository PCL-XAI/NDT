import numpy as np
import matplotlib as mpl
import sklearn_json as skljson
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn import tree
from cstree import CSTreeClassifier
from tool import preorder_dfs,to_skl_tree


strategy_colors = ['red', 'blue', 'darkgreen', 'orange', 'pink', 'purple']

def unique_label(labels, sort=False):
    label_set = set()
    if labels is not None:
        for l in labels:
            label_set.add(l)
    if sort:
        label_set = list(label_set)
        label_set.sort(reverse=False)
    return label_set

def widen(group1, group2, k=2):
    center1,center2 = np.mean(group1, axis=0),np.mean(group2, axis=0)
    center = (center1+center2)/2
    vector1,vector2 = center1-center, center2-center
    vector1_,vector2_ = vector1*k,vector2*k
    bias1,bias2 = vector1_- vector1,vector2_-vector2
    return bias1,bias2

def denoise(data1, data2, noise=0.01):
    noise_number = 0
    assert len(data1) == len(data2)
    assert type(noise)==float or type(noise)==int
    if type(noise) == float:
        assert noise < 1.0 and noise >= 0.0
        noise_number = int(len(data1)*noise)
    else:
        assert noise < len(data1) and noise >= 0
        noise_number = noise
    dist1,indice1 = distance(data1)
    dist2,indice2 = distance(data2)
    del_indice = []
    i1,i2 = 0,0
    for i in range(noise_number):
        if dist1[i1] > dist2[i2]:
            del_indice.append(indice1[i1])
            i1 += 1
        else:
            del_indice.append(indice2[i2])
            i2 += 1
    keep_indice = list(set(range(len(data1))) - set(del_indice))
    return data1[keep_indice], data2[keep_indice], keep_indice

# 去除一些距离中心远的点
def distance(data):
    center = np.mean(data, axis=0)
    distance = np.linalg.norm(data-center, axis=1)
    data_by_dist = np.stack([np.linalg.norm(data-center, axis=1), list(range(len(data)))]).T
    data_by_dist_ = np.array(sorted(data_by_dist, key=lambda x:x[0], reverse=True))
    return data_by_dist_[:,0],data_by_dist_[:,1]
# [:noise_number,1]

# draw arrows
def arrow(x, y, dx, dy, color1, color2, size=0.5, alpha=1, gain=1e-2, min_gain=1000):
    d = int((dx**2+dy**2)**0.5 / gain)
    d = d if d > min_gain else min_gain
    x = np.linspace(x, x+dx, d)
    y = np.linspace(y, y+dy, d)
    c = np.linspace(color1, color2, d)
    #print(d)
    plt.scatter(x, y, c=c, marker='.', s=size, cmap='RdBu', alpha=alpha)

# draw flow
def flow(from_x, to_x, from_y, to_y, from_label=None, to_label=None, from_center=None, to_center=None, strategy=None, margin_scale=1e-1, figsize=(12,12), draw_arrow=True, arrow_size=0.1):
    assert len(from_x) == len(to_x)
    assert len(unique_label(from_label)) <= len(list(mcolors.TABLEAU_COLORS)) and len(unique_label(to_label)) <= len(list(mcolors.TABLEAU_COLORS))
    plt.figure(figsize=figsize)
    colors = ['deeppink', 'lightsalmon'] #input
    #arrow_colors = ['tab:orange', 'tab:green', 'tab:pink', 'tab:purple', 'tab:brown', 'tab:blue', 'tab:olive', 'tab:cyan']
    arrow_colors = list(mcolors.TABLEAU_COLORS)
    plt.scatter(from_x, from_y, color=colors[0], alpha=0.2, s=100, edgecolors=colors[1])
    legend1,legend2 = [None] * len(unique_label(from_label)),[None] * len(unique_label(to_label))
    if from_label is not None: #来源的聚类
        for i in range(len(from_label)):
            plt.scatter(from_x[i], from_y[i], color=list(mcolors.TABLEAU_COLORS)[from_label[i]], alpha=0.4, s=15)
    else:
        plt.scatter(from_x, from_y, color=colors[0], alpha=0.33, s=15)
    if from_center is not None: #来源的聚类中心
        assert len(from_center) <= len(mcolors.TABLEAU_COLORS) 
        for i in range(len(from_center)):
            legend1[i] = plt.scatter(from_center[i,0], from_center[i,1], color=list(mcolors.TABLEAU_COLORS)[i], s=100, marker='P', edgecolors='w')
    colors = ['dodgerblue', 'deepskyblue'] #output
    plt.scatter(to_x, to_y, color=colors[1], alpha=0.2, s=100, edgecolors=colors[1])
    if to_label is not None: #终点的聚类
        for i in range(len(to_label)):
            plt.scatter(to_x[i], to_y[i], color=list(mcolors.TABLEAU_COLORS)[to_label[i]], alpha=0.4, s=15)
    else:
        plt.scatter(to_x, to_y, color=colors[0], alpha=0.33, s=15)
    if to_center is not None:#终点的聚类中心
        assert len(to_center) <= len(mcolors.TABLEAU_COLORS) 
        for i in range(len(to_center)):
            legend2[i] = plt.scatter(to_center[i,0], to_center[i,1], color=list(mcolors.TABLEAU_COLORS)[i], s=100, marker='X', edgecolors='w')
    strong = {} #{'2-3':0.34,'3-5':0.12}，记录root->child以及分裂分数
    if strategy: #描绘箭头
        for i in range(len(strategy)):
            s = strategy[i]
            for j in range(len(s.G_child)):
                strong[str(s.G_root) + '-' + str(s.G_child[j])] = s.S_child[j]
    if draw_arrow:
        for i in range(len(from_x)):
            arrow(from_x[i], from_y[i], to_x[i]-from_x[i], to_y[i]-from_y[i], color1=0.3, color2=0.7, size=0.05, alpha=0.01)
    x_width = (np.max([from_x,to_x]) - np.min([from_x,to_x])) * margin_scale
    y_width = (np.max([from_y,to_y]) - np.min([from_y,to_y])) * margin_scale
    xy_width = np.linalg.norm([x_width, y_width])
    for s in strong: #根据策略画趋势箭头
        head,tail = int(s.split('-')[0]),int(s.split('-')[1])
        dx,dy = (to_center[tail][0]-from_center[head][0])*0.5,(to_center[tail][1]-from_center[head][1])*0.5
        plt.arrow(x=from_center[head][0]+(to_center[tail][0]-from_center[head][0])*0.25, y=from_center[head][1]+(to_center[tail][1]-from_center[head][1])*0.25, 
                  dx=dx, dy=dy, width=np.max([xy_width*arrow_size, np.min([xy_width*arrow_size*2, np.linalg.norm([dx,dy]) * np.tanh(strong[s])**3])]), 
                  facecolor=arrow_colors[head], edgecolor='white', alpha=0.75)
    plt.xlim(np.min([from_x,to_x])-x_width, np.max([from_x,to_x])+x_width)
    plt.ylim(np.min([from_y,to_y])-y_width, np.max([from_y,to_y])+y_width)
    unique_,unique = list(unique_label(from_label)),list(unique_label(to_label))
    if to_label is not None: #画图例
        if from_label is not None:
            plt.legend(legend1 + legend2, ['from_'+str(unique_[i]) for i in range(len(unique_))] + ['to_'+str(unique[i]) for i in range(len(unique))], ncol=2)
        else:
            plt.legend(legend2, list(unique_label(to_label)))
    plt.show()
    plt.close()

def cstree_visual(cs_tree, artists, max_alpha=1.0, min_alpha=0.0):
    preorder_nodes = []
    preorder_dfs(cs_tree.root, preorder_nodes)
    assert len(preorder_nodes)==len(artists)
    assert max_alpha >= 0.0 and max_alpha <= 1.0
    assert min_alpha >= 0.0 and min_alpha <= 1.0
    assert max_alpha > min_alpha
    for i in range(len(preorder_nodes)):
        text = ''
        node = preorder_nodes[i]
        box = artists[i].get_bbox_patch()
        #box.set_boxstyle('round', rounding_size=1.0)
        if node.content['feature'] >= 0:
            feature_name = cs_tree.feature_name[node.content['feature']]
            if len(feature_name) > 10:
                feature_name = feature_name[:9] + '...'
            text += feature_name + '<=' + str(np.round(node.content['threshold'], 3)) + '\n' 
        if node.content['is_leave']:
            index_ = node.content['value'].index(np.max(node.content['value']))
            #text += 'gain = ' + str(np.round(node.content['impurity'],4)) + '\n'
            text += 'value = ' + str(list(np.array(node.content['value'], dtype=np.int))) + '\n' 
            text += 'class = ' + str(cs_tree.class_name[index_])
            box.set_linestyle('dashed')
            if type(cs_tree)==CSTreeClassifier:
                box.set_facecolor('white')
        else:
            if type(cs_tree) != CSTreeClassifier:
                text += 'value = ' + str(list(np.array(node.content['value'], dtype=np.int)))
            else:
                if node.strategy is None:#非策略层
                    #text += 'gain = ' + str(np.round(node.content['impurity'],4)) + '\n'
                    text += 'value = ' + str(list(np.array(node.content['value'], dtype=np.int)))
                    box.set_facecolor('white')
                else:
                    #text += 'gain = ' + str(np.round(node.content['impurity'],4)) + '\n'
                    #text += 'local_gain = ' + str(np.round(node.content['local_impurity'],4)) + '\n'
                    text += 'value = ' + str(list(np.array(node.content['value'], dtype=np.int))) + '\n'
                    local_value_str = str(list(np.array(node.content['local_value'], dtype=np.int)))
                    if len(node.content['local_value']) > 4:
                        local_value_split = list(i for i,value in enumerate(local_value_str) if value == ',')
                        local_value_str = local_value_str[:local_value_split[2]] + '\n' + local_value_str[local_value_split[2]:]
                    text += 'local_value = \n' + local_value_str
                    base_color = strategy_colors[node.strategy_layer] 
                    intense = get_color_intense(node.content['local_value'])
                    if intense == -1:
                        box.set_facecolor('white')
                    else:
                        box.set_facecolor(color_gradient(base_color, 'white', min_alpha + (max_alpha-min_alpha)*intense))
        artists[i].set_text(text)

# 返回，区分值
# 4,0.33 前者为区分下标，后者为透明度
# -1 不能区分
def get_color_intense(value):
    if len(value) == 0: return -1
    value = np.array(value)
    index = np.where(value==np.max(value))[0] #index 是二维数组，坑
    if len(index) < len(value):
        return value[index[0]]/np.sum(value)
    else:
        return -1

def color_gradient(c1, c2, mix=0): 
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c2 + mix*c1)

# 打印树
def print_tree(csdtc, feature_names, class_names, title=None, max_alpha=0.9, min_alpha=0.5, figsize=(16,16), dpi=300):
    csdtc_ = skljson.from_dict(to_skl_tree(csdtc))
    fig,ax = plt.subplots(ncols=1, figsize=figsize, dpi=dpi)
    artists = tree.plot_tree(csdtc_, 
                       feature_names=feature_names,  
                       class_names=class_names,
                       filled=True,
                       node_ids=True,
                       impurity=True,
                       rounded=True,
                       fontsize=4,
                       ax=ax)
    ax.set_title(title, fontsize=20)
    cstree_visual(csdtc, artists, max_alpha=max_alpha, min_alpha=min_alpha)
    plt.tight_layout()
    plt.show()


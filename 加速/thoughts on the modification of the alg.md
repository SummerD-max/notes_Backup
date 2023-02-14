# thoughts on the modification of the alg FedCh

```python


SERVER:

client_models = [INIT_MODEL, INIT_MODEL ... INIT_MODEL]
global_model = INIT_MODEL


class cluster:
    def __init__(self, cluster_head, local_models_index):
        cluster_head_idx = index of cluster head
        local_models_index = [index_of local models]

    # 训练簇内的全部客户端
    def train():
        for index in local_models_index:
            Train client_models[index]

    def aggregate():
        for index in local_models_index:
            # 加权更新 cluster model
            client_models[cluster_head_idx] = alpha * client_models[cluster_head_idx] + beta * client_models[index] 

cluster_num = 5
num_Of_Clients_Each_Cluster = [3, 3, 3, 3, 3]

# 根据不同簇的计算能力定义的全局模型聚合顺序
agg_order = []

# 分好的簇
clusters = [cluster1, cluster2, cluster3, cluster4, cluster5]

# 需要迭代的次数
iterations = T

while iterations:
# 按照聚合顺序遍历每个簇
    for idx in agg_order:
        # 簇内每个客户端进行训练，并将模型聚合到 cluster head
        clusters[idx].train()
        cluster[idx].aggregate()
        head_model = client_models[clusters[idx].cluster_head_idx]
        agg(head_model, global_model)
    interations -= 1

# 返回经过 T 次聚合后的全局模型

return agg_model




```
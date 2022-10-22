# deepsnap 

- 官网地址：https://snap.stanford.edu/deepsnap/

## 什么是 deep snap？

> DeepSNAP is a Python library to assist efficient deep learning on graphs. DeepSNAP features in its support for flexible graph manipulation, standard pipeline, heterogeneous graphs and simple API.

> 是一个Python函数库，用来帮助我们在图(graphs)上面进行**有效率**的深度学习，它支持图的操作，标准化流程，异构图和简单的API

类似的工具包有  PyTorch Geometric, DGL, GraphNets。

## 一些比较重要的 module!

![img](https://img2022.cnblogs.com/blog/2145900/202210/2145900-20221022113414382-312993843.png)

### deepsnap.batch

![img](https://img2022.cnblogs.com/blog/2145900/202210/2145900-20221022113826677-847922111.png)

大概意思应该是说 Batch 类可以将一批 Graph 集合成一个大的（不连接的）图，但是具体的用法[官方文档](https://snap.stanford.edu/deepsnap/modules/batch.html#id1)上没讲.

### deepsnap.dataset

![img](https://img2022.cnblogs.com/blog/2145900/202210/2145900-20221022113849468-1900459283.png)

#### GraphDataset 类

`classGraphDataset(graphs: Optional[List[deepsnap.graph.Graph]] = None,...)`

- graphs (list, optional) – A list of deepsnap.graph.Graph.

用来存放图数据集列表

#### Generator 类

![img](https://img2022.cnblogs.com/blog/2145900/202210/2145900-20221022113903290-621383548.png)

能够动态地生成各种图（ER graphs 等等，可以用 ERGenerator类 继承 Generator类 然后去重写 `generate()` 方法，`generate()` 方法中定义了图的产生方式

#### EnsembleGenerator 类

![img](https://img2022.cnblogs.com/blog/2145900/202210/2145900-20221022113914811-980228255.png)

根据传进去的 generators 列表以 gen_prob 的概率生成 dataset_len 长度的图数据集。

### 重头戏来了：deepsnap.Graph 的格式

![img](https://img2022.cnblogs.com/blog/2145900/202210/2145900-20221022113939163-1687954588.png)



文档中说 `Graph.G` 是 **NetworkX or SnapX graph object**

在代码中输出 

`print(type(dataset[0].G))`
```
output:
<class 'networkx.classes.graph.Graph'>
```
发现它确实是一个 networkx 包定义的 graph，于是我去找 networkx 包

那么要如何生成 network 的 graph 呢？

[networkx官网](https://networkx.org/documentation/stable/reference/classes/graph.html#methods)上有讲如何生成一张空的图，并往图中加节点和边，明天再看看

![](https://ts1.cn.mm.bing.net/th/id/R-C.aec2cd5132ccc8e03b108441b1453280?rik=1z39BNADsjPPVA&riu=http%3a%2f%2fim6.leaderhero.com%2fwallpaper%2f20150505%2f37b31b4a1d.jpg&ehk=nKRUCwjkxiA6BjzdV%2bJgJwHyLO6XpOZMPMO036%2b5x3c%3d&risl=&pid=ImgRaw&r=0)
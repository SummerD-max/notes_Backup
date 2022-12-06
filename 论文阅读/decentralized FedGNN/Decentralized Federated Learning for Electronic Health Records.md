# Decentralized Federated Learning for Electronic Health Records

## 2020 54th Annual Conference on Information Sciences and Systems (CISS)


主要贡献是用到了 

- Decentralized Stochastic Gradient Descent (DSGD)
- Decentralized Stochastic Gradient Tracking (DSGT)

这两个梯度优化算法

![img](https://img2022.cnblogs.com/blog/2145900/202211/2145900-20221115170935496-1005480480.png)

联邦化的算法就是最简单的联邦学习，
i.e. **updates parameters locally for several times r and if r is a multiple of Q, then update parameters for every node based on their neighbor nodes.**

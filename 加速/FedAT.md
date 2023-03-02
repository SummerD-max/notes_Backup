# FedAT （有点像 Hierarchical Fedrated Learning）

![img](https://img2023.cnblogs.com/blog/2145900/202302/2145900-20230228100858810-1937790399.png)

## Constraint Term（保证模型不偏移）

![img](https://img2023.cnblogs.com/blog/2145900/202302/2145900-20230228094422453-458358020.png)

通过在本地更新中添加这个正则项，保证本地模型和全局模型相近。

$w_k$ 客户端k的模型 $w$全局模型。

## Cross-Tier Weighted Aggregation（设置权重）

![img](https://img2023.cnblogs.com/blog/2145900/202302/2145900-20230228095431423-1401202117.png)

FedAT uses a new cross-tier, weighted aggregation heuristic, which dynamically adjusts the relative weights assigned to each tier based on the number of times a tier has updated the global mode. The goal of the weighted aggregation heuristic is to **help the global training converge faster**.

- **a relatively slower tier with a tier number 𝑚 would get assigned a relatively larger weight value**

- **avoid potential bias towards a subset of faster tiers**

## Compression

As studied in [39], many compression methods [3, 40] suffer from slow convergence rates in the Non-i.i.d. cases. Non-i.i.d. often introduces divergence of model weights collected from resource-heterogeneous clients. Due to such divergence, some lossy compression methods such as quantization and dequantization [51] may inevitably lead to huge errors and reduce global performance.

使用了一个有损的压缩(lossy compression) - Encoded Polyline Algorithm

- It can be configured to maintain a configurable precision by rounding the value to a specified decimal place. With the appropriate precision, the model could achieve the largest communication saving and minor performance loss.

## 不采取强制更新策略
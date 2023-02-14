## 向 dataframe 中添加行有两种方法， iloc 和 loc

不管什么方法，都要事先准备一个与原数据**列的个数**相同的list或者是Series，比如`[1, 2, 3, 4]`

~~~python
# 先定义一个 DataFream
import pandas as pd
df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])

```
   0  1  2  3
0  1  2  3  4
1  5  6  7  8
```
~~~



### 1. iloc[]

用iloc向 dataframe 中添加行使用如下代码：

```python
# 向第[2]行添加
df.iloc[2] = [9, 10, 11, 12]
```

会报错 `IndexError: iloc cannot enlarge its target object`，原因是dataframe没有第[2]行, 而 iloc 只能在原先有数据的行进行修改，不能插入新的行。



### 2.loc[]

使用loc向 dataframe 添加行

```
df.loc[2] = [9, 10, 11, 12]
print(df)

# output

0	1	2	3
0	1	2	3	4
1	5	6	7	8
2	9	10	11	12

# 成功插入了

# 输出df的行index
df.index
Int64Index([0, 1, 2], dtype='int64')
```

在loc中向 dataframe 添加行的原理是，loc[2] 去找 index 为 2 的行，如果没有则创建一个index为2的行，然后再执行 `df.loc[2] = [9, 10, 11, 12]`
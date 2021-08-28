
# 随机数生成函数
## 1 简单随机数
### 1.1 random.rand(d0, d1, ..., dn)生成一个(d0, d1, ..., dn)维的数组，元素取自[0, 1)上的均匀分布。
### 1.2 random.randn(d0, d1, ..., dn)生成一个(d0, d1, ..., dn)维的数组，元素是标准正态分布随机数，若没有参数，返回单个数据。
### 1.3 random. random(size=None)生成一个[0,1)之间size个随机数。
### 1.4 random. randint (m, n, size)生成 m，n范围的数组，形状为 size (传入元组形式的shape)。
### 1.5 choice(arr, size=None, replace=True, p=None)从 arr 数组中选取 size 个随机数，replace=True 表示可重复抽取，p 是 arr 中每个元素出现的概率，p 的长度和 arr 的长度必须相同，且 p 中元素之和为1，否则报错；若 arr 是整数，则 arr 代表的数组是 np.arange(a) 。
## 2 随机分布

函数 | 功能
 - | :-
binomial(n, p, size=None) | 产生 size 个二项分布的样本值，n 表示 n 次试验，p 表示每次实验发生（成功）的概率。函数的返回值表示 n 次实验中发生（成功）的次数
exponential(scale=1.0, size=None) | 产生 size 个指数分布的样本值，这里的 scale 是 β，为标准差，β=1/λ，λ为单位时间内事件发生的次数
normal(loc=0.0, scale=1.0, size=None) | 产生 size 个正态分布的样本值，loc 为正态分布的均值，scale 为正态分布的标准差，size 缺省，则返回一个数
poisson(lam=1.0, size=None) | 从泊松分布中生成随机数，lam 是单位时间内事件的平均发生次数
uniform(low=0.0, high=1.0, size=None) | 产生 size 个均匀分布[low,high)的样本值，size 为 int 或元组类型，如 size = (m, n, k)，则输出 m*n*k 个样本，缺省时输出1个值

## 3 随机排列
### 3.1 shuffle(x)
打乱对象 x（多维数组按照第一维打乱），直接在原来的数组上进行操作，改变原来数组元素的顺序，无返回值，x 可以是数组或者列表。
### 3.2 permutation(x)
打乱并返回新对象（多维数组按照第一维打乱），不直接在原数组上进行操作，而是返回一个新的打乱元素顺序的数组，并不改变原来的数组，x 可以是整数或者列表，如果是整数k，那就随机打乱 numpy.arange(k)
## 4 随机数生成器

# 数组运算函数的使用方法
## 1 ndarray 算术运算
数组最常用的运算是算术运算，可以为数组的每个元素加上或乘以某个数值，可以让两个数组对应元素之间做加、减、乘、除运算等。

例：若a, b为两个ndarray对象：
 
a + 6，表示 a 中所有元素都加 6 

a * 2，表示 a 中所有元素都乘以 2 

a + b，a - b，a * b，a / b，分别表示a与b中对应元素相加、减、乘、除（不是矩阵运算）。 

类似 a > b，在两个数组之间使用条件运算符，将会返回一个布尔数组，对应元素之间条件运算为真，值为 True，否则为 False。
## 2 NumPy 算术运算函数
一元算术运算函数

函数  | 功能 | 函数  | 功能
- | :- | - | :-
abs()  | 计算各元素的绝对值 |  ceil()  | 对各元素向上取整
sqrt() |  计算各元素的平方根 |  floor()  | 对各元素向下取整
square()  | 计算各元素的平方 |  rint() |  对各元素四舍五入到正数，保留 dtype
exp()  | 计算各元素的指数 | ex modf() |  浮点数分解函数
log(), log10()  | 计算各元素的对数  | cos(), sin(), tan() |  对各元素进行三角函数求值，numpy.degrees()函数弧度转角度
sign() | 计算各元素的正负号： 1，0， -1 | arcos(), arcsin(), arctan() | 对各元素进行反三角函数求值


二元算术运算函数

函数  | 功能 
- | :-
add(a, b)  | 将两个数组中对应的元素相加，a、b为两个 ndarray 数组
subtract(a, b)  | 将两个数组中对应的元素相减
multiply(a, b)  | 两个数组中对应的元素相乘
power(a, b)  | 对第一个数组中的元素 x，第二个数组中的对应位置的元素 y，计算 x 的 y 次方
greate, greate_equal, less, less_equal, equal, not_equal | 将两个数组中对应的元素进行比较运算，最终产生布尔型的数组。相当于运算符>、>=、<、<=、==、!=

## 3 ndarray 线性代数运算
NumPy常用的线性代数运算函数
- np.diag(A) 以一维数组的形式返回方阵 A 的对角线元素
- np.dot(A, B) 矩阵乘法，或向量内积
- np.trace(A) 计算矩阵 A 对角线元素的和
- np. linalg.det(A) 计算矩阵 A 的行列式
- np. linalg.eig(A) 计算方阵 A 的特征值和特征向量
- np. linalg.inv(A) 计算方阵 A 的逆
- np. linalg.pinv(A) 计算矩阵 A 的 Moore-Penrose 伪逆
- np. linalg.qr(A) 计算 QR 分解
- np. linalg.svd(A) 对矩阵 A 进行奇异值分解（SVD）
- np. linalg.solve(A, b) 解线性方程组 Ax=b，其中 A 为一个方阵
## 4 NumPy 统计函数
函数  | 功能
 - | :-
nd.max(axis=None)  | 返回指定 axis 最大值
nd.min(axis=None)  | 返回指定 axis 最小值
nd.ptp(axis)  | 返回指定axis最大值与最小值的差，即极差
nd.sum(axis=None)  | 返回指定 axis 的所有元素的和，默认求所有元素的和
nd.mean(axis=None)  | 返回指定 axis 的数组元素均值
nd.var(axis=None)  | 根据指定的 axis 计算数组的方差
nd.std(axis=None)  | 根据指定 axis 计算数组的标准差

注：axis=None 时，数组被当成一维数组；axis=0表示对列进行操作；axis=1表示对行进行操作。

## 5 ndarray 元素排序和筛选
- ndarray.sort()函数排序

ndarray.sort(axis=-1, kind=’quicksort’,order=None)
例：
``` import numpy as np
y = np.array([1, 3, 4, 9, 8, 7, 6, 5, 3, 10, 2])
y.sort() # 原地排序，y被改变
print(y)
y1 = np.array([[0, 15, 10, 5],
 [25, 22, 3, 2],
 [55, 45, 59, 50]])
y1.sort() # 按行排序
print(y1)
```
输出结果：

[ 1 2 3 3 4 5 6 7 8 9 10]

[[ 0 5 10 15]

[ 2 3 22 25]

[45 50 55 59]]

- ndarray.argsort() 函数排序

ndarray.argsort(axis=-1, kind=’quicksort’,
 order=None)

例：
```import numpy as np
x = np.array([[0, 15, 10, 5],
 [25, 22, 3, 2],
 [55, 45, 59, 50]])
sorted_idx = x[:, 2].argsort()
y = x[sorted_idx]
print(y)
```
输出结果：
[[25 22 3 2]

[ 0 15 10 5]

[55 45 59 50]]
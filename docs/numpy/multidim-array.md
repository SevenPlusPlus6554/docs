
### 0.1 NumPy 是什么
NumPy 是 Python 的一个高效的、开放源码的数学运算扩展库，广泛应用于数值计算、机器学习、图像处理等领域。它的核心是 ndarray 多维数组对象，用于存放多维数据，并针对数组提供大量的数学运算函数
### 0.2 ndarray 与 NumPy 的关系
ndarray 是 NumPy 中的多维数组，数组中的元素具有相同的类型，且可以被索引。
# ndarray 多维数组
## 1 ndarray 对象的结构
原始数组数据（raw array data），一个连续的 memory block；描述这些原始数据的元数据（metadata），数据的解释方式。
## 2 ndarray 对象的视图与副本
视图是原数据的一部分，对视图数据的修改会直接反映到原数据中；对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置
## 3 ndarray 对象的属性
属性 | 说明
- | :-
 T | 返回数组的转置 
 dtype ndarray | 对象的元素类型 
 ndim | 秩，即轴的数量或维度的数量
 shape | 数组的维度，对于矩阵，n 行 m 列，shape 为（m, n）
 size | 数组元素的总个数，相当于 .shape 中 n * m 的值
 itemsize ndarray | 对象中每个元素的大小，以字节为单位,，如 int 32,为4字节
 real/imag ndarray | 元素的实/虚部
 ndarray.flat | 返回一个数组迭代器，对 flat 赋值将会覆盖数组元素的值
 ndarray.nbytes | 返回数组的所有元素占用的存储空间

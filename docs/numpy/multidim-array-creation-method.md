
# ndarray 多维数创建方法
## 1 使用 array() 函数由列表或元组创建数组
numpy.array(object, dtype = None)
- 例：` x = np.array([[1, 2], [3, 4]])` 
## 2 使用 arange()、linspace() 等推导函数创建数组
` numpy.arange([start, ]stop[, step], dtype=None )` 

` linspace(start, stop, num=50, endpoint=True,dtype=None)`

logspace()函数创建等比数列
## 3 使用 ones()、zeros()、empty() 函数创建特殊数组
` numpy.ones(shape, dtype=None)  # 创建元素全为1的数组` 
- 例：` a = np.ones(shape=(3, 3), dtype=int)` 

` ones_like(arr)   # 数创建维数与已有数组 shape 相同，元素值全为1的数组` 
## 4 其他创建 ndarray 数组的方法
` eye(N, M=None, k=0)  # 创建对角线值为1，其余值为0的二维数组，k 为主对角线偏移量，M 为列数，可选。` 

` identity(n, dtype=None)  # 创建单位矩阵,n * n 的单位矩阵` 

` full(shape, fill_value, dtype=None)  # 创建固定值填充的数组` 
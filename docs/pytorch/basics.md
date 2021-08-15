# 基础知识

## 1 载入 PyTorch

为了使用 PyTorch，首先在 Python 代码的最上方加入以下代码来载入 PyTorch（注意是 `torch` 而非 `pytorch`）。

```Python
import torch
```

## 2 数据的表示

在 PyTorch 中，所有数据都使用**张量**（tensor）类型来储存，可以将其理解为 C/C++ 里的多维数组。PyTorch 中的张量可以是数学中的标量、向量、矩阵或任意高维张量。

tensor对象有三个属性：  
**rank**：即张量的维数  
**shape**：即张量的行数和列数  
**type**：即张量元素的数据类型

### 2.1 创建与初始化

#### 2.1.1 使用函数 `torch.tensor()` 或 `torch.arange()`，直接从数据创建 tensor。

```Python
# 标量
a0 = torch.tensor(0)
# 向量
a = torch.arange(12) # 对应 Python 中的 range(12)
a1 = torch.tensor([0, 1])
# 矩阵
a2 = torch.tensor([[0, 1, 2], [3, 4, 5]])
# 3维张量
a3 = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 	11]], [[12, 13], [14, 15]]])
# 把变量作为参数
data = [[1,2],[3,4]]
a_data = torch.tensor(data)
```

#### 2.1.2 从 NumPy 数组创建 tensor

```Python
import numpy as np
np_array = np.array(data)
a_np = torch.from_numpy(np_array)
```

#### 2.1.3 从另一个 tensor 创建 tensor

使用 `torch.ones_like(tensor[,dtype])` 来创建全 1 tensor，或使用 `torch.rand_like(tensor[,dtype])` 来创建随机数 tensor
	
```Python
# 保留 a_data 的性质
a_ones = torch.ones_like(a_data)
# 改变 a_data 的性质
a_rand = torch.rand_like(a_data, dtype=torch.float)
print(a_ones)
print(a_rand)
```

输出：
	
```
tensor([[1, 1],
		[1, 1]])
tensor([[0.2095, 0.9481],
		[0.2369, 0.0424]])
```

#### 2.1.4 用随机数或常量创建 tensor

用一个元组（tuple） `shape` 来表示 tensor 的维度。
	
```Python
shape = (2,3)
b_rand = torch.rand(shape)
b_ones = torch.ones(shape)
b_zeros = torch.zeros(shape)
print(b_rand)
print(b_ones)
print(b_zeros)
```
输出：
	
```
tensor([[0.8283, 0.5521, 0.4598],
		[0.3117, 0.7638, 0.0300]])
tensor([[1., 1., 1.],
		[1., 1., 1.]])
tensor([[0., 0., 0.],
		[0., 0., 0.]])
```

### 2.2 维度和尺寸

#### 2.2.1 维度大小与数目的获取

##### 2.2.1.1 `tensor.ndimension()`

用于获取 tensor 的维数（整数）

```Python
a0.ndimension() # 返回：0
a1.ndimension() # 返回：1
a2.ndimension() # 返回：2
```

##### 2.2.1.2 `tensor.nelement()`

用于获取张量总元素个数

```Python
a0.nelement() # 返回：1
a1.nelement() # 返回：2
a2.nelement() # 返回：6
```

##### 2.2.1.3 `tensor.size([dim])` 或 `tensor.shape`

用于获取张量每个维度的大小，返回结果类型为 `torch.Size`。`tensor.size()` 调用的是函数，想要获得特定维度的大小可以加入维度参数，而 `tensor.shape` 访问的是张量的属性。

```Python
a3.size()   # 返回：torch.Size([4, 2, 2])
a3.size(0)  # 返回：4
a3.size(1)  # 返回：2
a3.size(2)  # 返回：2
a3.size(-1) # 返回：2 （-1 表示倒数第一个）
a3.shape    # 返回：torch.Size([4, 2, 2])
```

\*标量没有尺寸

```Python
a0.size() # 返回：torch.Size([])
a0.shape  # 返回：torch.Size([])
```

#### 2.2.2 维度变换

##### 2.2.2.1 `tensor.view()`

将 tensor 中的数据按照行优先的顺序排成一个一维数组，之后按照参数组合成其它维度的 tensor。参数有两个时，可省略其中一个，用 -1 表示。

```Python
b = a.view(3, 4)  # 尺寸变为 [3, 4]
b = a.view(-1, 4) # 尺寸变为 [3, 4]
```

##### 2.2.2.2 `tensor.unsqueeze(dim)` 或 `tensor.squeeze([dim])`

`tensor.unsqueeze(dim)` 是给指定位置加上维数为 1 的维度；`tensor.squeeze()` 是去掉所有维数为 1 的维度；`tensor.squeeze(dim)` 是去掉指定的维数为 1 的维度。

```Python
b = a.unsqueeze(dim = 0) # 尺寸变为 [1, 12]
c = b.squeeze(dim = 0)   # 尺寸变为 [12]
```

### 2.3 数据类型

和一般的 Python 变量不同，PyTorch 中的数据是有类型的，其中常用的类型有布尔型（`torch.bool`）、整数（`torch.long`）和浮点数（`torch.float`）。

**注意：**整数和浮点数的默认精度在不同计算机上可能是不同的。在常见的 64 位计算机上，`torch.long` 一般是 `torch.int64`，`torch.float` 一般是 `torch.float32`。

#### 2.3.1 查看数据类型

与 C/C++ 的数组类似，同一张量中的所有值只能是同一类型。查看张量 `a` 的数据类型可以使用 `a.dtype` 属性。

```Python
a = torch.tensor([1, 2])
a.dtype # 返回torch.int64，相当于torch.long
```

#### 2.3.2 指定数据类型

如果想在创建张量时就指定数据类型，有以下两种方法：

方法 1：在 `torch.tensor` 中指定 `dtype` 参数，若类型不符合则会自动进行类型转换。

```Python
a = torch.tensor([True, True, False], dtype = torch.float)  # 结果：tensor([1., 1., 0.])
```

方法 2：直接调用对应数据类型张量的构造函数（注意区分大小写），若类型不符合也会自动进行类型转换。

```Python
b = torch.BoolTensor([1, 1, 0])  # 结果：tensor([True, True, False])
c = torch.LongTensor([1, 1, 0])  # 结果：tensor([1, 1, 0])
d = torch.FloatTensor([1, 1, 0]) # 结果：tensor([1., 1., 0.])
```

#### 2.3.3 数据类型转换

如果想将转换张量的数据类型，有以下两种方法：

方法 1：使用 `.to` 函数，如 `a.to(torch.float)`。

方法 2：使用 `.bool()`、`.long()`、`.float()` 等函数，如 `b.long()`、`c.bool()` 等。

## 3 张量的运算

对于维度相同的张量，直接采用批量化计算；对于维度不同的张量，遵循 broadingcasting 规则。

### 3.1 批量化计算

维度相同的张量可以直接相加。

```Python
c = a + b # 相当于：c[i] = a[i] + b[i]
```

### 3.2 Broadcasting 规则

算数运算中，从后向前对两个张量的维度进行遍历。如果至少一个张量的维度遍历结束时，两张量维度的值相同或者其中一个值为 1，则满足进行算术运算的要求，否则不满足。

若满足进行算术运算的要求，则在检查过程中，将相应维度取最大值，其中一个张量遍历结束时，复制另一个张量的剩余维度。

例1：现有三个张量 **a**，**b**，**c**，其维度分别为 [2,3,5]，[2,5]，[1,5]。若运行 `a + b` ，二者最后一维值均为 5，而倒数第二维值分别为 3 和 2，不满足要求。若运行 `a + c`，二者最后一维值均为5，倒数第二维有一值为 1，**c** 遍历结束，满足要求，运算结果的维度为 [2,3,5]。

例2：有一个长为 *m* 的向量 **a** 和一个长为 *n* 的向量 **b**，维度分别为 [m]，[n]，要创建一个 *m \* n* 的矩阵 **C** 满足第 *i* 行第 *j* 列的数是 *a<sub>i</sub> \+ b<sub>j</sub>*，则可以先将 **a** 变形成一个 *m \* 1* 的矩阵，维度变为 [m,1]，然后利用 broadcasting 规则来计算 **C**，最终得到 **C** 的维度即为 [m,n]。

```Python
C = a.unsqueeze(dim = -1) + b
```

### 3.3 导出运算结果

`tensor.item()` 用于返回单元素张量的元素值；`tensor.tolist()` 用于将张量作为（嵌套）列表返回；`tensor.cpu()` 用于将数据处理设备从其它设备拿到 CPU 上；`tensor.detach()` 返回一个新的tensor，仍指向原变量的存放位置，requirse_grad 变为false，得到的 tensor 不需要计算器梯度，不具有 grad；`tensor.numpy()` 用于将张量（不限维数）转换为 ndarray 变量，转换后 dtype 与 tensor 一致。

```Python
a.item()   # 只能用于标量，返回值类型为Python标量
b.tolist() # 不能用于标量，返回值类型为list
c.cpu().detach().numpy() # 通用，返回值类型为NumPy
```

## 4 PyTorch 常用模块

载入 PyTorch

```Python
import torch
```

`torch.nn.functional` 包含 convolution 函数、pooling 函数、非线性激活函数、normalization 函数、线性函数、距离函数（distance functions）、损失函数（loss functions）、vision functions 等，用以下代码载入。

```Python
import torch.nn.functional as F
```

参见 [WHAT IS *TORCH.NN* REALLY?](https://pytorch.org/tutorials/beginner/nn_tutorial.html)

```Python
from torch import nn
```

参见 [torch.optim](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/)

```Python
from torch import optim
```


## 5\* 使用 GPU 加速运算

如果你的计算机配有可用于科学计算的 NVIDIA GPU，且已经安装了 GPU 版 PyTorch 及相应版本的 NVIDIA CUDA 驱动，则可以使用 GPU 来加速 PyTorch 中的运算，在大规模运算时一般会有 5~10 倍的加速。可以使用函数 `torch.cuda.is_available()` 来确认计算机是否支持 GPU 加速，如果支持则会返回 `True`。

**提示：**这里只介绍单个 GPU 的情况。多个 GPU 的情况比较复杂，这里不再进一步介绍。

要使用 GPU，首先需指定运算设备。为了让代码在支持和不支持 GPU 的计算机上都能正常运行，可以使用以下代码。

```Python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

为了让 GPU 加速运算，需要在运算前使用 `.to(device)` 将待运算的数据从内存转移到 GPU 的显存；运算后的结果仍然会保存在显存里。已经转移过的数据和运算结果不必重新转移。

```Python
b = b.to(device)
c = a.to(device) + b     # c的结果已存至显存里
d = a.to(device) + b * c # 不必再转移b或c
```

**注意：**同一运算中涉及到的所有变量必须在同一设备里，假如 `a` 在内存里而 `b` 在显存里，则 `a + b` 会报错。

**提示：**尽管 PyTorch 中绝大多数运算都支持 GPU，但仍有少量不常用的运算目前还不支持 GPU。如果在使用的过程中报错，请用 `.cpu()` 函数将数据转移回内存之后再进行运算。
# 基础知识

## 载入 PyTorch

为了使用 PyTorch，首先在 Python 代码的最上方加入以下代码来载入 PyTorch（注意是 `torch` 而非 `pytorch`）。

```py
import torch
```

## 数据的表示

在 PyTorch 中，所有数据都使用**张量**（tensor）类型来储存，可以将其理解为 C/C++ 里的多维数组。创建张量可以使用函数 `torch.tensor`。PyTorch 中的张量可以是数学中的标量、向量、矩阵或任意高维张量。

```py
# 标量
a0 = torch.tensor(0)
# 向量
a1 = torch.tensor([0, 1])
# 矩阵
a2 = torch.tensor([[0, 1, 2], [3, 4, 5]])
# 3维张量
a3 = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]], [[12, 13], [14, 15]]])
```

### 维度和尺寸

@TODO

```py
a = torch.arange(12) # 对应Python里的range(12)
a.size()         # 返回：torch.Size([12])
a.size(dim = -1) # 返回：12
```

@TODO

```py
b = a.view(3, 4)  # 尺寸变为 [3, 4]
b = a.view(-1, 4) # 尺寸变为 [3, 4]
```

@TODO

```py
b = a.unsqueeze(dim = 0) # 尺寸变为 [1, 12]
c = b.squeeze(dim = 0)   # 尺寸变为 [12]
```

@TODO

标量没有尺寸。

```py
torch.tensor(0).size() # 返回：torch.Size([])
```

@TODO

### 数据类型

和一般的 Python 变量不同，PyTorch 中的数据是有类型的，其中常用的类型有布尔型（`torch.bool`）、整数（`torch.long`）和浮点数（`torch.float`）。

**注意：**整数和浮点数的默认精度在不同计算机上可能是不同的。在常见的 64 位计算机上，`torch.long` 一般是 `torch.int64`，`torch.float` 一般是 `torch.float32`。

#### 查看数据类型

与 C/C++ 的数组类似，同一张量中的所有值只能是同一类型。查看张量 `a` 的数据类型可以使用 `a.dtype` 属性。

```py
a = torch.tensor([1, 2])
a.dtype # 返回torch.int64，相当于torch.long
```

#### 指定数据类型

如果想在创建张量时就指定数据类型，有以下两种方法：

方法 1：在 `torch.tensor` 中指定 `dtype` 参数，若类型不符合则会自动进行类型转换。

```py
a = torch.tensor([True, True, False], dtype = torch.float) # 结果：tensor([1., 1., 0.])
```

方法 2：直接调用对应数据类型张量的构造函数（注意区分大小写），若类型不符合也会自动进行类型转换。

```py
b = torch.BoolTensor([1, 1, 0])  # 结果：tensor([True, True, False])
c = torch.LongTensor([1, 1, 0])  # 结果：tensor([1, 1, 0])
d = torch.FloatTensor([1, 1, 0]) # 结果：tensor([1., 1., 0.])
```

#### 数据类型转换

如果想将转换张量的数据类型，有以下两种方法：

方法 1：使用 `.to` 函数，如 `a.to(torch.float)`。

方法 2：使用 `.bool()`、`.long()`、`.float()` 等函数，如 `b.long()`、`c.bool()` 等。

## 张量的运算

@TODO

#### 批量化计算

@TODO

```py
c = a + b # 相当于：c[i] = a[i] + b[i]
```

#### Broadcasting 规则

@TODO

例如，有一个长为 $m$ 的向量 $\boldsymbol a$ 和一个长为 $n$ 的向量 $\boldsymbol b$，要创建一个 $m\times n$ 的矩阵 $\boldsymbol C$ 满足第 $i$ 行第 $j$ 列的数是 $a_i+b_j$，则可以先将 $\boldsymbol a$ 变形成一个 $m\times 1$ 的矩阵，然后利用 broadcasting 规则来计算 $\boldsymbol C$。

```py
C = a.unsqueeze(dim = -1) + b
```

#### 导出运算结果

@TODO

```py
a.item()   # 只能用于标量，返回值类型为Python标量
b.tolist() # 不能用于标量，返回值类型为list
c.cpu().detach().numpy() # 通用，返回值类型为NumPy
```

@TODO

## PyTorch 常用模块

@TODO

```py
import torch
```

@TODO

```py
import torch.nn.functional as F
```

@TODO

```py
from torch import nn
```

@TODO

```py
from torch import optim
```

@TODO

## \*使用 GPU 加速运算

如果你的计算机配有可用于科学计算的 NVIDIA GPU，且已经安装了 GPU 版 PyTorch 及相应版本的 NVIDIA CUDA 驱动，则可以使用 GPU 来加速 PyTorch 中的运算，在大规模运算时一般会有 5~10 倍的加速。可以使用函数 `torch.cuda.is_available()` 来确认计算机是否支持 GPU 加速，如果支持则会返回 `True`。

**提示：**这里只介绍单个 GPU 的情况。多个 GPU 的情况比较复杂，这里不再进一步介绍。

要使用 GPU，首先需指定运算设备。为了让代码在支持和不支持 GPU 的计算机上都能正常运行，可以使用以下代码。

```py
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

为了让 GPU 加速运算，需要在运算前使用 `.to(device)` 将待运算的数据从内存转移到 GPU 的显存；运算后的结果仍然会保存在显存里。已经转移过的数据和运算结果不必重新转移。

```py
b = b.to(device)
c = a.to(device) + b     # c的结果已存至显存里
d = a.to(device) + b * c # 不必再转移b或c
```

**注意：**同一运算中涉及到的所有变量必须在同一设备里，假如 `a` 在内存里而 `b` 在显存里，则 `a + b` 会报错。

**提示：**尽管 PyTorch 中绝大多数运算都支持 GPU，但仍有少量不常用的运算目前还不支持 GPU。如果在使用的过程中报错，请用 `.cpu()` 函数将数据转移回内存之后再进行运算。
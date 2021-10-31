# 基础知识

## 1 张量

TensorFlow 使用 **张量** （Tensor）作为数据的基本单位。TensorFlow 的张量在概念上等同于多维数组，我们可以使用它来描述数学中的标量（0 维数组）、向量（1 维数组）、矩阵（2 维数组）等各种量，示例如下：

```python
# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())

# 定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2))

# 定义两个2×2的常量矩阵
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
```

可以通过张量的``shape`` 、 ``dtype``分别查看张量的形状和数据类型：

```python
# 查看矩阵A的形状、类型和值
print(A.shape)      # 输出(2, 2)，即矩阵的长和宽均为2
print(A.dtype)      # 输出<dtype: 'float32'>
print(A.numpy())    # 输出[[1. 2.]
                    #      [3. 4.]]
```

我们可以使用tensorflow中的**操作**对已有的张量进行运算：

```python
C = tf.add(A, B)    # 计算矩阵A和B的和
D = tf.matmul(A, B) # 计算矩阵A和B的乘积
```

## 2 自动求导

### 2.1一元函数求导

在机器学习中，如果需要计算函数的导数，tf提供了强大的自动求导机制。以下代码展示了如何使用 `tf.GradientTape()` 计算函数 $y=x^2$ 在 $x=6$时的导数：

```python
import tensorflow as tf

x = tf.Variable(initial_value=6.)
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)
```

输出：

```python
tf.Tensor(9.0, shape=(), dtype=float32)
tf.Tensor(6.0, shape=(), dtype=float32)
```

这里 `x` 是一个初始化为 3 的 **变量** （Variable），使用 `tf.Variable()` 声明。与普通张量一样，变量同样具有形状、类型和值三种属性。使用变量需要有一个初始化过程，可以通过在 `tf.Variable()` 中指定 `initial_value` 参数来指定初始值。这里将变量 `x` 初始化为 `3.` [1](https://tf.wiki/zh_hans/basic/basic.html#f0)。变量与普通张量的一个重要区别是其默认能够被 TensorFlow 的自动求导机制所求导，因此往往被用于定义机器学习模型的参数。

`tf.GradientTape()` 是一个自动求导的记录器。只要进入了 `with tf.GradientTape() as tape` 的上下文环境，则在该环境中计算步骤都会被自动记录。比如在上面的示例中，计算步骤 `y = tf.square(x)` 即被自动记录。离开上下文环境后，记录将停止，但记录器 `tape` 依然可用，因此可以通过 `y_grad = tape.gradient(y, x)` 求张量 `y` 对变量 `x` 的导数。

### 4 多元函数求偏导

计算函数$L(w,b)=||Xw+b-y||^2$在 $w=(1,2)^T,b=1$ 时分别对$w,b$的偏导数。其中 !$X = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix},  y = \begin{bmatrix} 1 \\ 2\end{bmatrix}$。

```python
X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])        # 计算L(w, b)关于w, b的偏导数
print(L, w_grad, b_grad)
```

输出：

```python
tf.Tensor(125.0, shape=(), dtype=float32)
tf.Tensor(
[[ 70.]
[100.]], shape=(2, 1), dtype=float32)
tf.Tensor(30.0, shape=(), dtype=float32)
```

## 3 线性回归

以下展示了如何使用 TensorFlow 计算线性回归。可以注意到，程序的结构和前述 NumPy 的实现非常类似。这里，TensorFlow 帮助我们做了两件重要的工作：

- 使用 `tape.gradient(ys, xs)` 自动计算梯度；
- 使用 `optimizer.apply_gradients(grads_and_vars)` 自动更新模型参数。

```python
X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y))
    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自动根据梯度更新参数
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
```

在这里，我们使用了前文的方式计算了损失函数关于参数的偏导数。同时，使用 `tf.keras.optimizers.SGD(learning_rate=5e-4)` 声明了一个梯度下降 **优化器** （Optimizer），其学习率为 5e-4。优化器可以帮助我们根据计算出的求导结果更新模型参数，从而最小化某个特定的损失函数，具体使用方式是调用其 `apply_gradients()` 方法。

注意到这里，更新模型参数的方法 `optimizer.apply_gradients()` 需要提供参数 `grads_and_vars`，即待更新的变量（如上述代码中的 `variables` ）及损失函数关于这些变量的偏导数（如上述代码中的 `grads` ）。具体而言，这里需要传入一个 Python 列表（List），列表中的每个元素是一个 `（变量的偏导数，变量）` 对。比如上例中需要传入的参数是 `[(grad_a, a), (grad_b, b)]` 。我们通过 `grads = tape.gradient(loss, variables)` 求出 tape 中记录的 `loss` 关于 `variables = [a, b]` 中每个变量的偏导数，也就是 `grads = [grad_a, grad_b]`，再使用 Python 的 `zip()` 函数将 `grads = [grad_a, grad_b]` 和 `variables = [a, b]` 拼装在一起，就可以组合出所需的参数了。

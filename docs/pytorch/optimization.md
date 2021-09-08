# 优化模型参数

有了模型和数据，需要通过优化数据上的参数来训练、验证和测试模型。训练模型是一个迭代的过程；在每次迭代（称为 epoch）中，模型对输出进行猜测，计算其猜测中的误差（损失），收集误差相对于其参数的导数，并使用梯度下降优化这些参数。要了解这个过程的更详细的步骤，请查看 [3Blue1Brown 的反向传播视频](https://www.youtube.com/watch?v=tIeHLnjs5U8)。

## 1 [代码准备](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#prerequisite-code)

加载前文[数据加载和处理]()及[向量数据建模：MLP - 2 构建神经网络](https://ie.readthedocs.io/zh_CN/latest/pytorch/mlp/#2)中的代码。

```Python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
	root="data",
	train=True,
	download=True,
	transform=ToTensor()
)

test_data = datasets.FashionMNIST(
	root="data",
	train=False,
	download=True,
	transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
	def __init__(self):
		super(NeuralNetwork, self).__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10),
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits

model = NeuralNetwork()
```

输出：

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## 2 [超参数](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#hyperparameters)

超参数是可调节的参数，用于控制模型优化过程。不同的超参数值会影响模型训练和收敛速度（更多内容详见 [HYPERPARAMETER TUNING WITH RAY TUNE](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)）

定义以下超参数：

* `learning_rate`：每个批处理 / 迭代需要更新多少模型参数。值越小，学习速度越慢；然而值过大可能导致训练过程中行为不可预测。

* `batch_size`：在参数更新之前通过网络传播的数据样本的数量。

* `epochs`：迭代数，在数据集上迭代的次数。

```Python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## 3 [优化循环](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimization-loop)

设置了超参数后可用一个优化循环来训练和优化模型。优化循环的每一次迭代称为 epoch。

每一次迭代包括两个主要部分：

* 训练循环：遍历训练集并尝试收敛到最优参数；

* 验证 / 测试循环：遍历测试集以检查模型性能是否改善。

下面简单介绍训练循环中的一些概念。详情见优化循环的[完整实现](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-impl-label)。

### 3.1 损失函数

未经训练的网络在面对一些训练数据时很可能不会给出正确的答案。损失函数度量的是所得到的结果与目标值的不相似程度，在训练时，我们希望将其最小化。为了计算损失，我们使用给定数据样本的输入进行预测，并将其与真实的数据标签值进行比较。

常见的损失函数包括： `nn.MSELoss`（均方误差），用于回归任务；`nn.NLLLoss`（负对数似然），用于分类；`nn.CrossEntropyLoss`，结合了 `nn.LogSoftmax` 和 `nn.NLLLoss`。

将模型输出的对数传递给 `nn.CrossEntropyLoss`，对数归一化并计算预测误差。

```Python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

### 3.2 优化器

优化：在每个训练步骤中调整模型参数以减少模型误差的过程。

**优化算法**定义如何执行优化过程（在本例中，我们使用随机梯度下降）。所有优化逻辑都封装在 `optimizer`（优化器）对象中。在这里，我们使用 SGD 优化器；此外，PyTorch 中还有许多不同的优化器，如 ADAM 和 RMSProp，它们可以更好地处理不同类型的模型和数据。

通过注册需要训练的模型参数来初始化优化器，并传入学习率超参数。

```Python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

在训练循环中，优化分为三个步骤:

* 调用 `optimizer.zero_grad()` 来重置模型参数的梯度。渐变默认累加；为了防止重复计数，在每次迭代时显式地将它们归零。

* 通过调用 `loss.backwards()` 来反向传播预测损失。PyTorch 存储每个参数的损耗的梯度。

* 得到梯度后，调用 `optimizer.step()` 来根据向后传递过程中收集的梯度调整参数。

## 4 [完整实现](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-implementation)

定义 `train_loop` 以循环优化代码，定义 `test_loop` 以根据测试数据评估模型性能：

```Python
def train_loop(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	for batch, (X, y) in enumerate(dataloader):
		# Compute prediction and loss
		pred = model(X)
		loss = loss_fn(pred, y)

		# Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct = 0, 0

	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			test_loss += loss_fn(pred, y).item()
			correct += (pred.argmax(1) == y).type(torch.float).sum().item()

	test_loss /= num_batches
	correct /= size
	print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

初始化损失函数和优化器，并将其传递给 `train_loop` 和 `test_loop`：

\*迭代次数可随意添加以跟踪模型的改进性能

```Python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
	print(f"Epoch {t+1}\n-------------------------------")
	train_loop(train_dataloader, model, loss_fn, optimizer)
	test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

输出：

```
Epoch 1
-------------------------------
loss: 2.304330  [    0/60000]
loss: 2.297135  [ 6400/60000]
loss: 2.276578  [12800/60000]
loss: 2.274323  [19200/60000]
loss: 2.251969  [25600/60000]
loss: 2.219396  [32000/60000]
loss: 2.235033  [38400/60000]
loss: 2.193696  [44800/60000]
loss: 2.203193  [51200/60000]
loss: 2.164317  [57600/60000]
Test Error:
 Accuracy: 42.6%, Avg loss: 2.161588

Epoch 2
-------------------------------
loss: 2.176873  [    0/60000]
loss: 2.164621  [ 6400/60000]
loss: 2.110821  [12800/60000]
loss: 2.126994  [19200/60000]
loss: 2.081725  [25600/60000]
loss: 2.020193  [32000/60000]
loss: 2.056971  [38400/60000]
loss: 1.978224  [44800/60000]
loss: 1.987843  [51200/60000]
loss: 1.911017  [57600/60000]
Test Error:
 Accuracy: 59.0%, Avg loss: 1.908803

Epoch 3
-------------------------------
loss: 1.944936  [    0/60000]
loss: 1.909998  [ 6400/60000]
loss: 1.798946  [12800/60000]
loss: 1.836452  [19200/60000]
loss: 1.739229  [25600/60000]
loss: 1.682956  [32000/60000]
loss: 1.712705  [38400/60000]
loss: 1.613176  [44800/60000]
loss: 1.628942  [51200/60000]
loss: 1.520131  [57600/60000]
Test Error:
 Accuracy: 63.1%, Avg loss: 1.536381

Epoch 4
-------------------------------
loss: 1.603712  [    0/60000]
loss: 1.563944  [ 6400/60000]
loss: 1.415494  [12800/60000]
loss: 1.483423  [19200/60000]
loss: 1.369704  [25600/60000]
loss: 1.362556  [32000/60000]
loss: 1.384496  [38400/60000]
loss: 1.309461  [44800/60000]
loss: 1.331194  [51200/60000]
loss: 1.231564  [57600/60000]
Test Error:
 Accuracy: 64.4%, Avg loss: 1.255288

Epoch 5
-------------------------------
loss: 1.333331  [    0/60000]
loss: 1.313506  [ 6400/60000]
loss: 1.145696  [12800/60000]
loss: 1.250660  [19200/60000]
loss: 1.126949  [25600/60000]
loss: 1.153840  [32000/60000]
loss: 1.182107  [38400/60000]
loss: 1.121818  [44800/60000]
loss: 1.145909  [51200/60000]
loss: 1.065338  [57600/60000]
Test Error:
 Accuracy: 65.2%, Avg loss: 1.082845

Epoch 6
-------------------------------
loss: 1.154285  [    0/60000]
loss: 1.157415  [ 6400/60000]
loss: 0.970255  [12800/60000]
loss: 1.108754  [19200/60000]
loss: 0.981628  [25600/60000]
loss: 1.016651  [32000/60000]
loss: 1.058487  [38400/60000]
loss: 1.003933  [44800/60000]
loss: 1.028309  [51200/60000]
loss: 0.963359  [57600/60000]
Test Error:
 Accuracy: 66.3%, Avg loss: 0.973785

Epoch 7
-------------------------------
loss: 1.032044  [    0/60000]
loss: 1.058307  [ 6400/60000]
loss: 0.852228  [12800/60000]
loss: 1.017036  [19200/60000]
loss: 0.893169  [25600/60000]
loss: 0.921922  [32000/60000]
loss: 0.978590  [38400/60000]
loss: 0.928219  [44800/60000]
loss: 0.948833  [51200/60000]
loss: 0.895926  [57600/60000]
Test Error:
 Accuracy: 67.4%, Avg loss: 0.900662

Epoch 8
-------------------------------
loss: 0.943577  [    0/60000]
loss: 0.990556  [ 6400/60000]
loss: 0.768960  [12800/60000]
loss: 0.953487  [19200/60000]
loss: 0.835906  [25600/60000]
loss: 0.853781  [32000/60000]
loss: 0.922632  [38400/60000]
loss: 0.877674  [44800/60000]
loss: 0.892046  [51200/60000]
loss: 0.847884  [57600/60000]
Test Error:
 Accuracy: 68.8%, Avg loss: 0.848415

Epoch 9
-------------------------------
loss: 0.876188  [    0/60000]
loss: 0.940220  [ 6400/60000]
loss: 0.707270  [12800/60000]
loss: 0.906877  [19200/60000]
loss: 0.795750  [25600/60000]
loss: 0.803136  [32000/60000]
loss: 0.880152  [38400/60000]
loss: 0.842024  [44800/60000]
loss: 0.849560  [51200/60000]
loss: 0.811139  [57600/60000]
Test Error:
 Accuracy: 70.1%, Avg loss: 0.808887

Epoch 10
-------------------------------
loss: 0.822409  [    0/60000]
loss: 0.900021  [ 6400/60000]
loss: 0.659346  [12800/60000]
loss: 0.871149  [19200/60000]
loss: 0.765414  [25600/60000]
loss: 0.764470  [32000/60000]
loss: 0.845658  [38400/60000]
loss: 0.815226  [44800/60000]
loss: 0.816064  [51200/60000]
loss: 0.781372  [57600/60000]
Test Error:
 Accuracy: 71.3%, Avg loss: 0.777356

Done!
```

## 5 拓展阅读

* [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)

* [torch.optim](https://pytorch.org/docs/stable/optim.html)

* [Warmstart Training a Model](https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html)

___

转载自：  
[OPTIMIZING MODEL PARAMETERS](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
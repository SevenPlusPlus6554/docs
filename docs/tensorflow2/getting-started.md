# 安装 TensorFlow

## 1 安装与环境配置

### 1.1 pip安装tensorflow

在安装pip后，可以在使用win+r命令，输入cmd打开控制窗，在控制窗下使用如下命令安装tensor flow2:

```python
 pip install tensorflow
```

![start_01](../img/start_01.png)

等待一段时间即可自动安装完毕。

### 1.2 验证安装是否是否成功

在控制窗输入"python"进入python，输入

```python
import tensorflow as tf
```

![start_02](..\img\start_02.png)

若点击回车后正常运行未报错，则证明tensorflow安装成功。

#### *Windows系统下装载tensorflow可能出现的问题及处理方法：*     

如果你在 Windows 下安装了 TensorFlow 2.1 正式版，可能会在导入 TensorFlow 时出现 DLL 载入错误 。此时安装 Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019 即可正常使用。


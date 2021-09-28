# 发布更新

1. 编辑页面并保存后，回到 github，点击 `Fetch origin`，填写 `summary` 后点击 `commit` 提交更改

2. 点击 `Push Origin`，将更改 push 上去

3. 按住 `Windows + R`，打开 `cmd`，使用 `cd` 命令进入 clone 下来的文件所在文件夹

4. 执行命令 `mkdocs build` 构建版本

5. 回到 github，重复步骤 1 和 2

6. 进入 readthedocs 网站，`IE科协技术资料` 库，在 `概况` 一栏点击 `build`，将 github 上的最新版本构建到 readthedocs 网站上
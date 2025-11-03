# Nuitka 打包教程

Nuitka 是一个强大的 Python 编译器，可以将 Python 脚本打包为高效的可执行文件（如 .exe），适用于无 Python 环境的系统。以下是使用 Nuitka 打包的详细步骤。

## 1. 安装 Nuitka 和依赖环境

安装 Nuitka 在虚拟环境中运行以下命令安装 Nuitka：

``` shell
pip install -U nuitka
```

安装编译器 Windows 用户：推荐安装 MinGW64 或 Visual Studio (VS2017+)。 Linux/Mac 用户：确保已安装 GCC 或 Clang。

## 2. 打包命令示例

以下是常用的打包命令及其参数说明：

**基础打包**
将 Python 脚本打包为独立可执行文件：

``` shell
python -m nuitka --standalone --onefile --remove-output your_script.py
# --standalone：生成独立运行的文件夹或文件。
# --onefile：将所有内容打包为单个可执行文件。
# --remove-output：清理临时文件。
```

**带插件支持的打包**
如果脚本依赖特定库（如 PySide6、NumPy 等），需启用相关插件：

``` shell
python -m nuitka --standalone --onefile --enable-plugin=pyside6 --plugin-enable=numpy your_script.py
```

**自定义图标和其他信息**
为生成的可执行文件添加图标和版本信息：

``` shell
python -m nuitka --standalone --onefile \
--windows-icon-from-ico=your_icon.ico \
--windows-product-name="YourApp" \
--windows-file-version=1.0.0 \
your_script.py
```

``` text
--mingw64 #默认为已经安装的vs2017去编译，否则就按指定的比如mingw(官方建议)
--standalone 独立环境，这是必须的(否则拷给别人无法使用)
--windows-disable-console 没有CMD控制窗口
--output-dir=out 生成exe到out文件夹下面去
--show-progress 显示编译的进度，很直观
--show-memory 显示内存的占用
--enable-plugin=pyside6
--plugin-enable=tk-inter 打包tkinter模块的刚需
--plugin-enable=numpy 打包numpy,pandas,matplotlib模块的刚需
--plugin-enable=torch 打包pytorch的刚需
--plugin-enable=tensorflow 打包tensorflow的刚需
--windows-icon-from-ico=你的.ico 软件的图标
--windows-company-name=Windows下软件公司信息
--windows-product-name=Windows下软件名称
--windows-file-version=Windows下软件的信息
--windows-product-version=Windows下软件的产品信息
--windows-file-description=Windows下软件的作用描述
--windows-uac-admin=Windows下用户可以使用管理员权限来安装
--linux-onefile-icon=Linux下的图标位置
--onefile 像pyinstaller一样打包成单个exe文件(2021年我会再出教程来解释)
--include-package=复制比如numpy,PyQt5 这些带文件夹的叫包或者轮子
--include-module=复制比如when.py 这些以.py结尾的叫模块
```

```shell
python -m nuitka --standalone --onefile --show-memory  --follow-imports --show-progress --follow-import-to=src --plugin-enable=numpy,matplotlib --output-dir=out  --static-libpython=no --windows-product-name="EvaluationStabCS" --windows-file-version=1.0.0 --windows-icon-from-ico=./logo.ico main.py

```

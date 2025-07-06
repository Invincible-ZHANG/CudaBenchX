# CudaBenchX
A simple and practical CUDA parallel computing benchmark and performance analysis tool

##  项目目标：
构建一个 C++/CUDA 项目，对比 CPU 与 GPU 在以下任务中的性能差异：
| 子任务     | 内容                           |
| ------- | ---------------------------- |
| 矩阵乘法  | 经典 GEMM，支持不同 block/grid 大小配置 |
| 向量加法  | 多线程并行处理（`A + B → C`）         |
| 归约操作  | 并行求和、最大值                     |
| 直方图统计 | 数据分桶统计，模拟图像处理场景              |
| 时间记录  | 每个核函数执行时间、加速比对比              |

## 初始化项目结构
```
CudaBenchX/
├── benchmark/                   # 📊 保存 Benchmark 运行结果（如 CSV），可选上传
│   └── results.csv              # ← 可加入 .gitignore（如不打算分享数据）
├── include/                     # 📁 放置公共头文件（如 timer.h）
│   └── timer.h                  # ⏱️ GPU/CPU 计时代码（可选）
├── scripts/                     # ⚙️ 存放构建、运行脚本
│   ├── build.sh                 # 🐧 Linux/Git Bash 下的构建脚本
│   └── build.bat                # 🪟 Windows 下的构建脚本（你正在使用的）
├── src/                         # 🔧 所有源代码文件（.cu / .cpp）
│   ├── main.cpp                 # 🚪 可选主函数入口（用于封装多个模块）
│   ├── vector_add.cu            # ➕ GPU 向量加法实现
│   └── gemm_cuda.cu             # 🧮 CUDA 矩阵乘法实现（如你后续添加）
├── .gitignore                   # 🚫 忽略构建文件/IDE缓存/临时文件等
├── README.md                    # 📘 项目介绍文档（必备！）
├── LICENSE                      # ⚖️ 开源协议（如 MIT、Apache 2.0）
└── CMakeLists.txt               # 🧱 CMake 构建系统配置（如你启用 CMake）

```

## 更新日式（ChangLog）
 ###2025-07-06
 - 向量加法（vector add）
 - 项目结构
  -  LICENSE

学习：
   - 一个工作正常的 CUDA 项目骨架 
 - 成功编译 + 运行 .cu 文件
 - .bat 脚本一键构建 
 - GitHub 仓库同步 



## 🔧 Build
✅ 方法一：使用 Visual Studio 提供的编译环境（最推荐）
NVIDIA 官方推荐方式，就是用 Visual Studio 的开发者命令行 来运行 nvcc。
（x64 Native Tools Command Prompt for VS 2022）

```cmd
cd /d "E:\open source\CudaBenchX"
scripts\build.bat
```
说明：
 - /d 是允许切换驱动器（从 C盘到 E盘）
 - 用引号包住路径，因为有空格

 ## Run
 ```cmd
 vector_add

```


## 学习笔记
（随手记，之后会进行归纳整理）

**1.  .cu 文件是什么？**
.cu 是 CUDA C/C++ 源代码文件的扩展名。

它表示这个源代码文件中包含了可以运行在 NVIDIA GPU 上的代码（kernel），而不是只在 CPU 上运行的普通 C/C++ 代码。
| 内容                 | 举例                                                      |
| ------------------ | ------------------------------------------------------- |
| 💻 包含普通 C/C++ 代码   | 你可以像写 `.cpp` 那样写 main 函数、循环、函数调用等                       |
| ⚡ 包含 GPU Kernel 代码 | `__global__ void kernel(...)` 就是运行在 GPU 上的函数            |
| 🚀 用 `nvcc` 编译     | NVIDIA 的 CUDA 编译器可以识别 `.cu`，将其编译为支持 GPU 的可执行程序          |
| 🧠 支持 CUDA 的特殊关键字  | 如 `__global__`、`__device__`、`__shared__`、`cudaMalloc` 等 |





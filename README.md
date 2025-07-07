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


## GPU 并行编程
### 1.CUDA 基础结构：Host + Device 分工
| 区域          | 含义          | 示例代码                             |
| ----------- | ----------- | -------------------------------- |
| Host（CPU）   | 控制程序流程、管理内存 | `main()`、`new`、`cudaMemcpy`      |
| Device（GPU） | 高并行执行计算任务   | `__global__ void vectorAdd(...)` |

CUDA 的编程模型本质是：主机控制 + 设备计算

### 2. 核函数（__global__）与并行执行
```
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
```
 - __global__：表示这是一个 设备端核函数，由主机调用，在 GPU 上执行
 - 每个线程执行一段相同的代码（SIMT 模型）
 - blockIdx, threadIdx：系统自动为你分配的线程编号，用来确定当前线程处理哪个元素
```
int i = blockIdx.x * blockDim.x + threadIdx.x;
```
当前线程在线性一维数组中的全局索引（编号）：每个线程算出“自己”该处理第几个数据元素。
| 元素            | 含义                  |
| ------------- | ------------------- |
| `blockIdx.x`  | 当前线程块的编号（第几个 Block） |
| `blockDim.x`  | 每个线程块有多少个线程         |
| `threadIdx.x` | 当前线程在线程块内的编号        |

```
vectorAdd<<<4, 256>>>(...);  // 4 个 block，每个 block 有 256 个线程
```
blockIdx.x = 0~3（共有 4 个 block）
threadIdx.x = 0~255（每个 block 有 256 个线程）


### 3. SIMT 模型 是什么？
Single Instruction, Multiple Threads 
它是 CUDA GPU 的核心执行方式，也可以理解为：
“每个线程执行一样的代码（指令），但处理不同的数据”
** 和传统模型比较： **
| 模型   | 全称                                   | 描述                          | 例子               |
| ---- | ------------------------------------ | --------------------------- | ---------------- |
| SIMD | Single Instruction, Multiple Data    | 一条指令作用于多个数据位（如向量寄存器）        | AVX 指令集、ARM Neon |
| SIMT | Single Instruction, Multiple Threads | 每个线程单独拥有程序计数器，执行同样的代码但可独立分支 | CUDA 中每个线程       |

** SIMT = 线程级别的 SIMD 模型增强版。**


###  4. 网格配置（Grid + Block）






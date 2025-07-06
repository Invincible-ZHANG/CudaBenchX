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


## 🔧 Build

```bash
./scripts/build.sh
```

 ## Run
 ```bash
 ./scripts/run_all.sh
```

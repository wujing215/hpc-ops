<div align="center">
  <img src="./assets/hpc-ops-logo.png" width="780" alt="HPC-Ops" />
  <h2 align="center">
    HPC-Ops
  </h2>
  <p>
    <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.8%2B-76B900">
    <img alt="GPU" src="https://img.shields.io/badge/GPU-H20%20%7C%20SM90-76B900">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-3776AB">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
  </p>
</div>

HPC-Ops 是由腾讯混元AI基础设施团队开发的**生产级、高性能、易用**的LLM推理算子库。

它专注于主导实际服务延迟和吞吐量的热点路径：注意力机制、MoE、GEMM、采样、归一化以及通信计算融合。这些内核专为现代NVIDIA GPU设计，并与广泛使用的推理框架和内核库（如vLLM、SGLang、FlashInfer、NCCL、cuBLAS和TensorRT-LLM）进行基准测试。

<p align="center">
  <a href="#为什么选择-hpc-ops">为什么选择 HPC-Ops?</a> |
  <a href="#更新日志">更新日志</a> |
  <a href="#性能表现">性能表现</a> |
  <a href="#算子目录">算子目录</a> |
  <a href="#快速开始">快速开始</a> |
  <a href="#路线图">路线图</a>
</p>

## 为什么选择 HPC-Ops?

- **SOTA性能与生产验证**: 深度优化的内核专为NVIDIA H20 GPU定制，在注意力机制、GEMM、MoE、采样和融合通信计算算子方面提供SOTA性能。已在腾讯大规模生产推理中验证。
- **易于集成**: 简洁的Python API设计，可无缝集成到流行的推理框架如vLLM和SGLang中，测试和基准测试使验证变得简单直接。
- **丰富的精度支持**: 原生支持多种数据类型，包括BF16和具有不同量化方案的FP8，以及用于精度敏感推理场景的混合精度内核。
- **现代CUDA教程**: 通过紧凑的生产算子实现，提供使用CUDA、CuTe、CUTLASS、cp.async、TMA、PDL和multicast构建SOTA内核的实践示例。

## 更新日志

### 2026年6月

<details>
<summary><strong>动态解码注意力</strong>: 针对混合长度解码工作负载的运行时任务调度。</summary>

在线解码注意力具有高度动态的工作负载：请求长度在不同解码步骤中可能差异很大，同一批次内的请求可能具有非常不同的KV缓存长度。静态split-k调度无法适应这种形状分布，因此长请求通常会产生CTA级别的尾延迟，而较短请求则使计算资源利用不足。

HPC-Ops引入了动态任务调度路径，将所有请求分割成统一的KV图块，在每个解码步骤之前分配图块，并使用贪心装箱策略在CTA之间平衡它们。注意力内核直接消费生成的任务映射，而组合内核合并split-k结果。这使得每个CTA的工作更加均匀，减少了长上下文和混合长度解码批次中的长尾延迟。

</details>

<details>
<summary><strong>稀疏注意力</strong>: 用于长上下文工作负载的FP8块稀疏预填充注意力。</summary>

当大部分KV缓存与当前查询无关时，长上下文预填充注意力受内存带宽限制。块稀疏模式让内核跳过整个KV图块，但在FP8精度下利用稀疏性需要仔细的图块量化和掩码感知调度，以避免精度损失和负载不平衡。

HPC-Ops提供了一个FP8块稀疏预填充注意力内核，它接受预计算的块掩码，完全跳过被掩码的KV图块，并使用每图块FP8缩放来保持稀疏模式中的数值质量。

</details>

<details>
<summary><strong>路由GEMM</strong>: 用于精度敏感稀疏计算的BF16 x FP32 GEMM。</summary>

某些推理GEMM使用BF16激活但需要FP32敏感权重，例如MoE路由器GEMM和稀疏或线性注意力中的状态压缩GEMM。由于现代Tensor Core不提供原生FP32 Tensor Core吞吐量，朴素的FP32 GEMM回退到CUDA核心，而直接使用BF16或TF32可能引入大的精度退化或额外转换开销。

HPC-Ops将每个FP32权重分解为高BF16分量和低BF16残差分量的固定比例为`1/256`，然后将结果计算为两个BF16 Tensor Core GEMM的融合线性组合。实现将两个GEMM保持在一个内核内，共享输入移动，将中间累加器保存在寄存器中，并一次性写入最终结果，以更高的吞吐量保持FP32级别的精度。

</details>

<details>
<summary><strong>融合MoE</strong>: 具有每张量和块级缩放的FP8低延迟MoE。</summary>

FusedMoE路径通过将路由、门控-上GEMM、激活量化、下GEMM和top-k加权归约融合到一个流水线执行路径中，瞄准低延迟LLM MoE推理。与先收集再GEMM的设计相比，门控-上GEMM通过路由索引直接读取原始令牌，移除了独立的收集阶段及其额外的内存流量。

路由/索引预处理通过共享内存计数和连续专家输出范围减少全局原子压力。cp.async GEMM路径在低延迟机制中移除Warp专业化，增加CTA驻留，并将延迟隐藏从CTA内软件流水线转移到跨CTA硬件调度。PDL然后将阶段链接以减少内核启动气泡。

</details>

<details>
<summary><strong>融合AllReduce + RMSNorm</strong>: 用于张量并行推理的通信、残差加和归一化融合。</summary>

张量并行推理经常将AllReduce、残差加和RMSNorm作为独立阶段运行，即使它们形成一个逻辑操作：归一化缩减后的激活加上残差。这在已经通信密集的路径周围创建了额外的内核启动和重复的HBM读写。

HPC-Ops将AllReduce、残差加和RMSNorm融合到NVLink原生内核中。高吞吐量模式使用CUDA multicast用于大令牌预填充状形状，而低延迟模式使用带有PDL重叠的Lamport P2P双内核设计用于小令牌解码形状。两种模式都使用两阶段通信调度来减少通信开销，同时将归一化保持在集体路径中融合。

</details>

<details>
<summary><strong>采样器</strong>: 用于重复惩罚、温度缩放、Top-K、Top-P、Softmax和随机采样的融合解码后处理。</summary>

解码步骤采样通常在词汇维度上链接许多小内核，包括重复惩罚、温度缩放、top-k、top-p、softmax、随机采样和惩罚掩码更新。在小批量服务中，这些碎片化的内核和重复的全局内存传递可能成为可见的延迟瓶颈。

HPC-Ops提供了一个融合采样器，将完整的采样管道折叠成两个CUDA内核，当适用时自动分派更轻的温度专用快速路径。它将惩罚掩码更新保持在GPU上，为小批量使用跨块的细粒度并行，使用本地堆/合并式选择优化小top-k情况，并将top-k归约与softmax统计结合以减少词汇读取和固定启动开销。

</details>

## 算子目录

| 领域 | 算子家族 | 主要场景 | 精度 | 入口点 |
|---|---|---|---|---|
| 注意力 | 预填充/解码注意力，分页KV缓存，动态解码调度 | 具有可变请求长度的在线LLM服务 | BF16 / FP8 | `hpc/attention.py` |
| 注意力 | 块稀疏预填充注意力 | 具有块级稀疏掩码和分页KV缓存的长上下文预填充 | FP8 | `hpc/attention.py`, `hpc/stem.py` |
| GEMM | BF16 x FP32 GEMM | MoE路由器GEMM，稀疏/线性注意力状态压缩GEMM | BF16 x FP32 | `hpc/gemm.py` |
| 分组GEMM | FP8分组GEMM | 专家并行和分组专家矩阵乘法 | FP8 | `hpc/group_gemm.py` |
| 融合MoE | 融合FP8 MoE流水线 | TP/EP形状下的低延迟MoE推理 | FP8 | `hpc/fuse_moe.py` |
| 通信 | 融合AllReduce + 残差 + RMSNorm | 张量并行后GEMM融合 | BF16 | `hpc/allreduce.py` |
| 采样 | 融合采样器 | 解码步骤令牌采样 | BF16 / FP32 | `hpc/sampler.py` |
| 归一化 / RoPE / 激活 | RMSNorm, RoPE + KV存储, 激活量化 | 常见LLM层工具 | BF16 / FP8 | `hpc/normalization.py`, `hpc/rope.py`, `hpc/act.py` |

## 性能表现

| 算子 | 基准测试范围 | 基线 | 代表性结果 |
|---|---|---|---|
| 注意力 BF16 | 预填充，解码 | FlashInfer, FA2, FA3, TensorRT-LLM | 最高1.33x预填充，2.22x解码 |
| 注意力 FP8 | 预填充，解码 | FlashInfer, FA3, TensorRT-LLM | 最高1.12x预填充，2.0x解码 |
| 稀疏注意力 FP8 | 预填充，不同稀疏率和序列长度 | MIT-BSA BF16, FlashPrefill-BSA BF16, HPC-Dense FP8, FA3-Dense FP8 | 最高3.16x |
| 动态解码注意力 | 可变KV缓存解码，长单批次情况，混合长度批次情况 | 静态split-k调度 | 最高2.88x |
| BF16 x FP32 GEMM | 路由器GEMM形状，状态压缩GEMM形状 | cuBLAS FP32, cuBLAS TF32 | 最高3.22x |
| 融合FP8 MoE | DeepSeek-V3, Hunyuan-V3, Qwen3-235B在TP=8 EP=1, TP=1 EP=8下 | vLLM CUTLASS, vLLM Triton, SGLang | 最高1.6x TP, 1.5x EP |
| 融合AllReduce + 残差 + RMSNorm | 单节点BF16张量并行形状，隐藏大小4096/5120/7168 | NCCL, FlashInfer | 最高1.76x |
| 融合采样器 | BF16 logits，词汇大小120832，批次大小1-512 | vLLM风格PyTorch, FlashInfer | 最高8.5x |
| 分组GEMM FP8 | 专家分组矩阵乘法，预填充，解码 | DeepGEMM | 最高1.1x预填充，1.88x解码 |

*性能因情况而异；参见`benchmark/`目录进行复现。*

## 快速开始

### 要求
- NVIDIA SM90架构GPU
- Python 3.8或更高版本
- 支持C++17的编译器
- CUDA工具包：CUDA 12.8或更高版本

*您可以通过安装requirements-dev.txt中列出的模块来设置环境。*

### 从源码安装

```bash
git clone https://github.com/Tencent/hpc-ops.git
cd hpc-ops

# 构建包
make wheel
python3 -m pip install dist/*.whl
```

### 基本用法

示例：GroupGEMM fp8内核用法
```python
import torch
import hpc

num_tokens = 1024
num_group, n, k = 8, 4096, 4096
x = torch.randn((num_tokens, k), dtype=torch.float, device="cuda").to(torch.float8_e4m3fn)
w = torch.randn((num_group, n, k), dtype=torch.float, device="cuda").to(torch.float8_e4m3fn)
scale = torch.full((num_group,), 1.0, dtype=torch.float, device="cuda")
num_tokens_per_group = torch.full((num_group,), 8, dtype=torch.int32, device="cuda")
cu_num_tokens_per_group = torch.cumsum(torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), num_tokens_per_group]), dim=0).to(torch.int32)

output = hpc.group_gemm_pertensor_fp8(
    x, w, num_tokens_per_group, cu_num_tokens_per_group, scale,
)
```

*关于其他算子的用法，请参考tests/目录中相应的测试文件。*

## 路线图

- **扩展量化支持**: 灵活的策略（包括4位/8位混合精度），量化注意力和GEMM的内核优化，平衡速度和精度。
- **巨型内核**: 将多个连续算子融合到单个内核中，以减少内核间启动开销和中间内存流量。
- **下一代硬件支持**: 将HPC-Ops扩展到更先进的GPU架构。
- **低精度通信内核**: 为分布式推理添加低精度通信内核，例如AllReduce和AllGather。

欢迎贡献！如果您发现了内核错误或构建了更快的算子，我们很乐意看到您的PR。

⭐ **给这个仓库点星**以关注我们的进展。
我们正在持续改进性能，使您的LLM推理更快更高效。
更多改进正在进行中。
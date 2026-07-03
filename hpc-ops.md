HPC-Ops 是一个典型的 **Python 前端 + CUDA C++ 后端** 的高性能算子库，采用 PyTorch 自定义算子机制（`torch.ops`）进行绑定。
[[hpc-ops-diagram.svg]]

都用到了 cuTe CUTLASS TMA
### 核心目录结构

|目录/文件|作用|
|---|---|
|[`hpc/`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fhpc%2F%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fhpc%2F%22%2C%22scheme%22%3A%22file%22%7D)|**Python 前端接口层**，每个 `.py` 文件对应一类算子，通过 `torch.ops.hpc.*` 调用底层 C++|
|[`hpc/__init__.py`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fhpc%2F__init__.py%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fhpc%2F__init__.py%22%2C%22scheme%22%3A%22file%22%7D)|自动发现所有模块并导出函数，加载编译好的 `_C.abi3.so`|
|[`src/`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2F%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2F%22%2C%22scheme%22%3A%22file%22%7D)|**CUDA/C++ 后端实现**，每个子目录对应一类算子|
|`src/*/entry.cc`|PyTorch 算子注册入口，定义 `torch::library` 绑定|
|`src/*/kernels.cuh`|核心 CUDA 内核模板实现|
|[`src/utils/`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Futils%2F%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Futils%2F%22%2C%22scheme%22%3A%22file%22%7D)|通用工具（TMA 描述符、向量化加载等）|
|[`3rd/cutlass/`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2F3rd%2Fcutlass%2F%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2F3rd%2Fcutlass%2F%22%2C%22scheme%22%3A%22file%22%7D)|CUTLASS/CuTe 头文件依赖|
|[`tests/`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Ftests%2F%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Ftests%2F%22%2C%22scheme%22%3A%22file%22%7D)|正确性测试，也是**最好的用法示例**|
|[`benchmark/`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fbenchmark%2F%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fbenchmark%2F%22%2C%22scheme%22%3A%22file%22%7D)|性能基准测试，对比 vLLM/SGLang/FlashInfer 等|

**算子分类**

|类别|文件|核心技术|
|---|---|---|
|注意力|`attention.py` + `stem.py`|Warp Specialization, 动态任务调度, 块稀疏|
|MoE|`fuse_moe.py`|融合路由+GEMM+激活, cp.async 流水线|
|GEMM|`gemm.py`, `group_gemm.py`|BF16xFP32 分解, 分组矩阵乘|
|通信融合|`allreduce.py`|NVLink multicast, Lamport P2P|
|采样|`sampler.py`|融合 top-k/top-p/softmax/采样|
|工具算子|`normalization.py`, `rope.py`, `act.py`|RMSNorm, RoPE, 激活量化|


#### Attn
`attention.py` 定义了所有用户接口，分为 **Prefill** 和 **Decode** 两大类：
##### Prefill 系列函数

|函数|用途|精度|KV Cache|
|---|---|---|---|
|`attention_prefill_bf16`|首次 prefill，Q/K/V 连续存放|bf16|无|
|`attention_with_kvcache_prefill_bf16`|带 paged KV cache 的 prefill|bf16|有|
|`attention_with_kvcache_prefill_fp8`|FP8 量化 prefill|fp8→bf16|有|
|`attention_with_kvcache_blocksparse_prefill_fp8`|块稀疏注意力 prefill|fp8→bf16|有+mask|

**关键参数模式**：

- `q`: `[total_seq, num_head_q, num_dim_qk]` — 所有 batch 的 Q 拼接在一起
- `kcache/vcache`: `[num_blocks, block_size, num_head_kv, num_dim]` — paged KV cache
- `block_ids`: `[num_batch, max_blocks]` — 页表，每个 batch 映射到哪些物理 block
- `cu_seqlens_q`: `[num_batch + 1]` — 每个 batch 的 Q 起始位置
- `quant_type`: 4 种 FP8 量化方案（见 `QuantType` 枚举）

##### Decode 系列函数

|函数|用途|特性|
|---|---|---|
|`attention_decode_bf16`|BF16 解码|SplitK|
|`attention_decode_fp8`|FP8 解码|SplitK + 动态任务调度 + MTP|

**Decode 独有参数**：

- `mtp`: Multi-Token Prediction 的 draft token 数（0/1/2/3）
- `new_kv_included`: `num_seq_kvcache` 是否已包含新写入的 KV
- `splitk`: 是否使用 SplitK 并行
- `task_map`: 预调度的任务映射（由辅助函数生成）


### 辅助函数（Decode 任务调度三件套）

这是一个**两阶段调度**设计：

`1. get_attention_decode_task_workspace()  → 分配 task_map 显存 2. assign_attention_decode_task()         → 填充任务调度信息 3. attention_decode_fp8(..., task_map=...) → 使用调度信息执行` 

`task_map` 的内存布局（每条任务 48 字节 = 12 个 int32）：

`[header: num_tile_per_cta, num_total_ctas, num_head_kv, max_num_batch] [CTA0 的任务列表...] [CTA1 的任务列表...] ... [num_chunks 表: 每个 (head_kv, batch) 的 chunk 数]` 

每条 `TaskScheduleInfo` 记录一个 CTA 要处理的任务片段：

`struct TaskScheduleInfo {     int ihead_kv;        // 哪个 KV head     int ibatch;          // 哪个 batch     int ichunk;          // 第几个 chunk     int iseq_start;      // KV 序列起始位置     int num_seqkv;       // 本 chunk 的 KV 长度     int num_seqkvcache;  // 总 KV cache 长度     int num_tile_kv;     // KV tile 数     int num_tile_full;   // 满 tile 数（用于 causal mask）     int is_casual_chunk; // 是否包含 causal 对角线 };` 

---

## 二、C++ Entry 层 — 参数校验与调度决策

[`entry.cc`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fentry.cc%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fentry.cc%22%2C%22scheme%22%3A%22file%22%7D) 做了三件核心事情：

### 1. 参数校验

所有 tensor 必须 CUDA、dtype 匹配、形状一致。典型约束：

- `num_dim_qk == 128 && num_dim_v == 128`（目前只支持 dim=128）
    
- `block_size == 32 || 64`（bf16 decode）/ `block_size == 64`（fp8 decode）
    
- `heads_per_group == 4 || 8`（GQA 分组比）
    
- `mtp ∈ {0, 1, 2, 3}`
    

### 2. SplitK 自适应决策

`// BF16 decode: if (num_batch <= 32) splitk = 16;   // 小batch多切分 else                 splitk = 4;    // 大batch少切分  // FP8 decode (有 task_map 时): splitk = kCtaPerSmMap[sm_major_version][num_seq_q-1] * get_sm_count(); // sm90: {4, 3, 3, 2}  → mtp=0时每SM 4个CTA，mtp=3时每SM 2个CTA // sm100: {1, 1, 1, 1}` 

### 3. TMA 描述符分配

每个 batch 分配 2~4 个 TMA 描述符（Q/K/V/Y），在 kernel 启动前由 [`update_batched_tma`](command:gongfeng.gongfeng-copilot.chat.open-symbol-in-file?%5B%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fprefill%2Fkernels.cuh%22%2C%22external%22%3A%22file%3A%2F%2F%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fprefill%2Fkernels.cuh%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fprefill%2Fkernels.cuh%22%2C%22scheme%22%3A%22file%22%7D%2C%22update_batched_tma%22%2C%5B%7B%22line%22%3A95%2C%22character%22%3A16%7D%2C%7B%22line%22%3A95%2C%22character%22%3A34%7D%5D%5D) kernel 填充实际地址。

---

## 三、Prefill Kernel 层 — 两种执行策略

[`prefill.cc`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fprefill%2Fprefill.cc%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fprefill%2Fprefill.cc%22%2C%22scheme%22%3A%22file%22%7D) 根据**工作量大小**选择策略：

`int max_total_blocks = ceil(max_seq_q / 64) * num_batch * num_head_q; if (max_total_blocks < get_sm_count() * 2) {     // 策略1: multi_stage — 工作量小，每个block处理多个tile     prefill::multi_stage_dim128_async(...) } else {     // 策略2: warp_spec — 工作量大，用Warp Specialization     prefill::warp_spec_dim128_async(...) }` 

### 策略对比

|特性|multi_stage|warp_spec|
|---|---|---|
|适用场景|tile 数 < 2×SM 数|tile 数 ≥ 2×SM 数|
|Grid 大小|按 tile 分配|`dim3 grid(get_sm_count())`|
|并行策略|每个 CTA 处理一个 tile，空闲时取下一个|每个 SM 持续消费 tile 队列|
|核心技术|多阶段流水线 + TMA|Warp Specialization + TMA + barrier|

### Config 模板（[config.h](vscode-webview://1eceli340dvkgt1p4b0aal52tfr3lpt6imqlgo62immts8c4jebm/data/home/ivyjwu/work/hpc-ops/src/attention/prefill/config.h)）

所有 kernel 由模板配置驱动，以 `AttentionPrefillConfig` 为例：

`Config = AttentionPrefillConfig<     Tin, Tout,                    // 数据类型     TiledMmaQK, TiledMmaPV,       // MMA 指令选择     kTileM=128, kTileN=64,        // Q tile 128行, KV tile 64行     kTileK=128, kTileV=128,       // head dim     kStage=2,                     // 流水线阶段数     kWarpgroupM=2, kWarpgroupN=1 // warpgroup 布局 >;` 

CuTe 的 `SLayout` 系列自动计算共享内存的 swizzle 布局，确保 TMA 加载和 GMMA 计算无 bank conflict。

### 核心 Kernel 流程（[`kernels.cuh`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fprefill%2Fkernels.cuh%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fprefill%2Fkernels.cuh%22%2C%22scheme%22%3A%22file%22%7D)）

`┌─────────────────────────────────────────────────┐ │ 1. update_batched_tma kernel                     │ │    每个 batch 一个 block，填充 TMA 描述符地址      │ ├─────────────────────────────────────────────────┤ │ 2. attention_prefill_warp_spec kernel            │ │    ┌─────────────────────────────────────────┐   │ │    │ a. load Q to shared memory (TMA)        │   │ │    │ b. prefetch K[0..kStage-2], V[0..kStage-2]│  │ │    │ c. for each KV tile:                    │   │ │    │    - prefetch next K/V (TMA async)      │   │ │    │    - wait K barrier                     │   │ │    │    - P = QK^T (GMMA, warpgroup)         │   │ │    │    - causal mask                        │   │ │    │    - online_softmax(P) → scale & update │   │ │    │    - wait V barrier                     │   │ │    │    - Y += PV (GMMA, warpgroup)          │   │ │    │ d. final_online_softmax(Y)              │   │ │    │ e. store Y (TMA store)                  │   │ │    └─────────────────────────────────────────┘   │ └─────────────────────────────────────────────────┘` 

**Online Softmax** 是核心算法 — 不需要存储完整的 attention matrix：

`// 对每个 KV tile: row_max = max(P[m,:])                    // 当前 tile 的最大值 gMax[m] = max(gMax[m], row_max)          // 全局最大值更新 scale = exp2(last_max - gMax[m])         // 缩放因子 P[m,:] = exp2(P[m,:] - gMax[m])          // 重归一化 gSum[m] = gSum[m] * scale + sum(P[m,:])  // 累加求和 Y[m,:] = Y[m,:] * scale                  // 缩放历史输出 Y[m,:] += P[m,:] @ V[:,n]               // 累加当前 tile 贡献` 

---

## 四、Decode Kernel 层 — SplitK + 动态调度

[`decode.cc`](command:gongfeng.gongfeng-copilot.chat.open-relative-path?%7B%22%24mid%22%3A1%2C%22fsPath%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fdecode%2Fdecode.cc%22%2C%22path%22%3A%22%2Fdata%2Fhome%2Fivyjwu%2Fwork%2Fhpc-ops%2Fsrc%2Fattention%2Fdecode%2Fdecode.cc%22%2C%22scheme%22%3A%22file%22%7D) 分两条路径：

`有 task_map? → dynamic 路径 (sm90/dynamic/)                 CTA 按预分配的任务列表执行，负载均衡 无 task_map? → static 路径 (sm90/static/)                 CTA 按 (batch, head_kv) 静态分配，可能有负载不均衡` 

### SplitK 两阶段设计

`阶段1: SplitK Kernel (每个 CTA 处理 KV 的一段)   → 输出 split_out[batch, split_id, seq_q, head_q, dim_v] (float32)   → 输出 lse[batch, split_id, head_kv, seq_q, heads_per_group] (float32)  阶段2: Combine Kernel (合并 split_out → 最终输出 bf16)   → 使用 lse 做 log-sum-exp 归一化   → 最终输出 Y[batch*seq_q, head_q, dim_v] (bf16)` 

### 动态调度优势

Decode 场景下不同 batch 的 KV cache 长度差异巨大（1~8192），静态分配会导致长序列的 CTA 成为瓶颈。动态调度通过 `assign_attention_decode_task` 将所有 (batch, head_kv, chunk) 任务**均匀分配**到所有 CTA，消除长尾延迟。

---

## 五、FP8 量化方案

`QuantType` 枚举定义了 4 种量化粒度：

|值|名称|Q scale|K scale|V scale|
|---|---|---|---|---|
|0|`QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD`|per-token per-head|per-token-group per-head per-dim-group|per-head|
|1|`QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR`|per-token per-head|per-tensor (标量)|per-tensor (标量)|
|2|`QPERTENSOR_KPERTENSOR_VPERTENSOR`|per-tensor|per-tensor|per-tensor|
|3|`...QKHADAMARD`|同 0 + Hadamard 变换|同 0|同 0|

计算公式：`softmax(QK^T * qscale * kscale / sqrt(d)) * V * vscale`

目前 prefill 支持 type 0/1，decode 支持 type 0/1。`kFp8PrefillPScale = 256.0` 是一个固定的 P 矩阵缩放因子，防止 softmax 后的 FP8 中间结果溢出。

---

## 六、推荐学习路径

`Step 1: 跑通最简单的测试   → tests/test_attention_prefill_bf16.py   → 理解输入输出 shape 和参数含义  Step 2: 理解 Prefill dispatch 逻辑   → src/attention/prefill/prefill.cc (140行，简洁清晰)   → 理解 multi_stage vs warp_spec 的选择条件  Step 3: 深入一个具体 kernel   → src/attention/prefill/warp_spec_dim128.cu (93行，launch 入口)   → 追踪到 kernels.cuh 中的 attention_prefill_bf16_warp_specialization_kernel  Step 4: 理解 Decode 调度   → tests/test_attention_decode_qpertoken_perhead_kvpertensor_fp8.py   → 关注 task_map 的分配和使用流程  Step 5: 深入 Decode kernel   → src/attention/decode/sm90/static/ (static 路径，较简单)   → src/attention/decode/sm90/dynamic/ (dynamic 路径，理解任务调度)` 

`kernels.cuh` 是整个库最复杂的文件（132KB / 3766 行），建议先通过 `warp_spec_dim128.cu` 的 93 行入口理解整体框架，再逐步深入 kernel 内部的 CuTe 张量操作和 warp specialization 细节。
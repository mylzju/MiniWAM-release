# Stage 1：KL、重建与 SIGReg 消融计划

状态：SIGReg、四组配置、表征诊断与梯度标定工具已实现。真实数据短跑、权重锁定和正式实验待特征缓存就绪后执行；C 配置的 0.01 仅为试跑值。

## 1. 目标与实验范围

比较不同 Stage 1 训练方案对编码器表征和 Stage 2 策略成功率的影响。主实验采用 A/B/C/D 四组；预算只允许三组时保留 A/B/C，但 B→C 只能解释为整体方案变化。

默认先在仓库已有的 LIBERO + DINOv3 配方上开展，使用相同的特征缓存和潜变量容量。Wan-VAE、RoboTwin 作为后续复验，不与主实验混合汇总。

SIGReg 指 LeJEPA 的 Sketched Isotropic Gaussian Regularization。它约束样本集合的表征分布，KL 约束每个输入对应的后验，两者并不等价。参考 [论文](https://arxiv.org/abs/2511.08544) 和 [作者的最小实现](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md)。本实验将该正则接入动作监督训练，效果由本项目实验验证。

## 2. 固定实验矩阵

Stage 1 总目标：

```text
L = L_ID + lambda_rec * L_recon + beta(step) * L_KL
         + lambda_sigreg * L_SIGReg(mu)
```

| 组别 | Stage 1 目标 | lambda_rec | kl_beta | lambda_sigreg | use_kl / Stage 1 采样 |
| --- | --- | --- | --- | --- | --- |
| A | ID + recon + KL | 1.0 | 1e-6 | 0 | true / 保留 |
| B | ID + recon | 1.0 | 0 | 0 | true / 保留 |
| C | ID + SIGReg | 0 | 0 | 待标定的 λ | true / 保留 |
| D | ID only | 0 | 0 | 0 | true / 保留 |

- **A→B**：只移除 KL 损失；A 保留原有 1000 次前向的 KL warmup 语义，B 的权重始终为零。
- **B→D**：只移除重建，可分析重建在无 KL 条件下的贡献。
- **D→C**：只加入 SIGReg，可分析 SIGReg 在无重建条件下的贡献。
- **B→C**：同时移除重建、加入 SIGReg，只比较整体训练方案。
- **A→C**：比较完整方案，不能单独归因于某一项损失。

四组均设置 `lambda_action: 1.0`、Stage 1 `lambda_latent: 0.0`。C/D 同时设置 `model.reconstruction.decoder_enabled: false` 和 `lambda_rec: 0.0`。这里的 “ID only” 指只优化 ID 损失，仍保留与 A/B 一致的随机瓶颈。

`use_kl: true` 保留 `logvar_head`，`kl_beta: 0` 仅移除 KL 的加权损失。当前 `miniwam/model.py::_encode_view_slots` 显式传入 `sample=(self.stage == 1)`，所以仅调用 `eval()` 也不能关闭该路径的 Stage 1 采样。

可选扩展，均不计入首轮四组：

| 扩展组 | 目标与设置 | 可回答的问题 |
| --- | --- | --- |
| E | ID + recon + SIGReg；采样与 A/B 相同，KL=0 | E↔B 测量有重建时的 SIGReg 贡献；E↔A 在保留重建和采样时比较正则方案 |
| C_det / D_det | 分别复现 C/D，均设 `use_kl: false` | 在确定性编码器下复验 SIGReg 贡献 |

C_det↔C、D_det↔D 包含采样和后验头结构的变化；若要严格隔离采样本身，应另加保留后验头、仅令 `sample=False` 的显式控制，不把两类实验混写。

## 3. SIGReg 统计口径

### 约束对象与样本单位

1. 直接约束 Stage 2 使用的确定性后验均值 `mu`，不约束采样结果，不新增投影 MLP，也不对输入先做白化、标准化或 L2 归一化。
2. 对每个相机槽位，将单个样本的 `mu_tokens: [S,D]` 按固定顺序展平为 `[S*D]`，形成 `[N_valid,S*D]` 输入。主视角、腕部视角分别计算，左右腕槽位也分别统计。
3. 不将同一样本的空间 token 当作独立样本混入 batch；否则固定的位置编码也可能产生看似丰富的整体分布，而不同输入对应的表征已经相同。
4. 用 `main_present` / `wrist_present` 排除缺失相机。每槽位得到损失后，按有效样本数加权求平均；全局有效样本数不足 2 的槽位跳过并记录计数。该统计单位是本项目对 SIGReg 的适配，必须写入实验元数据。

### 数值实现

- 按作者最小实现采用随机单位方向、一维经验特征函数和 Epps–Pulley 积分；起始固定 `num_slices=256`、`num_points=17`、`t_max=3.0`。
- 保留该参考实现的对称积分权重和样本数乘子 `N_valid`，再对方向取平均。不得在调权重后静默改为另一种 reduction。
- 投影、三角函数、特征函数均值和积分使用 FP32，并在该区域显式关闭 autocast；需要时按投影方向分块控制显存。
- 每次前向重新采样方向，使用独立随机数生成器，避免改变动作噪声、扩散 timestep 和重参数化噪声的随机序列。记录并恢复生成器状态；验证使用固定方向和随机种子。
- 默认权重为零。关闭 SIGReg 时不执行额外编码、随机采样或分布式通信，保持原路径行为。
- 实现时固定参考源码 commit，并记录其许可和必要署名；新增依赖前先评估是否只需一个小型本地 PyTorch 模块。

### 多卡与梯度累积

SIGReg 的统计 batch 是**一次前向中、各卡合并后的有效样本**，不是优化器的累计 global batch。四组固定 GPU 数、每卡 microbatch 和梯度累积步数；记录每个槽位的实际统计样本数。

多卡使用相同投影方向，跨卡汇总可微的 cos/sin 总和与有效计数，然后计算全局特征函数。不能简单平均各卡已经平方后的局部 SIGReg。某卡缺少某槽位时仍按固定槽位顺序参与 collective，以零和/零计数贡献，避免因本地 presence 分支造成死锁。

梯度累积只累积各 microbatch 的正则梯度，不等价于对累计后的全部样本计算一次 SIGReg。首轮固定这一口径，不引入跨 step 队列。

## 4. 代码与配置交付

以下为实现接入点；四个 overlay 可直接使用，C 的权重仍需先在真实数据上标定。

| 位置 | 计划改动 |
| --- | --- |
| `miniwam/models/sigreg.py`（新增） | 实现 FP32 SIGReg、mask 和约定的统计 reduction |
| `miniwam/models/latent_encoder.py` | 复用现有 `posterior_parameters()`，让同一次 backbone 前向同时提供 mu 和采样结果；保持默认 `forward()` 四项返回值及现有行为 |
| `miniwam/model.py` | 在多视角路径保留有效 mu，汇总 SIGReg，并仅在 Stage 1 加入总损失；保持 ID 的条件与采样路径一致 |
| `miniwam/protocol.py` | 校验权重、积分参数、输入对象和统计模式；Stage 2 正则权重必须为零 |
| `miniwam/checkpointing.py` | 保存 SIGReg 来源、统计口径和随机数状态；resume 校验训练配置，Stage 1→2 允许停用训练正则；旧 checkpoint 缺少新字段时按关闭处理 |
| `configs/ablations/objective/` | 新增四个具名 overlay；必要时再补 E 和确定性版本 |
| `configs/ablations/README.md` | 说明组别、对比边界和两阶段命令 |
| `scripts/analysis/analyze_stage1_representations.py`（新增） | 固定样本、方向和种子的离线表征诊断，输出可比较的 JSON/CSV |

配置接口（其他默认项包括 `seed: 42`、`slice_chunk_size: 64`）：

```yaml
model:
  sigreg:
    target: mu
    sample_unit: flattened_grid_per_camera_slot
    num_slices: 256
    num_points: 17
    t_max: 3.0
    distributed_statistics: global_microbatch
loss:
  lambda_sigreg: 0.0
```

四个 overlay 为 `id_recon_kl.yaml`、`id_recon_no_kl.yaml`、`id_sigreg.yaml`、`id_only.yaml`。它们显式固定 `use_kl: true`，通过 `stage1:` 段控制训练目标，并通过 `stage2:` 段固定 `checkpoint.policy_init: fresh`、`loss.lambda_sigreg: 0.0`。C/D 关闭解码器。

已有 `objective/inverse_only.yaml` 只关闭重建，会继承基础配方的 KL，不能直接作为这里的 D。

同一实验 overlay 传给两个阶段；配置加载器据文件名自动匹配 Stage 1 输出。正式生成配置时验证解析结果，防止 Stage 2 误载基线或其他 seed 的编码器。

## 5. Stage 2 与公平性约束

所有组统一：

- `checkpoint.policy_init: fresh`，仅加载各自 Stage 1 的主/腕编码器并冻结；其余策略参数用相同 Stage 2 seed 初始化。
- 使用 `sample=False` 的 mu 作为目标；保留与对应 Stage 1 一致的 `use_kl` 结构以满足严格加载，即使 Stage 2 不用采样。
- `lambda_action: 1.0`、`lambda_latent: 1.0`，KL、重建和 SIGReg 的训练权重均为零。
- 固定数据清单、特征缓存版本、动作归一化、horizon、latent grid、通道数、优化器、学习率、batch、训练长度、保存节点和评估协议。
- Stage 1 和 Stage 2 均从各组规定的起点训练；B 不从 A 的训练结果继续训练。

`fresh` 的同 seed 不能只靠配置声明：检查 Stage 2 加载完成后的非编码器参数一致。不同 decoder 或正则模块的构造可能消耗 RNG，应保证它们不会改变后续策略模块的初始参数。四组共同存在的 Stage 1 模块也做相同检查。

LIBERO 保留当前配方的 Stage 1 16,260 / Stage 2 54,200 次优化器更新。A 的训练目标沿用原配方，但四组采用统一 fresh 初始化，因此 A 的 Stage 2 结果应重新运行，不能直接引用 `policy_init: stage1` 的发布基线。

不同目标的 mu 尺度可能不同，会影响 Stage 2 去噪目标和 loss。首轮保留所有组相同的 scheduler、目标处理和权重，记录尺度；如后续进行目标归一化，必须作为所有组一致的新实验，单独报告。

## 6. 权重标定与执行顺序

1. **实现与配置检查**：完成损失、一次编码复用、统计通信、四个 overlay、checkpoint 兼容和诊断脚本。
2. **短程 smoke**：四组各跑约 100 次更新；确认损失开关、采样行为、编码器梯度、presence mask 和 Stage 1→2 加载；验证 SIGReg 关闭时原行为保持一致。
3. **标定 SIGReg 权重**：在固定训练子集、固定 batch 上检查原始损失及其对共享编码器参数的梯度范数。以加权 SIGReg/ID 梯度比约 0.1 的数量级作为起始 λ，再测试 `{λ/10, λ, 10λ}`，各约 500–1000 次更新。该比值仅用于确定搜索尺度，不作为效果保证；若梯度接近零，先检查实现和统计样本。
4. **锁定 λ 与协议**：依据数值稳定性、留出集 ID 和 mu 诊断排除失败候选；如候选接近，可使用预先划定的开发任务做相同预算的 Stage 2 pilot。记录所有试验及计算开销，正式测试集不参与选择。
5. **单 seed 主实验**：A/B/C/D 均用训练 seed 42 完成两个阶段，再统一评估。Stage 2 保存节点和最终 checkpoint 选择规则事先固定，默认比较相同最终更新步。
6. **多 seed 确认**：预算允许时四组补训练 seed 43、44；每个 seed 在两个阶段和对应组之间配对。若只能重复部分组，优先 C/D，并明确其他比较仍只有单 seed。
7. **可选扩展**：根据研究问题再做 E 或 C_det/D_det，沿用已经锁定的规则；如重新调参，记录独立调参预算。

不将 `1e-6` 的 KL 系数直接用于 SIGReg。正式报告区分调参运行和主实验，不能只展示最佳 λ/seed。

多 seed 输出目录采用 `outputs/<benchmark>_<arm>_seed<seed>_stage<stage>`，显式设置 `--output_dir` 和 Stage 2 的 `--stage1_checkpoint`；只修改 `--seed` 不会自动改变输出目录。

命令模板（先标定 C 的权重，再进行正式比较）：

```bash
python scripts/train.py --config configs/libero/stage1.yaml \
  configs/ablations/objective/id_sigreg.yaml --seed 42 \
  --output_dir outputs/libero_C_seed42_stage1

python scripts/train.py --config configs/libero/stage2.yaml \
  configs/ablations/objective/id_sigreg.yaml --seed 42 \
  --stage1_checkpoint outputs/libero_C_seed42_stage1/latest \
  --output_dir outputs/libero_C_seed42_stage2
```

## 7. 诊断、评估与验收

### 诊断和效果指标

| 层次 | 指标与口径 |
| --- | --- |
| Stage 1 优化 | 原始/加权 ID、recon、KL、SIGReg；定期记录 ID/SIGReg 对编码器的梯度范数 |
| 表征 | 每相机槽位 mu 均值、标准差分位数、协方差谱/有效秩、跨样本距离；额外检查每 token 位置的跨样本方差，记录展平维度及样本数导致的秩上限 |
| 随机瓶颈 | logvar、后验 std 的分布和极值、非有限值计数；无 KL 时仍需检查这些量 |
| Stage 2 | action loss、latent loss、固定更新步上的收敛曲线；同时记录目标 mu 尺度与实际训练耗时/峰值显存 |
| 最终任务 | LIBERO 四个 suite 的成功率及宏平均；固定 LIBERO-Plus-2K manifest 的成功率及扰动分项 |

验证数据在训练前按 episode 划分并锁定，两阶段、所有组采用同一排除规则，避免相邻帧泄漏；这会成为四组共同的数据协议调整，须与原配方区分。若必须保留原配方的全部训练数据，则只报告固定训练子集诊断，不称为验证集，也不用正式测试结果调 λ。

ID 诊断固定动作噪声、扩散 timestep 和潜变量采样随机种子，必要时对相同的多个噪声种子求平均；mu 的确定性诊断单独报告。Stage 1 总 loss 因组间目标不同，不能直接用于排序。

使用现有 `docs/evaluation.md` 的 LIBERO 协议及 `scripts/eval/manifests/libero_plus_ablation_2k_seed42.txt`。同一测试任务、初始状态和 policy seed 在各组之间配对；训练 seed 与评估协议 seed 分开记录。汇报每个训练 seed 的结果、均值/标准差和组间差值；任务配对 bootstrap 区间与跨训练 seed 的离散程度分开报告。

### 必须通过的实现检查

- **数值与梯度**：固定输入/方向时匹配参考实现；足够样本下标准高斯统计量低于常量塌缩、显著平移/缩放分布；常量输入损失有限，不要求其梯度必非零；正常输入梯度有限并到达 mu/backbone。
- **统计单位**：所有输入产生相同网格、但网格内 token 各不相同时，仍能检测跨样本塌缩；缺失槽位的占位特征不影响损失或梯度。
- **分布式**：两卡含不等有效计数、某卡无腕部相机时，损失和编码器梯度匹配单进程拼接参考，无重复 world-size 缩放或 collective 死锁。
- **行为回归**：A/B 的结构和采样路径一致，B/C/D 的 KL 加权值为零；C/D 差异仅为 SIGReg；关闭 SIGReg 不消耗其 RNG；原 posterior API 测试通过。
- **迁移与续训**：四组 Stage 1→2 均可严格加载；冻结编码器；非编码器参数同 seed 一致；旧 checkpoint 可在关闭 SIGReg 时使用；启用 SIGReg 的 resume 恢复随机方向序列。
- **配置解析**：覆盖 LIBERO 和 RoboTwin overlay 解析，核对输出目录、对应 Stage 1 路径、两阶段权重与采样结构；不改动原有 `inverse_only.yaml` 的语义。

交付物为实现、四组配置、诊断工具及其检查结果，随后产生逐 seed 的训练/评估记录和对比表。实验结论根据实际成功率与不确定性填写；SIGReg 损失下降或有效秩提高都不能单独证明策略提升。

## 8. 已提供的诊断入口

```bash
python scripts/analysis/analyze_stage1_representations.py \
  --config configs/libero/stage1.yaml configs/ablations/objective/id_sigreg.yaml \
  --features_root /path/to/features --device cuda:0 \
  --batch_size 16 --max_samples 256 --calibrate \
  --output outputs/sigreg_calibration.json
```

不更新参数。校准输出包含原始 ID / SIGReg 梯度范数、0.1 梯度比对应的建议搜索中心及十倍范围；它不是已经验证的最优 λ。多卡训练应以 global microbatch 样本量做单卡校准（显存允许时），不能用 local microbatch 的结果充当同一统计。

分析已训练编码器时，将 `--config ...` 换成 `--checkpoint <stage1>`。工具输出 JSON、CSV 和固定帧索引 `.indices.json`；其他组使用同一根目录顺序及 `--sample_indices` 复用样本。离线 SIGReg 诊断固定为 256 方向、17 节点、[0,3] 对称积分，统计全部选中样本；梯度校准使用模型配置及首个 microbatch，两者在输出中区分。

默认只标记为 `training_diagnostic`；`--split_label heldout` 本身不创建划分。通过 `data.episode_manifest` 或 `--episode_manifest` 指定并强制筛选 episode，清单摘要写入报告。当前已下载并校验原始 LIBERO 数据；正式消融尚无性能结果。本机执行入口和固定划分见 [运行说明](running_sigreg_experiments.md)。

## 9. 实施检查记录（2026-11-03）

- `python -m pytest -q`：294 passed、6 skipped；其中 3 项 GPU 测试在沙箱内因无法访问 CUDA 跳过，另 3 项依赖外部 RoboTwin 安装。
- 在可访问 CUDA 的环境单独运行 `python -m pytest -q tests/test_sigreg_cuda.py`：3 passed，覆盖 FP32 统计与 BF16 autocast、四组各 100 次合成 DINO 特征更新，以及双卡 NCCL/FSDP 某卡缺少整个腕部相机角色的训练。
- 两进程 Gloo 测试验证不同有效样本数下的损失与参数平均梯度匹配单进程全局参考；全局 RNG 不受投影采样影响，SIGReg 自身的随机序列可保存/恢复。
- A/B/C/D 的两阶段配置、同 seed 公共参数初始化、fresh encoder transfer、旧 checkpoint 协议兼容及诊断 CLI 输出均已检查。

这些是实现和数值稳定性检查，不构成真实数据的消融性能结果。真实数据 100-step smoke、500–1000-step λ 搜索、两阶段正式训练和成功率评估尚未运行。

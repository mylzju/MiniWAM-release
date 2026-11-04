# 本机 LIBERO 消融实验

使用 `scripts/experiments/run_libero_sigreg.py` 执行 seed 42 的 A/B/C/D 实验，详细研究设计见 [消融计划](sigreg_ablation_plan.md)。

数据共 1,712 条演示、277,713 帧。每个任务固定留出 2 条演示，训练 1,632 条、263,999 帧，开发集 80 条、13,714 帧。划分清单及 SHA256 位于 `outputs/libero_sigreg_seed42/protocol/`。动作和本体感觉的归一化统计只用训练演示计算；缓存包含全部演示，两个训练阶段都按清单排除开发集。

双卡固定每卡 batch 32、梯度累积 4，优化器 global batch 为 256，SIGReg 统计 batch 为 64。四组 Stage 1 都保留随机潜变量采样，Stage 2 都仅继承自己的编码器并采用 `policy_init: fresh`。该数据划分和 fresh 初始化与原发布配方有差别。

执行顺序：

1. 校验 DINOv3、UMT5 和环境；用一条演示检查提取路径，再双卡分片生成全部 DINOv3 / UMT5 缓存。
2. 四组分别做 Stage 1 / Stage 2 各 100 次真实数据更新，检查损失开关、checkpoint 迁移和开发集表征。
3. 用固定 64 样本校准 ID / SIGReg 编码器梯度，以梯度比 0.1 的 λ 为中心；三个候选各训练 500 次更新。
4. 依据 256 个固定开发集帧、噪声 seed 101/102/103 的平均 ID loss 选择 λ。所有统计必须有限；每个相机槽位的协方差迹须大于 1e-8、有效秩须大于 1.01。失败会停止队列，不用正式 rollout 成功率选 λ。
5. 四组分别从头完成 Stage 1 的 16,260 次和 Stage 2 的 54,200 次更新；Stage 2 每 5,420 次保存。短跑与候选试验不作为正式训练起点。

启动或在修复失败后重新启动（先确认没有同一队列在运行；脚本有进程锁）：

```bash
export MINIWAM_DATA_ROOT='/mnt/ugreen-nas/MiniWAM-data'  # 替换为自己已挂载、可写的 NAS 根目录
conda run --no-capture-output -n miniwam python scripts/experiments/run_libero_sigreg.py \
  --data_root "$MINIWAM_DATA_ROOT/LIBERO-fastwam" \
  --features_root "$MINIWAM_DATA_ROOT/dino_features_libero_sigreg_split42"
```

原始数据和特征缓存均在 NAS 根目录下；路径写入锁定协议，改路径时使用新的输出目录。

`--through preprocess|smoke|calibration|training` 可指定停止阶段，默认 `training`。已成功且命令不变的阶段会跳过；失败提取按已完成 episode 继续，失败训练从相同 seed 重新开始该次运行。未实现中断后逐位复现完整训练随机数状态，因此不会自动恢复半途训练。`--wait_assets 7200` 只等待已经启动的 UMT5 下载，不发起新下载。

W&B 安装、登录后，可添加 `--wandb --wandb_project miniwam-sigreg`，每个训练阶段会使用独立的 run 名称。保持锁定的 `protocol/shared.yaml` 原样，使用命令行开关启用日志。已完成阶段的命令指纹包含参数和 Python 路径，因此在新服务器或从关闭 W&B 改为开启时，使用新的 `--output_dir` 保留历史记录并复用特征缓存。完整步骤见 [服务器迁移与训练指南](server_migration_zh.md)。

运行状态：

- `outputs/libero_sigreg_seed42/status.json`：当前阶段、子进程 PID 和日志位置。
- `outputs/libero_sigreg_seed42/logs/<阶段名>.log`：各阶段详细输出。
- `outputs/libero_sigreg_seed42/runs/*/metrics.jsonl`：rank 0 的训练指标；`latest/` 指向完整 checkpoint。
- `outputs/libero_sigreg_seed42/diagnostics/`：开发集 ID loss、mu / logvar 统计和固定采样索引。
- `outputs/libero_sigreg_seed42/protocol/lambda_selection.json`：全部候选、分数和选择结果。

发送 SIGTERM 给 `outputs/libero_sigreg_seed42/queue.pid` 中的队列 PID 可停止队列及它创建的当前训练/提取进程。不要停止机器上其他已有任务。

队列完成表示训练完成；LIBERO / LIBERO-Plus 的模拟器 rollout 成功率评估仍需后续执行，不能把短跑 loss 或表征秩当作最终实验结论。

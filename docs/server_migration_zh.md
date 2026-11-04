# MiniWAM 新服务器联网重建与训练指南

更新：2026-11-04。主线对应当前 **LIBERO + DINOv3 + SIGReg A/B/C/D 消融**，训练 seed 42。本文按**新服务器从网上下载依赖、模型权重和原始数据，再重新提取特征、从头训练**编排。按第 1–8 节执行即可开始训练。

新机直接从自己的 GitHub 仓库克隆包含 SIGReg 实验的完整代码，**DINOv3 从 ModelScope（魔搭）下载**，UMT5 和 LIBERO 数据从 Hugging Face 下载；约 158GiB 的特征缓存由新机生成，历史实验输出留在旧机。原始数据压缩包和模型资产下载量合计约 15–16GiB，另加环境依赖和 Git 仓库流量。按本文流程，无需从旧服务器传文件。

新服务器示例目录是 `/data/MiniWAM-release`，请按实际数据盘修改。命令使用 Bash，除明确标注旧服务器外均在新服务器的仓库根目录执行。适用于 Linux x86_64、Ubuntu 22.04/24.04；SSH 和 GPU 驱动由服务器提供方配置。

## 1. 当前状态与资源预算

| 项目 | 当前状态 / 建议 |
| --- | --- |
| 已验证硬件 | 2 × RTX A6000 48GB，驱动 580.178.04，约 251GiB 内存 |
| 新机建议 | 双卡、每卡 48GB 或更多；内存 128GiB 或更多。这是建议，未验证最低内存需求 |
| 环境 | Python 3.10；torch 2.11.0+cu128；torchvision 0.26.0+cu128 |
| 关键依赖 | numpy 1.26.4、transformers 4.57.6、protobuf 4.25.3、wandb 0.27.2 |
| 原始数据 | 新机下载四个压缩包，合计约 4.4GiB；`datasets/LIBERO-fastwam/` 含压缩包及解压数据约 8.8GiB |
| 特征缓存 | 新机重新生成 `datasets/dino_features_libero_sigreg_split42/`，约 158GiB |
| 模型资产 | 新机下载 DINOv3、UMT5 权重和 tokenizer，`pretrained/` 约 11GiB |
| 实验输出 | 新建 `outputs/libero_sigreg_seed42_newserver/`，从短跑、标定开始执行 |
| 磁盘预算 | 原始数据、模型和特征约 178GiB，另加环境和后续上百 GiB 的 checkpoint。建议至少 500GiB 空闲，优先 1TB SSD/NVMe 数据盘 |
| 训练布局 | 2 卡 × 每卡 batch 32 × 梯度累积 4 = 256；SIGReg 统计 batch = 64 |
| 数据划分 | 1,632 条训练演示、80 条开发集演示；归一化统计仅使用训练演示 |

旧机 11 月 3 日完成了 A/B/C 两阶段各 100 步短跑；D 的 Stage 1 在约 95 步收到 SIGTERM，未保存 checkpoint。λ 标定和正式训练尚未开始。新机重新执行短跑和标定。

**使用包含 SIGReg 实验的代码版本。** 本文与 SIGReg 实现、四组配置、实验队列一同提交。第 3 节直接克隆 GitHub 的 `main` 分支、检查所需源码并记录实际 commit；旧的初始发布 commit `b639db68bb4be2c2113fc20c27b44c1e382f9dcd` 不包含这些实验代码。

新机优先用 580 系列或更新且支持目标 GPU 的驱动。CUDA 12.x 的最低兼容范围仍有功能限制，应以实际 CUDA 运算验证为准。[NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

## 2. 新服务器系统准备

**以下均在新服务器执行：**

```bash
export MINIWAM_ROOT='/data/MiniWAM-release'
mkdir -p "$MINIWAM_ROOT"
cd "$MINIWAM_ROOT"

cat /etc/os-release
uname -m
nvidia-smi
df -h . /tmp
free -h

sudo apt-get update
sudo apt-get install -y \
  python3 curl wget ca-certificates git git-lfs openssh-client tar tmux \
  build-essential cmake pkg-config ffmpeg \
  libegl1 libgl1 libglib2.0-0 libosmesa6 libsm6 libxext6 libxrender1
```

架构应为 `x86_64`。PyTorch CUDA wheel 包含所需 CUDA 用户态运行库，此流程无需另装完整 CUDA Toolkit。APT 默认源慢时可先按附录 A 切换；APT、pip、Conda、PyTorch、ModelScope、HF 分别使用不同下载服务。

## 3. 在新机从 GitHub 获取完整代码

**新服务器：** 使用第 2 节准备的空目录。

```bash
export MINIWAM_ROOT='/data/MiniWAM-release'
git clone --branch main https://github.com/mylzju/MiniWAM-release.git "$MINIWAM_ROOT"
cd "$MINIWAM_ROOT"
mkdir -p migration_artifacts
git rev-parse HEAD | tee migration_artifacts/git-head.txt
git status --short

python3 - <<'PY'
from pathlib import Path

required = [
    'miniwam/models/sigreg.py', 'miniwam/episode_split.py',
    'scripts/experiments/prepare_libero_sigreg.py',
    'scripts/experiments/run_libero_sigreg.py',
    'scripts/analysis/analyze_stage1_representations.py',
    'scripts/benchmark_download_mirrors.py',
    'scripts/download_dinov3_modelscope.sh',
    *[f'configs/ablations/objective/{name}.yaml'
      for name in ('id_recon_kl', 'id_recon_no_kl', 'id_sigreg', 'id_only')],
]
missing = [name for name in required if not Path(name).is_file()]
assert not missing, f'Missing SIGReg source files: {missing}'
print('SIGReg source files ready')
PY
```

源码检查应输出 `SIGReg source files ready`。如需复现某个固定版本，在克隆后先用 `git checkout --detach <完整 commit>` 切换，再执行版本记录和源码检查。后续按第 4–6 节联网安装环境、下载资产。

`migration_artifacts/` 只保存本机版本记录、测速结果和环境变量，已加入 `.gitignore`。数据、权重、特征缓存、实验输出和 Python 环境也由 `.gitignore` 排除，均按后文在新机下载或生成。

## 4. 比较国内源速度并选源

### 4.1 在新服务器实测

脚本只依赖系统 Python 标准库，不安装包、不修改全局配置：

```bash
mkdir -p migration_artifacts
python3 scripts/benchmark_download_mirrors.py \
  --rounds 2 --mib 8 --timeout 20 --prefer-domestic \
  --output migration_artifacts/mirrors.json \
  --write-env migration_artifacts/mirrors.env

source migration_artifacts/mirrors.env
cat migration_artifacts/mirrors.env
```

| 类别 | 候选 | 测试文件 |
| --- | --- | --- |
| pip | 清华、阿里云、中科大、交大、官方 | 同一个 numpy 1.26.4 / Python 3.10 / Linux wheel |
| PyTorch cu128 | 交大、官方 | torch 2.11.0+cu128 / Python 3.10 / Linux wheel |
| conda-forge | 清华、中科大、北外、官方 | `linux-64/repodata.json.zst` |
| HF | HF-Mirror、官方 | 固定 revision 的 LIBERO Goal 压缩包 |

每次读取前 8MiB，顺序下载，第二轮反转候选顺序；按两轮都成功时的**索引解析加下载的中位耗时**选择。检查二进制文件头，避免把镜像验证页当作高速文件。文件内容下载量最多约 208MiB，另有索引流量。

JSON 的 `best` 为全部候选中最快的，`best_domestic` 为国内最快的。`--prefer-domestic` 优先将可用国内源写入 env；国内候选全部失败时回退到可用官方源。希望按绝对速度选源，去掉该参数重测。

生成四个变量：`MINIWAM_PIP_INDEX`、`MINIWAM_TORCH_INDEX`、`MINIWAM_CONDA_CHANNEL`、`MINIWAM_HF_ENDPOINT`。HF 测速及端点选择用于 UMT5 和 LIBERO 数据；DINOv3 固定从 ModelScope 下载。某一类别全部失败时脚本返回非零，先检查 JSON 错误、网络或代理，再重测。单独重测可用 `--groups pip` 等；`--write-env` 会覆盖文件，单组测试请另起文件名。

### 4.2 旧服务器的实测结果

2026-11-04 北京时间 10:21 开始，未设置显式代理环境变量；两轮、每轮 8MiB、下载时限 15 秒。下表为下载速度中位数，单位 MiB/s；选择源的总耗时还包含索引解析。[完整测量 JSON](assets/mirror_benchmark_20261104.json)

| 类别 | 清华 | 中科大 | 阿里云 | 交大 / 北外 / HF-Mirror | 官方 |
| --- | ---: | ---: | --- | --- | ---: |
| pip | 4.07 | **25.00** | 两轮超时 | 交大 3.55 | 2.24 |
| conda-forge | 4.67 | **10.29** | — | 北外 4.57 | 2.10 |
| PyTorch cu128 | — | — | — | 交大两轮 HTTP 502 | **2.08** |
| HF 文件 | — | — | — | HF-Mirror 1.23 | **1.43** |

**当前网络：pip/conda-forge 用中科大，PyTorch 用官方 cu128，HF 官方略快。** 新机必须以自己的测速为准。8MiB 采样不保证代表 11GB 大文件的持续速度，也不涵盖完整依赖求解时间。

需要手动回退时，已验证的候选为：

```bash
export MINIWAM_PIP_INDEX='https://mirrors.ustc.edu.cn/pypi/simple/'
export MINIWAM_CONDA_CHANNEL='https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/'
export MINIWAM_TORCH_INDEX='https://download.pytorch.org/whl/cu128/'
export MINIWAM_HF_ENDPOINT='https://huggingface.co/'
```

后面默认加载 `mirrors.env`。若手动选源后需要保持选择，将相应 export 行也写入该文件。

## 5. 创建固定版本的环境

### 5.1 安装 Miniforge

已有 Conda 时跳过安装，加载它的 `conda.sh` 后进入 5.2。没有时：

```bash
export MINIWAM_BOOTSTRAP_DIR="$(mktemp -d /tmp/miniwam-bootstrap.XXXXXX)"
cd "$MINIWAM_BOOTSTRAP_DIR"
curl -fL --retry 3 \
  https://mirrors.ustc.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-26.9.4-1-Linux-x86_64.sh \
  -o Miniforge3-26.9.4-1-Linux-x86_64.sh
curl -fL --retry 3 \
  https://mirrors.ustc.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-26.9.4-1-Linux-x86_64.sh.sha256 \
  -o Miniforge3-26.9.4-1-Linux-x86_64.sh.sha256
sha256sum -c Miniforge3-26.9.4-1-Linux-x86_64.sh.sha256 && \
  bash Miniforge3-26.9.4-1-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
cd "$MINIWAM_ROOT"
```

安装器固定为 26.9.4-1。镜像更新后不再提供该文件时，将两条 URL 的共同前缀换成 `https://github.com/conda-forge/miniforge/releases/download/26.9.4-1/`。安装目录已存在时使用现有安装，不重复安装。

### 5.2 配置 Conda、pip 和 CUDA wheel 源

```bash
source migration_artifacts/mirrors.env
cat > migration_artifacts/condarc <<EOF
channels:
  - ${MINIWAM_CONDA_CHANNEL}
channel_priority: strict
show_channel_urls: true
EOF
export CONDARC="$MINIWAM_ROOT/migration_artifacts/condarc"

conda create -y -n miniwam --override-channels \
  -c "$MINIWAM_CONDA_CHANNEL" \
  python=3.10 pip setuptools wheel packaging ninja
conda activate miniwam
export PYTHONNOUSERSITE=1
conda env config vars set PYTHONNOUSERSITE=1
python -m pip config --site set global.index-url "$MINIWAM_PIP_INDEX"
python -m pip install --index-url "$MINIWAM_PIP_INDEX" --upgrade pip setuptools wheel

python -m pip install \
  --index-url "$MINIWAM_PIP_INDEX" \
  --extra-index-url "$MINIWAM_TORCH_INDEX" \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
```

基础环境使用 conda-forge，PyTorch 明确安装 `cu128` 构建。交大缺少对应 wheel 或返回 502 时，使用官方 `https://download.pytorch.org/whl/cu128/`；普通依赖仍可从国内 pip 源获取。

### 5.3 安装其余固定依赖和 W&B

`environment.yml` 自带固定源。下面提取它的版本约束，并显式使用刚选的源：

```bash
python -m pip install --index-url "$MINIWAM_PIP_INDEX" 'PyYAML==6.0.3'
python - <<'PY'
from pathlib import Path
import yaml

environment = yaml.safe_load(Path('environment.yml').read_text())
requirements = next(item['pip'] for item in environment['dependencies'] if isinstance(item, dict) and 'pip' in item)
requirements = [item for item in requirements if not item.startswith(('--', 'torch==', 'torchvision=='))]
Path('migration_artifacts/requirements-pinned.txt').write_text('\n'.join(requirements) + '\n')
PY

python -m pip install --index-url "$MINIWAM_PIP_INDEX" -r migration_artifacts/requirements-pinned.txt
python -m pip install --index-url "$MINIWAM_PIP_INDEX" -e '.[logging]'
python -m pip check

python - <<'PY'
import torch, torchvision, numpy, transformers, wandb
from importlib.metadata import version

print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('torchvision:', torchvision.__version__)
print('numpy:', numpy.__version__, 'transformers:', transformers.__version__)
print('protobuf:', version('protobuf'), 'wandb:', wandb.__version__)
assert torch.__version__ == '2.11.0+cu128'
assert torchvision.__version__ == '0.26.0+cu128'
assert version('protobuf') == '4.25.3'
assert wandb.__version__ == '0.27.2'
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2, '默认 SIGReg 队列需要两张可见 GPU'
assert torch.cuda.is_bf16_supported(), '此配方使用 BF16'
x = torch.ones((32, 32), device='cuda')
print('GPU computation:', (x @ x).mean().item())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

`pip check` 应无冲突。**W&B 固定 0.27.2，protobuf 固定 4.25.3。** 旧机 W&B 0.31.0 不支持当前 protobuf，导致 `cannot import name 'Imports'`。按项目版本安装，保持 protobuf 约束。[W&B 0.27.2 依赖声明](https://github.com/wandb/wandb/blob/v0.27.2/pyproject.toml)

### 5.4 安装固定 LIBERO 并生成新机路径

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
git -C LIBERO checkout --detach 8f1084e3132a39270c3a13ebe37270a43ece2a01
python - <<'PY'
from pathlib import Path
import yaml

root = Path.cwd()
libero = root / 'LIBERO/libero/libero'
config = {
    'assets': str(libero / 'assets'),
    'bddl_files': str(libero / 'bddl_files'),
    'benchmark_root': str(libero),
    'datasets': str(root / 'LIBERO/libero/datasets'),
    'init_states': str(libero / 'init_files'),
}
Path('.libero').mkdir(exist_ok=True)
Path('.libero/config.yaml').write_text(yaml.safe_dump(config))
PY
export LIBERO_CONFIG_PATH="$MINIWAM_ROOT/.libero"
python scripts/verify_miniwam_environment.py --require-cuda --check-libero
```

已有干净、相同 commit 的 LIBERO checkout 时跳过 clone/checkout，但重新生成路径配置。不要沿用旧机 `.libero/config.yaml`。

### 5.5 保存后续会话要加载的变量

```bash
cat > migration_artifacts/runtime.env <<EOF
export MINIWAM_ROOT="$MINIWAM_ROOT"
export MINIWAM_RUN_DIR="outputs/libero_sigreg_seed42_newserver"
export PYTHONNOUSERSITE=1
export HF_HUB_CACHE="$MINIWAM_ROOT/pretrained/hf-cache"
export HF_ENDPOINT="${MINIWAM_HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LIBERO_CONFIG_PATH="$MINIWAM_ROOT/.libero"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR=/tmp/miniwam_numba_cache
export MPLCONFIGDIR=/tmp/miniwam_matplotlib
export WANDB_PROJECT=miniwam-sigreg
export WANDB_RUN_GROUP=libero-sigreg-seed42
export WANDB_MODE=online
EOF
source migration_artifacts/runtime.env
```

运行变量默认使用离线模型加载。第 6.1.2 和 6.3 节下载 HF 资产时会临时取消离线开关，第 6.4 节恢复；ModelScope 下载不受这两个开关影响。

## 6. 在新机下载模型和原始数据

| 内容 | 网上来源 | 新机处理 |
| --- | --- | --- |
| DINOv3 ViT-B/16 | [ModelScope：facebook/dinov3-vitb16-pretrain-lvd1689m](https://modelscope.cn/models/facebook/dinov3-vitb16-pretrain-lvd1689m) | 固定版本下载、逐文件校验，写入本地离线缓存 |
| UMT5 tokenizer | [google/umt5-xxl](https://huggingface.co/google/umt5-xxl/tree/66cb9e7e85526fe440a945569e42c72fb6cbc0ad) | 只下载 tokenizer 文件到 Hub cache |
| UMT5 文本编码器 | [Wan-AI 的指定权重文件](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/blob/b62153b54fce61f5908de23f6d8715e147de0daf/models_t5_umt5-xxl-enc-bf16.pth) | 只下载约 11GB 的单文件到 `pretrained/language/` |
| LIBERO 原始数据 | [yuanty/LIBERO-fastwam](https://huggingface.co/datasets/yuanty/LIBERO-fastwam) | 下载固定 revision 的四个 LeRobot 2.1 压缩包并校验、解压 |
| DINO/UMT5 特征缓存 | 由上述原始数据和模型计算 | 第 8.3 节在新机生成约 158GiB 缓存 |

DINOv3 使用 ModelScope 下载脚本；UMT5 和原始数据使用第 4 节选出的 HF 端点。下载中断后重跑对应命令，可复用已验证文件和未完成的下载。[HF CLI 说明](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/cli)

### 6.1 下载模型权重和 tokenizer

#### 6.1.1 DINOv3：从 ModelScope 下载

```bash
export HF_HUB_CACHE="$MINIWAM_ROOT/pretrained/hf-cache"
bash scripts/download_dinov3_modelscope.sh
```

脚本使用系统 `curl` 和 `sha256sum`，固定 ModelScope commit `23d0280ae6ee4ced592a3459674ad027d3c18906`，下载模型、配置、预处理配置和许可/说明文件。五个文件全部通过 SHA256 校验后，完成本地缓存登记；权重文件为 342,662,192 字节。

这些文件与现有实验固定的 DINOv3 版本逐字节一致，因此本地缓存仍使用配置中的 revision `5931719e67bbdb9737e363e781fb0c67687896bc`。`pretrained/hf-cache/` 是 Transformers 的**本地兼容缓存目录**；DINOv3 的网络下载请求只发往 ModelScope，无需 HF 模型访问申请或 HF 登录。训练、预检和评估按原有模型标识离线读取即可。

#### 6.1.2 UMT5：下载 tokenizer 和文本编码器

这两个 HF 仓库为公开资产，通常无需登录：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HUB_CACHE="$MINIWAM_ROOT/pretrained/hf-cache"
export HF_ENDPOINT="${MINIWAM_HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

(
  set -e
  hf download google/umt5-xxl \
    --revision 66cb9e7e85526fe440a945569e42c72fb6cbc0ad \
    --include '*token*' '*.model'
  hf download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth \
    --revision b62153b54fce61f5908de23f6d8715e147de0daf \
    --local-dir pretrained/language

  python - <<'PY'
import os
from pathlib import Path

# 固定下载的是 commit；离线解析默认分支时也指向这版 tokenizer。
ref = Path(os.environ['HF_HUB_CACHE']) / 'models--google--umt5-xxl/refs/main'
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text('66cb9e7e85526fe440a945569e42c72fb6cbc0ad')
PY
)
```

本次不需要 Wan-VAE、Wan 视频扩散模型或发布训练 checkpoint；UMT5 只下载指定的文本编码器单文件。

UMT5 和数据可使用 HF-Mirror。出现 `xet-read-token` 404 时先 `export HF_HUB_DISABLE_XET=1` 再重跑；镜像缺少固定 revision 时，改 `export HF_ENDPOINT=https://huggingface.co` 重试。需要 HF 登录时用 `HF_ENDPOINT=https://huggingface.co hf auth login`；DINOv3 的 ModelScope 下载独立于这些设置。[镜像端点说明](https://hf-mirror.com/)

### 6.2 验证下载的模型资产

```bash
bash scripts/download_dinov3_modelscope.sh --check-only
sha256sum -c <<'EOF'
7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d  pretrained/language/models_t5_umt5-xxl-enc-bf16.pth
EOF
python scripts/verify_miniwam_environment.py \
  --require-cuda --check-libero --check-assets --visual-encoder dinov3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
import torch
from transformers import AutoTokenizer
from miniwam.models.dinov3 import DINOv3FrameEncoder

dino = DINOv3FrameEncoder(revision='5931719e67bbdb9737e363e781fb0c67687896bc')
with torch.inference_mode():
    features = dino.encode_for_miniwam(torch.zeros(1, 3, 224, 224, dtype=torch.uint8))
assert features.shape == (1, 196, 768), features.shape
assert features.dtype == torch.bfloat16, features.dtype
print('Offline DINOv3 ready:', dino.resolved_revision, tuple(features.shape))
tokenizer = AutoTokenizer.from_pretrained('google/umt5-xxl', local_files_only=True)
print('Offline tokenizer ready:', type(tokenizer).__name__)
PY
```

全部通过后再进入预处理。队列会强制离线加载模型，因此权重、DINO snapshot 和 tokenizer 都需提前下载完整；提取完成后仍保留这些资产供后续预检使用。

### 6.3 下载并解压原始数据

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_ENDPOINT="${MINIWAM_HF_ENDPOINT:-https://huggingface.co}"
(
  set -e
  mkdir -p datasets/LIBERO-fastwam
  for suite in libero_10 libero_spatial libero_object libero_goal; do
    hf download yuanty/LIBERO-fastwam "${suite}_no_noops_lerobot.tar.gz" \
      --repo-type dataset \
      --revision ee018b997c430bb12b5bf3c892d744798c5a2f91 \
      --local-dir datasets/LIBERO-fastwam
  done
  cd datasets/LIBERO-fastwam
  sha256sum -c <<'EOF'
eb5af5f46d9348f8ba0146cf90de718ce102f261348983c0fd2df08d50dd1059  libero_10_no_noops_lerobot.tar.gz
7c374be75133e1a6779a0e07b9452c16294777be577b89553f1cf5350eafd8f9  libero_spatial_no_noops_lerobot.tar.gz
732405d5deaf3019c244bacd3f26c9617f25c871750586c9968747b6cfefedda  libero_object_no_noops_lerobot.tar.gz
a21ae10171535585fb43e6405d9efa09ff38ef34689e4176428ca005af3a39ea  libero_goal_no_noops_lerobot.tar.gz
EOF
  for suite in libero_10 libero_spatial libero_object libero_goal; do
    tar -xzf "${suite}_no_noops_lerobot.tar.gz"
  done
)
```

四个包合计 4,692,241,555 字节，约 4.4GiB；全部校验通过才会开始解压。使用这里固定的 LeRobot 2.1 压缩包，保持当前提取脚本需要的 `meta/episodes.jsonl` 目录结构。

### 6.4 验证原始数据数量

```bash
source migration_artifacts/runtime.env
python - <<'PY'
import json
from pathlib import Path
from scripts.experiments.prepare_libero_sigreg import SUITES

episodes, frames = 0, 0
for suite in SUITES:
    path = Path('datasets/LIBERO-fastwam') / suite / 'meta/episodes.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    count, length = len(rows), sum(row['length'] for row in rows)
    print(suite, count, 'episodes,', length, 'frames')
    episodes += count
    frames += length
assert (episodes, frames) == (1712, 277713), (episodes, frames)
PY
```

通过后进入第 7–8 节。队列会在新机固定训练/开发集清单、双卡提取 DINO/UMT5 特征并构建指令缓存；首次运行需计算全部 episode，中断后重跑会校验复用已完成的 episode。

主线使用 `dino_features_libero_sigreg_split42/`，其统计只用训练演示计算。发布基线的 `dino_features_libero_aggregate_minmax/` 使用不同统计范围，不能改名混入。即使缓存完整，也保留原始 suite 目录，队列还会检查元数据和原始路径。

## 7. W&B 登录和记录

```bash
python -c "import wandb; print(wandb.__version__)"
wandb login --relogin --verify
```

版本应为 `0.27.2`。浏览器打开 [W&B 授权页](https://wandb.ai/authorize)，登录后将 API key 粘贴到终端提示处；不写进 Git、本文或启动脚本。

默认项目 `miniwam-sigreg`，可修改 `runtime.env` 的 `WANDB_PROJECT`。使用团队空间时额外设置 `export WANDB_ENTITY='your_team_or_username'`。

第 8 节所有命令都带 `--wandb --wandb_project "$WANDB_PROJECT"`。每个训练阶段有独立 run 名称，例如 `smoke_A_stage1`、`pilot_C_0_stage1`、`formal_C_seed42_stage2`；在网页按 `group=libero-sigreg-seed42` 筛选。

每 10 步自动记录 loss、action loss、latent loss、KL、重建、SIGReg、学习率和配置，本地也保留 `metrics.jsonl`。**保持 `protocol/shared.yaml` 原样**：它由 `prepare()` 锁定，手改其中 `use_wandb: false` 为 true 会阻止队列启动。命令行 `--wandb` 能启用日志，无需改协议。

不能连接 W&B 时，在启动训练前改为离线模式，保留 `--wandb`：

```bash
export WANDB_MODE=offline
```

联网后同步已完成的离线 run：

```bash
for run in wandb/offline-run-*; do
  [ -d "$run" ] || continue
  WANDB_MODE=online wandb sync "$run"
done
```

本地 `wandb/` 保留到同步完成。重新 source `runtime.env` 会恢复其中写的 online 模式，离线启动时要在 source 之后设置 `WANDB_MODE=offline`。

## 8. 启动 SIGReg 四组实验

### 8.1 进入 tmux

```bash
tmux new -s miniwam
```

在 tmux 新会话中重新激活环境和变量；已有 Conda 路径不同时替换第一行，仓库位置不同时替换 cd：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate miniwam
cd /data/MiniWAM-release
source migration_artifacts/mirrors.env
source migration_artifacts/runtime.env
python scripts/verify_miniwam_environment.py \
  --require-cuda --check-libero --check-assets --visual-encoder dinov3
```

SSH 断开后任务仍运行。先按 `Ctrl+b` 再按 `d` 离开会话；`tmux attach -t miniwam` 重新连接。

### 8.2 使用新输出目录

`runtime.env` 已设置 `MINIWAM_RUN_DIR=outputs/libero_sigreg_seed42_newserver`。新机从空输出目录开始；旧实验目录保留在旧机，供查阅历史。

第 6 节下载原始数据和模型，第 8.3 节在新机生成特征缓存，再重新短跑和标定。不要将旧机的 `completed/` 标记放入新输出目录：标记包含解释器路径等命令指纹，换机后通常不同，会报 `Changed command for completed phase`。新机同一队列后续重启，会跳过已完成且命令相同的阶段。

### 8.3 从原始数据生成特征缓存

这一步会使用双卡实际计算全部特征，约 158GiB 的结果写入 `datasets/dino_features_libero_sigreg_split42/`。预处理时间需另计；中断后重复同一命令，会校验并复用已完成的 episode。

```bash
python scripts/experiments/run_libero_sigreg.py \
  --output_dir "$MINIWAM_RUN_DIR" --through preprocess \
  --wandb --wandb_project "$WANDB_PROJECT"
```

预期最终状态 `cache_ready`：1,712 条演示、277,713 帧；`protocol/` 有训练/开发集清单、共享配置和资产记录；缓存根目录有 `libero_instruction_features.pt`。

### 8.4 四组两阶段各 100 步短跑

```bash
python scripts/experiments/run_libero_sigreg.py \
  --output_dir "$MINIWAM_RUN_DIR" --through smoke \
  --wandb --wandb_project "$WANDB_PROJECT"
```

预期 `smoke_complete`，A/B/C/D 的 Stage 1、Stage 2 都有 `step_000100/`，W&B 出现 8 个训练 run。确认曲线和项目正确后启动长训练。

### 8.5 标定 λ，并完成正式训练

```bash
python scripts/experiments/run_libero_sigreg.py \
  --output_dir "$MINIWAM_RUN_DIR" --through training \
  --wandb --wandb_project "$WANDB_PROJECT"
```

队列跳过相同命令已完成的准备/短跑，依次执行：

1. 固定 64 样本标定 ID / SIGReg 编码器梯度，产生 3 个 λ 候选。
2. 每个候选训练 500 步，用同一批 256 个开发集样本及噪声 seed 101/102/103 比较 ID loss，筛除非有限和塌缩结果。
3. 写 `protocol/lambda_selection.json`、`selected_sigreg.yaml`。
4. A/B/C/D 各自从头训练 Stage 1 **16,260** 步、Stage 2 **54,200** 步；Stage 2 每 **5,420** 步保存。

希望标定后先查看结果时，先用 `--through calibration`，检查 JSON，再用同一条 training 命令。

| 组别 | Stage 1 | Stage 2 |
| --- | --- | --- |
| A | ID + 重建 + KL | 冻结本组编码器，fresh 初始化策略 |
| B | ID + 重建 | 同上 |
| C | ID + SIGReg，λ 来自标定 | 同上 |
| D | ID only | 同上 |

正式训练不从短跑/候选接着训练。C 配置中的 0.01 仅为短跑初值；正式 C 使用标定结果。四组保留相同的随机瓶颈、数据划分和归一化统计。

**时间预算：** 四组正式训练共 281,840 次优化器更新。旧机短跑约 3.2–3.4 秒/步，按同速外推仅正式训练约需 10–11 天，未含缓存、标定、诊断、评估；新机按实际 `elapsed_seconds / step` 重新估算。

## 9. 进度、恢复与评估

### 9.1 查看进度

另开终端：

```bash
cd /data/MiniWAM-release
source migration_artifacts/runtime.env
cat "$MINIWAM_RUN_DIR/status.json"
ls -lt "$MINIWAM_RUN_DIR/logs/"
nvidia-smi
df -h .
```

| 位置 | 内容 |
| --- | --- |
| `status.json` | 当前阶段、子进程 PID、日志路径；时间戳为 UTC |
| `logs/<阶段名>.log` | 该阶段详细控制台输出 |
| `runs/*/metrics.jsonl` | 每 10 步训练指标 |
| `runs/*/step_*/`、`latest` | 完整 checkpoint 与最新相对链接 |
| `diagnostics/*heldout.json` / `.csv` | 开发集 ID loss、mu 分布、有效秩 |
| `protocol/lambda_selection.json` | 候选和最终 λ |
| `software_snapshots/` | 代码哈希、关键依赖版本 |
| `wandb/` | 本地/离线 W&B 记录 |

例如 `tail -f "$MINIWAM_RUN_DIR/logs/formal_A_seed42_stage1.log"` 可实时查看。此处 Ctrl+c 只退出日志查看；在运行队列的终端按 Ctrl+c 会停止队列及其子进程。最终 `training_complete` 表示训练完成，还需 rollout 评估。

### 9.2 中断后恢复

同一新服务器、同一目录，确认旧队列已退出，重复 8.5 的**相同完整命令**。成功且命令相同的阶段跳过；提取复用完整 episode；失败训练从相同 seed 重跑该阶段，旧指标改名留档。队列不自动从半途 checkpoint 继续长跑。

训练器支持原命令加 `--resume <完整checkpoint>`，恢复模型、优化器、scheduler 和数据位置，但未保存完整训练/worker RNG，不能保证逐位复现。SIGReg 恢复还校验 GPU 数、每卡 batch 和累积口径。若手工恢复，保留原队列记录，用相同配置和训练总步数启动训练器，不能将其当成已被队列接管。

阶段重跑会产生新 W&B run。当前手工 resume 的 run ID 规则可能与首次自动 ID 不同，不保证接回原曲线，用明确名称/group 区分记录。

### 9.3 将开发集诊断汇总到 W&B

训练曲线自动上传；诊断脚本默认只输出 JSON/CSV。阶段完成后可执行：

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import wandb

table = wandb.Table(columns=['run', 'heldout_id_loss', 'main_effective_rank', 'wrist_effective_rank'])
for path in sorted((Path(os.environ['MINIWAM_RUN_DIR']) / 'diagnostics').glob('*heldout.json')):
    report = json.loads(path.read_text())
    table.add_data(path.stem, report['id_evaluation']['mean_id_loss'],
                   report['slots']['main_0']['effective_rank'], report['slots']['wrist_0']['effective_rank'])
with wandb.init(project=os.environ['WANDB_PROJECT'], name='heldout-summary', job_type='analysis') as run:
    run.log({'heldout/summary': table})
PY
```

有效秩反映表征方差分布；任务效果还要看开发集动作目标与 rollout 成功率。

### 9.4 完整训练后评估

以 A 为例，其他组替换名称并使用独立输出目录：

```bash
export MINIWAM_EVAL_CHECKPOINT="$MINIWAM_RUN_DIR/runs/formal_A_seed42_stage2/latest"
export MINIWAM_TEXT_CACHE='datasets/dino_features_libero_sigreg_split42/libero_instruction_features.pt'

python scripts/eval/eval_libero_miniwam.py \
  --checkpoint "$MINIWAM_EVAL_CHECKPOINT" --text_feature_cache "$MINIWAM_TEXT_CACHE" \
  --task_suite_name libero_10 --task_id 0 --num_trials 2 \
  --output_dir "$MINIWAM_RUN_DIR/eval/smoke_A"

python scripts/eval/run_libero_parallel_miniwam.py \
  --checkpoint "$MINIWAM_EVAL_CHECKPOINT" --text_feature_cache "$MINIWAM_TEXT_CACHE" \
  --task_suite_names libero_10,libero_spatial,libero_object,libero_goal \
  --num_trials 50 --gpu_ids 0,1 --output_dir "$MINIWAM_RUN_DIR/eval/formal_A"
```

两次单任务试跑用于检查链路。正式四组使用相同的四 suite × 10 任务 × 50 次协议；不使用正式 rollout 选 λ。训练 checkpoint 不自动附带发布模型的指令文件，因此显式提供缓存。LIBERO-Plus 另见 [评估文档](evaluation.md)。

## 10. 不同 GPU 数量与普通基线

### 10.1 当前队列固定两张卡

队列内部固定物理 GPU 0、1 和 `nproc_per_node=2`，提取也固定双卡分片；外部 `CUDA_VISIBLE_DEVICES` 不会覆盖内部选择。使用主线命令前确认这两张卡空闲。

硬件不同需统一调整所有组的布局，并同步更新锁定协议、提取分片和标定 batch。优先同时保持：

```text
优化器 global batch = GPU 数 × 每卡 batch × 累积 = 256
SIGReg 统计 batch   = GPU 数 × 每卡 batch = 64
```

| GPU 数 | 每卡 batch | 梯度累积 | global batch | SIGReg 统计 batch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 4 | 256 | 64 |
| 2 | 32 | 4 | 256 | 64 |
| 4 | 16 | 4 | 256 | 64 |
| 8 | 8 | 4 | 256 | 64 |

表中布局是否放入显存需短跑验证；不是直接给队列设置某个环境变量就能切换。若减小前向 batch、增加累积，即使 global batch 仍为 256，SIGReg 统计口径也已改变，应四组统一调整、重新标定 λ、使用新输出目录。

### 10.2 只跑一个普通发布配方基线

这条分支使用全部演示的归一化统计及 Stage 2 `policy_init: stage1`，与主线的开发集划分/fresh 初始化不同，结果应单独报告。基线另生成约 158GiB 特征，注意空间。

```bash
VISUAL_ENCODER=dinov3 bash scripts/data/extract_dataset_feature_libero.sh

torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/libero/stage1.yaml --fsdp \
  --batch_size 32 --gradient_accumulation_steps 4 \
  --wandb --wandb_project miniwam-baseline --wandb_run_name libero-stage1

torchrun --standalone --nproc_per_node=2 scripts/train.py \
  --config configs/libero/stage2.yaml --fsdp \
  --batch_size 32 --gradient_accumulation_steps 4 \
  --stage1_checkpoint outputs/libero_stage1/latest \
  --wandb --wandb_project miniwam-baseline --wandb_run_name libero-stage2
```

按顺序执行，Stage 1 成功保存后再启动 Stage 2。提取路径及训练参数详见 [数据](data.md)、[训练](training.md)。

## 附录 A：可选切换 Ubuntu APT 源

只用于 Ubuntu x86_64。保留系统发行版代号和组件，备份后替换常见官方地址：

```bash
cat /etc/os-release
uname -m
export MINIWAM_APT_STAMP="$(date +%Y%m%d-%H%M%S)"
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
  sudo cp -a /etc/apt/sources.list.d/ubuntu.sources "/etc/apt/sources.list.d/ubuntu.sources.bak.$MINIWAM_APT_STAMP"
  sudo sed -i -E \
    's@https?://([a-z]{2}\.)?(archive|security)\.ubuntu\.com/ubuntu/?@https://mirrors.ustc.edu.cn/ubuntu/@g' \
    /etc/apt/sources.list.d/ubuntu.sources
else
  sudo cp -a /etc/apt/sources.list "/etc/apt/sources.list.bak.$MINIWAM_APT_STAMP"
  sudo sed -i -E \
    's@https?://([a-z]{2}\.)?(archive|security)\.ubuntu\.com/ubuntu/?@https://mirrors.ustc.edu.cn/ubuntu/@g' \
    /etc/apt/sources.list
fi
sudo apt-get update
```

24.04 通常使用 `ubuntu.sources`，22.04 通常使用 `sources.list`。云厂商自定义 URL 不会被上述命令替换，可保留或参考 [Ubuntu 镜像帮助](https://mirrors.ustc.edu.cn/help/ubuntu.html)。执行后查看文件确认 URI。

前面的中科大测速结论不代表 APT 速度。可另外比较同一 Ubuntu 文件，重复两轮，只比较 HTTP 成功且完整下载的结果：

```bash
. /etc/os-release
for mirror in https://mirrors.ustc.edu.cn/ubuntu https://mirrors.tuna.tsinghua.edu.cn/ubuntu https://mirrors.aliyun.com/ubuntu; do
  curl -fL --max-time 30 -o /dev/null \
    -w "$mirror HTTP=%{http_code} bytes=%{size_download} speed=%{speed_download}B/s total=%{time_total}s\n" \
    "$mirror/dists/$VERSION_CODENAME/main/binary-amd64/Packages.xz"
done
```

## 附录 B：常见问题

| 现象 | 处理 |
| --- | --- |
| `cannot import name 'Imports'` | 激活 miniwam，执行 `python -m pip install 'wandb==0.27.2' 'protobuf==4.25.3'`，检查 import 和 pip check |
| CUDA 不可用 / no kernel image | 检查 GPU/驱动支持，确认 torch `+cu128`，运行第 5 节 CUDA 验证 |
| 交大 PyTorch 502 / 缺少 wheel | PyTorch 索引换官方 cu128，普通依赖保持国内源 |
| DINOv3 下载失败 | 核对 ModelScope 网络和第 6.1.1 节固定版本，重跑下载脚本；已校验文件会复用 |
| DINOv3 SHA256 不匹配 | 将报错的 `.incomplete` 文件改名后重跑；保留脚本内固定校验值 |
| UMT5 / 数据的 HF 401 / 403 | 使用官网端点登录 HF，确认所选端点支持对应仓库和 revision |
| 离线找不到 DINO/tokenizer | 核对 HF_HUB_CACHE；DINO 运行下载脚本的 --check-only，tokenizer 核对 refs/main；按第 6.1 节补齐 |
| locked experiment artifact | 恢复原协议或使用新输出目录；W&B 使用 CLI 开关 |
| Changed command for completed phase | 解释器/路径/参数变化，新机使用独立输出目录，从预处理开始生成本机记录 |
| stats_id / manifest hash 不匹配 | 使用本次队列生成的缓存、统计和划分；不要混用基线缓存或修改 metadata 绕过 |
| 队列进程锁占用 | 确认是否已经有同目录队列运行；锁文件存在本身不等于被占用 |
| CUDA OOM | 排查其他 GPU 任务；改 batch 时统一所有组、记录统计口径并重新标定 |
| W&B 无曲线 | 核对 --wandb、WANDB_MODE、项目/团队权限，以及是否仍在预处理 |
| W&B 网络超时 | offline 模式记录，联网后 sync；失败训练阶段按队列规则重跑 |
| 磁盘不足 | 查看 datasets/outputs 大小，扩容或转存完整 checkpoint；保留需要评估/恢复的文件 |

## 附录 C：安装完成后的最短启动命令

在 tmux 中，按实际路径执行：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate miniwam
cd /data/MiniWAM-release
source migration_artifacts/mirrors.env
source migration_artifacts/runtime.env
python scripts/experiments/run_libero_sigreg.py \
  --output_dir "$MINIWAM_RUN_DIR" --through training \
  --wandb --wandb_project "$WANDB_PROJECT"
```

## 参考

- 本仓库：[安装](install.md)、[数据](data.md)、[训练](training.md)、[实验队列](running_sigreg_experiments.md)、[消融设计](sigreg_ablation_plan.md)。
- 镜像：[清华 PyPI](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)、[清华 Conda](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)、[中科大 Conda](https://mirrors.ustc.edu.cn/help/anaconda.html)、[交大 PyTorch](https://mirror.sjtu.edu.cn/docs/pytorch-wheels)、[HF-Mirror](https://hf-mirror.com/)。
- DINOv3：[ModelScope 模型页](https://modelscope.cn/models/facebook/dinov3-vitb16-pretrain-lvd1689m)、[ModelScope SDK 下载接口](https://github.com/modelscope/modelscope/blob/master/modelscope/hub/file_download.py)。
- 发行方：[Miniforge](https://github.com/conda-forge/miniforge)、[NVIDIA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[W&B 0.27.2](https://github.com/wandb/wandb/tree/v0.27.2)。

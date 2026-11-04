# MiniWAM 新服务器迁移与训练指南（无 sudo、国内镜像）

更新：2026-11-04。主线对应当前 **LIBERO + DINOv3 + SIGReg A/B/C/D 消融**，训练 seed 42。新服务器**没有 sudo 权限，能访问国内镜像，不能直连 GitHub、HF 官方、PyTorch 官方及 W&B 等海外服务**。本文按这一条件安排安装和下载；按第 1–8 节执行即可开始训练。

依赖优先从国内镜像安装，**DINOv3 固定从 ModelScope（魔搭）下载**；UMT5 和 LIBERO 原始数据尝试 HF-Mirror。代码来自自己的 GitHub fork，通过可联网的电脑或登录节点制作源码 bundle 后传入；国内镜像缺失的安装包、无法直达的模型/数据由中转机**重新从网上下载**后按需传入，见第 3 节和附录 A。约 158GiB 特征缓存由新机生成，历史输出留在旧机。原始数据和模型下载量约 15–16GiB，环境安装包另计；无需整体复制旧服务器的数据、环境或实验目录。

默认仓库目录为 `$HOME/work/MiniWAM-release`，收件目录为 `$HOME/miniwam-migration`；也可改为自己有写权限且容量足够的数据盘。命令使用 Bash，标注“中转机”的步骤在能访问海外下载源的电脑/节点执行，其余在新服务器执行。适用于 Linux x86_64、Ubuntu 22.04/24.04；全程不修改系统软件源，不需要管理员权限。SSH、NVIDIA 内核驱动和 GPU 设备访问由服务器提供方配置。

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

**使用包含 SIGReg 实验的代码版本。** 本文与 SIGReg 实现、四组配置、实验队列一同提交。第 3 节从自己的 fork 的 `main` 制作源码 bundle、检查所需源码并记录实际 commit；旧的初始发布 commit `b639db68bb4be2c2113fc20c27b44c1e382f9dcd` 不包含这些实验代码。

新机优先用 580 系列或更新且支持目标 GPU 的驱动。CUDA 12.x 的最低兼容范围仍有功能限制，应以实际 CUDA 运算验证为准。[NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

## 2. 新服务器：在用户目录准备工具

### 2.1 检查资源和目录权限

```bash
export MINIWAM_ROOT="$HOME/work/MiniWAM-release"
export MINIWAM_INBOX="$HOME/miniwam-migration"
mkdir -p "$(dirname "$MINIWAM_ROOT")" "$MINIWAM_INBOX"

cat /etc/os-release
uname -m
nvidia-smi
df -h "$(dirname "$MINIWAM_ROOT")" /tmp
free -h
```

架构应为 `x86_64`，并能看到两张 GPU。若 `nvidia-smi` 不存在或设备不可访问，需由提供方配置驱动/设备授权；用户目录安装不能替代内核驱动。PyTorch CUDA wheel 自带 CUDA 用户态运行库，无需另装完整 CUDA Toolkit。

以下各段命令成功后再执行下一段；不要在下载、校验或安装失败后继续启动训练。

### 2.2 安装用户目录下的 Miniforge

已有可用 Conda 时加载其 `etc/profile.d/conda.sh`，跳到 2.3；其余情况：

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
```

已有同名安装目录时使用现有安装，不覆盖。若系统只有 `wget`，可用 `wget -O <文件名> <URL>` 下载；两者都没有或镜像不再保留此固定版本时，由中转机下载安装器及同版 SHA256 文件后传入，再在新机校验安装。[中科大说明](https://mirrors.ustc.edu.cn/help/github-release.html)指出该服务仅保留最新 release；不要在新机回退到无法访问的 GitHub。

### 2.3 从国内 Conda 源安装基础工具

先有 Python 和 Git，才能导入源码并测速。环境和包缓存均放在用户可写目录，即使使用提供方的只读 Conda 也如此：

```bash
export MINIWAM_CONDA_CHANNEL='https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/'
export CONDA_PKGS_DIRS="$HOME/.cache/miniwam/conda-pkgs"
cat > "$MINIWAM_INBOX/bootstrap-condarc" <<EOF
channels:
  - ${MINIWAM_CONDA_CHANNEL}
channel_priority: strict
show_channel_urls: true
EOF
export CONDARC="$MINIWAM_INBOX/bootstrap-condarc"
conda create -y -p "$HOME/.conda/envs/miniwam" --override-channels \
  -c "$MINIWAM_CONDA_CHANNEL" \
  python=3.10 pip 'setuptools>=77' wheel packaging ninja pyyaml \
  git curl tmux ca-certificates
conda activate "$HOME/.conda/envs/miniwam"
export PYTHONNOUSERSITE=1
conda env config vars set PYTHONNOUSERSITE=1
python --version
git --version
```

若该 Conda 镜像不可达，可将 channel 换成清华或北外的 conda-forge 国内地址后重试。已有该环境时先激活，并用 `conda install -p "$HOME/.conda/envs/miniwam" --override-channels -c "$MINIWAM_CONDA_CHANNEL" ...` 补齐上述包；不要删除已有环境。后面不执行通用的 `conda env create -f environment.yml` 或 `setup_miniwam_env.sh`，它们仍包含其他网络环境使用的源/克隆步骤。

## 3. 通过中转机获取自己的仓库和固定 LIBERO 源码

### 3.1 中转机：新下载源码并制作 bundle

可用自己的电脑、能访问 GitHub 的登录节点，或仍可联网的旧服务器作为中转机；本步骤只打包从 GitHub 新下载的源码。以下目录应为空，已存在时另选目录：

```bash
export MINIWAM_DELIVERY="$HOME/miniwam-delivery"
export MINIWAM_RELAY_WORK="$(mktemp -d /tmp/miniwam-source.XXXXXX)"
mkdir -p "$MINIWAM_DELIVERY"
git clone --branch main https://github.com/mylzju/MiniWAM-release.git \
  "$MINIWAM_RELAY_WORK/MiniWAM-release"
git -C "$MINIWAM_RELAY_WORK/MiniWAM-release" bundle create \
  "$MINIWAM_DELIVERY/MiniWAM-release.bundle" main
git -C "$MINIWAM_RELAY_WORK/MiniWAM-release" rev-parse HEAD \
  > "$MINIWAM_DELIVERY/git-head.txt"

git clone --no-checkout https://github.com/Lifelong-Robot-Learning/LIBERO.git \
  "$MINIWAM_RELAY_WORK/LIBERO"
git -C "$MINIWAM_RELAY_WORK/LIBERO" branch miniwam-pinned \
  8f1084e3132a39270c3a13ebe37270a43ece2a01
git -C "$MINIWAM_RELAY_WORK/LIBERO" bundle create \
  "$MINIWAM_DELIVERY/LIBERO.bundle" miniwam-pinned
(
  cd "$MINIWAM_DELIVERY"
  sha256sum MiniWAM-release.bundle LIBERO.bundle git-head.txt > SOURCE_SHA256SUMS
)
```

将 `MiniWAM-release.bundle`、`LIBERO.bundle`、`git-head.txt`、`SOURCE_SHA256SUMS` 传到新机 `$HOME/miniwam-migration/`。可使用平台文件上传、内网共享盘或 SSH/SFTP；例如在中转机执行以下命令，先替换账号/主机，非默认 SSH 端口加 `-P <端口>`：

```bash
scp "$MINIWAM_DELIVERY/MiniWAM-release.bundle" \
  "$MINIWAM_DELIVERY/LIBERO.bundle" "$MINIWAM_DELIVERY/git-head.txt" \
  "$MINIWAM_DELIVERY/SOURCE_SHA256SUMS" \
  user@new-server:~/miniwam-migration/
```

这是源码交付，保留 Git 历史和固定 revision，不包含旧机的模型、数据、环境和输出。新机无法访问 GitHub 时，后续代码更新也可经中转机重新制作 bundle；修改只提交到自己的 `mylzju/MiniWAM-release`。

### 3.2 新服务器：校验并克隆本地 bundle

```bash
(
  cd "$MINIWAM_INBOX"
  sha256sum -c SOURCE_SHA256SUMS
) && git clone --branch main "$MINIWAM_INBOX/MiniWAM-release.bundle" "$MINIWAM_ROOT"
cd "$MINIWAM_ROOT"
git remote set-url origin https://github.com/mylzju/MiniWAM-release.git
git remote add upstream https://github.com/J1dan/MiniWAM-release.git
git config remote.pushDefault origin
git config branch.main.pushRemote origin
mkdir -p migration_artifacts
git rev-parse HEAD | tee migration_artifacts/git-head.txt
cmp migration_artifacts/git-head.txt "$MINIWAM_INBOX/git-head.txt"
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

源码检查应输出 `SIGReg source files ready`。仓库目标目录应为空或不存在；已有源码时先确认版本和本地修改，不能直接覆盖。需要固定 MiniWAM 版本时，在中转机先切换对应 commit 并为其建分支，再制作该分支的完整 bundle，两端记录相同 SHA。新机的 `origin` 已指向自己的 fork；配置 remote 不会联网，`upstream` 仅供识别原始来源。

`migration_artifacts/` 只保存本机版本记录、测速结果和环境变量，已加入 `.gitignore`。数据、权重、特征缓存、实验输出和 Python 环境也由 `.gitignore` 排除，均按后文在新机下载或生成。

## 4. 比较国内源速度并选源

### 4.1 在新服务器实测

脚本只依赖 Python 标准库。先测试必需的 pip/Conda 国内入口：

```bash
mkdir -p migration_artifacts
python scripts/benchmark_download_mirrors.py \
  --groups pip conda --domestic-only --rounds 2 --mib 8 --timeout 20 \
  --output migration_artifacts/mirrors.json \
  --write-env migration_artifacts/mirrors.env

source migration_artifacts/mirrors.env
cat migration_artifacts/mirrors.env
```

| 类别 | 候选 | 测试文件 |
| --- | --- | --- |
| pip | 清华、阿里云、中科大、交大 | 同一个 numpy 1.26.4 / Python 3.10 / Linux wheel |
| PyTorch cu128 | 交大 | torch 2.11.0+cu128 / Python 3.10 / Linux wheel |
| conda-forge | 清华、中科大、北外 | `linux-64/repodata.json.zst` |
| HF | HF-Mirror | 固定 revision 的 LIBERO Goal 压缩包 |

每次读取前 8MiB，顺序下载，第二轮反转候选顺序；按两轮都成功时的**索引解析加下载的中位耗时**选择。检查二进制文件头，避免把镜像验证页当作高速文件。四类国内候选共 9 个、两轮最多读取 144MiB，另有索引流量。

`--domestic-only` 会在请求前排除官方入口，国内候选全失败时返回非零，**不会回退到官方源**。`--prefer-domestic` 仍会测试官方并可能回退，不适用于这台新机。JSON 记录最终下载的 `final_host`；国内入口可能重定向到海外 CDN，此选项不能让海外 CDN 变为国内服务。

再分别测试容易缺包或重定向的 PyTorch 和 HF。失败时保留记录，走附录 A 的对应中转分支；不要因此改装其他 torch/CUDA 版本：

```bash
unset MINIWAM_TORCH_INDEX MINIWAM_HF_ENDPOINT
if python scripts/benchmark_download_mirrors.py \
  --groups torch --domestic-only --rounds 2 --mib 8 --timeout 20 \
  --output migration_artifacts/torch-mirrors.json \
  --write-env migration_artifacts/torch-mirrors.env; then
  cat migration_artifacts/torch-mirrors.env >> migration_artifacts/mirrors.env
else
  echo 'cu128 国内入口不可用：按附录 A.1 准备 wheelhouse。'
fi
if python scripts/benchmark_download_mirrors.py \
  --groups hf --domestic-only --rounds 2 --mib 8 --timeout 20 \
  --output migration_artifacts/hf-mirrors.json \
  --write-env migration_artifacts/hf-mirrors.env; then
  cat migration_artifacts/hf-mirrors.env >> migration_artifacts/mirrors.env
else
  echo 'HF 镜像文件不可达：UMT5/数据按附录 A.2 中转。'
fi
source migration_artifacts/mirrors.env
```

主文件生成 `MINIWAM_PIP_INDEX`、`MINIWAM_CONDA_CHANNEL`；可用时再追加 `MINIWAM_TORCH_INDEX`、`MINIWAM_HF_ENDPOINT`。单组测速的 `--write-env` 会覆盖目标文件，所以必须使用不同文件名。HF 的一次成功只验证 LIBERO Goal 样本，不能保证 tokenizer、UMT5 及其他 suite 全部可达；它们在第 6 节分别下载并验证。DINOv3 独立使用 ModelScope。

### 4.2 旧服务器的实测结果

2026-11-04 北京时间 10:21 开始，未设置显式代理环境变量；两轮、每轮 8MiB、下载时限 15 秒。下表为下载速度中位数，单位 MiB/s；选择源的总耗时还包含索引解析。[完整测量 JSON](assets/mirror_benchmark_20261104.json)

| 类别 | 清华 | 中科大 | 阿里云 | 交大 / 北外 / HF-Mirror | 官方 |
| --- | ---: | ---: | --- | --- | ---: |
| pip | 4.07 | **25.00** | 两轮超时 | 交大 3.55 | 2.24 |
| conda-forge | 4.67 | **10.29** | — | 北外 4.57 | 2.10 |
| PyTorch cu128 | — | — | — | 交大两轮 HTTP 502 | **2.08** |
| HF 文件 | — | — | — | HF-Mirror 1.23 | **1.43** |

这组历史结果允许连接海外服务，**不能作为新机可联网的证据**。其中 HF-Mirror 最终主机也是 `us.aws.cdn.hf.co`，交大 cu128 两轮均失败；因此必须保留中转分支。8MiB 采样不保证代表 11GB 大文件的持续速度，也不涵盖完整依赖求解时间。

本次修订另在当前服务器做了国内入口单轮 1MiB 复核：pip/Conda 均可选中科大；交大 cu128 已恢复，最终主机为 `s3.jcloud.sjtu.edu.cn`；HF-Mirror 仍转到 `us.aws.cdn.hf.co`。[本次复核 JSON](assets/mirror_benchmark_domestic_20261104.json)。这只是当前服务器的小样本检查，新机仍按 4.1 实测，不能沿用可达性结论。

若只是测速失败但已单独确认国内源可用，可手动保存国内候选：

```bash
export MINIWAM_PIP_INDEX='https://mirrors.ustc.edu.cn/pypi/simple/'
export MINIWAM_CONDA_CHANNEL='https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/'
# 以下两项仅在对应真实文件测试通过后设置；否则按附录 A 中转。
# export MINIWAM_TORCH_INDEX='https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/'
# export MINIWAM_HF_ENDPOINT='https://hf-mirror.com'
```

后面默认加载 `mirrors.env`。若手动选源后需要保持选择，将相应 export 行也写入该文件。

## 5. 创建固定版本的运行环境

### 5.1 安装用户态工具、编译器和图形库

沿用第 2 节的用户环境，不向系统目录安装软件：

```bash
cd "$MINIWAM_ROOT"
source migration_artifacts/mirrors.env
export CONDA_PKGS_DIRS="$HOME/.cache/miniwam/conda-pkgs"
cat > migration_artifacts/condarc <<EOF
channels:
  - ${MINIWAM_CONDA_CHANNEL}
channel_priority: strict
show_channel_urls: true
EOF
export CONDARC="$MINIWAM_ROOT/migration_artifacts/condarc"
conda install -y -p "$HOME/.conda/envs/miniwam" --override-channels \
  -c "$MINIWAM_CONDA_CHANNEL" \
  ffmpeg cmake make pkg-config c-compiler cxx-compiler \
  libegl libgl libglib xorg-libsm xorg-libxext xorg-libxrender
conda activate "$HOME/.conda/envs/miniwam"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Conda 的 Git/tmux/FFmpeg/编译器替代原流程中的系统安装；`libegl`、`libgl`、`libglib` 等提供用户态动态库。[libegl 包说明](https://anaconda.org/conda-forge/libegl/)、[编译器说明](https://anaconda.org/conda-forge/cxx-compiler/)。这些包不能替代 NVIDIA 驱动的 EGL 实现：后续 rollout 渲染还要求提供方开放 GPU 图形驱动库，第 9.4 节会实际验证。

### 5.2 从仓库提取完整版本约束

不直接执行 `environment.yml` 内置的源配置；只提取包版本，并加入固定 W&B：

```bash
python - <<'PY'
from pathlib import Path
import yaml

environment = yaml.safe_load(Path('environment.yml').read_text())
requirements = next(item['pip'] for item in environment['dependencies']
                    if isinstance(item, dict) and 'pip' in item)
requirements = [item for item in requirements if not item.startswith('--')]
requirements.append('wandb==0.27.2')
Path('migration_artifacts/requirements-all.txt').write_text('\n'.join(requirements) + '\n')
PY
```

### 5.3 安装依赖：两种方式二选一

**方式一：国内镜像完整可用。** 仅在第 4 节 cu128 测试成功时执行；仍缺包、重定向不可达或编译失败时，改用方式二，不改版本约束：

```bash
(
  set -e
  : "${MINIWAM_TORCH_INDEX:?cu128 国内入口不可用，请使用下面的 wheelhouse 方式}"
  python -m pip --isolated install \
    --index-url "$MINIWAM_PIP_INDEX" \
    --extra-index-url "$MINIWAM_TORCH_INDEX" \
    -r migration_artifacts/requirements-all.txt
)
```

**方式二：中转机新下载的 wheelhouse。** 按附录 A.1 准备并传入 `$MINIWAM_INBOX/wheelhouse/`；包含 CUDA wheel、全部依赖和预先构建好的 wheel，不复制任何旧 Python 环境。在新机完全离线安装：

```bash
(
  set -e
  cd "$MINIWAM_INBOX/wheelhouse"
  sha256sum -c SHA256SUMS
  cmp requirements-all.txt "$MINIWAM_ROOT/migration_artifacts/requirements-all.txt"
  cd "$MINIWAM_ROOT"
  python -m pip --isolated install --no-index \
    --find-links "$MINIWAM_INBOX/wheelhouse" \
    -r migration_artifacts/requirements-all.txt
)
```

这里的 `--no-index` 禁止 pip 请求在线索引。wheelhouse 必须由与新机相同的 Linux x86_64/发行版和 Python 3.10 构建，不能用 macOS、Windows 或 Python 3.11 的 wheel 替代。

### 5.4 安装当前代码并验证 CUDA

上述任一方式成功后，执行共同步骤。依赖已安装，editable 安装禁止再次解析依赖和联网拉取构建工具：

```bash
python -m pip --isolated install --no-index --no-deps --no-build-isolation -e '.[logging]'
python -m pip check

python - <<'PY'
import ctypes
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
for name in ('libEGL.so.1', 'libGL.so.1', 'libglib-2.0.so.0'):
    ctypes.CDLL(name)
    print('User-space library ready:', name)
PY
python -m pip freeze > migration_artifacts/pip-freeze.txt
conda list --explicit > migration_artifacts/conda-explicit.txt
```

`pip check` 应无冲突。**W&B 固定 0.27.2，protobuf 固定 4.25.3。** 旧机 W&B 0.31.0 不支持当前 protobuf，导致 `cannot import name 'Imports'`。保持项目版本约束。[W&B 0.27.2 依赖声明](https://github.com/wandb/wandb/blob/v0.27.2/pyproject.toml)

### 5.5 从本地 bundle 安装固定 LIBERO

```bash
git clone --branch miniwam-pinned "$MINIWAM_INBOX/LIBERO.bundle" LIBERO
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

已有干净、相同 commit 的 LIBERO checkout 时跳过 clone/checkout，但重新生成路径配置。不要沿用旧机 `.libero/config.yaml`。校验器应能读取 `.git` 并确认固定 revision，不使用丢失版本信息的源码压缩包替代。

### 5.6 保存后续会话要加载的变量

```bash
cat > migration_artifacts/runtime.env <<EOF
export MINIWAM_ROOT="$MINIWAM_ROOT"
export MINIWAM_INBOX="$MINIWAM_INBOX"
export MINIWAM_RUN_DIR="outputs/libero_sigreg_seed42_newserver"
export PYTHONNOUSERSITE=1
export CONDA_PKGS_DIRS="$HOME/.cache/miniwam/conda-pkgs"
export CONDARC="$MINIWAM_ROOT/migration_artifacts/condarc"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HUB_CACHE="$MINIWAM_ROOT/pretrained/hf-cache"
export HF_ENDPOINT="${MINIWAM_HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LIBERO_CONFIG_PATH="$MINIWAM_ROOT/.libero"
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export NUMBA_CACHE_DIR="$HOME/.cache/miniwam/numba"
export MPLCONFIGDIR="$HOME/.cache/miniwam/matplotlib"
export WANDB_PROJECT=miniwam-sigreg
export WANDB_RUN_GROUP=libero-sigreg-seed42
export WANDB_MODE=offline
EOF
source migration_artifacts/runtime.env
```

训练默认离线加载模型并离线记录 W&B，重新 source 仍保持 offline。第 6.1.2 和 6.3 节的下载子 shell 临时取消模型离线开关，退出后恢复；ModelScope 下载不受这些开关影响。`HF_HUB_DISABLE_XET=1` 禁用 Xet 下载插件，但不能阻止镜像 HTTP 重定向至海外 CDN。[HF 环境变量说明](https://huggingface.co/docs/huggingface_hub/v0.36.0/package_reference/environment_variables)

## 6. 在新机下载模型和原始数据

| 内容 | 网上来源 | 新机处理 |
| --- | --- | --- |
| DINOv3 ViT-B/16 | [ModelScope：facebook/dinov3-vitb16-pretrain-lvd1689m](https://modelscope.cn/models/facebook/dinov3-vitb16-pretrain-lvd1689m) | 固定版本下载、逐文件校验，写入本地离线缓存 |
| UMT5 tokenizer | HF-Mirror 的 `google/umt5-xxl`；不可达时中转 | 只下载 tokenizer 文件到 Hub cache |
| UMT5 文本编码器 | HF-Mirror 的 `Wan-AI/Wan2.1-T2V-14B`；不可达时中转 | 只下载约 11GB 的指定单文件到 `pretrained/language/` |
| LIBERO 原始数据 | HF-Mirror 的 `yuanty/LIBERO-fastwam`；不可达时中转 | 下载固定 revision 的四个 LeRobot 2.1 压缩包并校验、解压 |
| DINO/UMT5 特征缓存 | 由上述原始数据和模型计算 | 第 8.3 节在新机生成约 158GiB 缓存 |

DINOv3 使用 ModelScope 下载脚本。UMT5 和原始数据仅在第 4 节 HF 文件测试成功时尝试直接下载；失败则按附录 A.2 获取新下载的文件，再执行第 6.2 节和 6.3 节的校验/解压步骤。不在新机切回 HF 官方端点或执行官方登录。下载中断后重跑，可复用已验证文件和未完成的下载。[HF CLI 说明](https://huggingface.co/docs/huggingface_hub/v0.36.0/guides/cli)

### 6.1 下载模型权重和 tokenizer

#### 6.1.1 DINOv3：从 ModelScope 下载

```bash
export HF_HUB_CACHE="$MINIWAM_ROOT/pretrained/hf-cache"
bash scripts/download_dinov3_modelscope.sh
```

脚本使用 PATH 中的 `curl` 和 `sha256sum`，固定 ModelScope commit `23d0280ae6ee4ced592a3459674ad027d3c18906`，下载模型、配置、预处理配置和许可/说明文件。五个文件全部通过 SHA256 校验后，完成本地缓存登记；权重文件为 342,662,192 字节。

这些文件与现有实验固定的 DINOv3 版本逐字节一致，因此本地缓存仍使用配置中的 revision `5931719e67bbdb9737e363e781fb0c67687896bc`。`pretrained/hf-cache/` 是 Transformers 的**本地兼容缓存目录**；DINOv3 的网络下载请求只发往 ModelScope，无需 HF 模型访问申请或 HF 登录。训练、预检和评估按原有模型标识离线读取即可。

#### 6.1.2 UMT5：下载 tokenizer 和文本编码器

这两个仓库为公开资产，通常无需登录。镜像真实文件可达时：

```bash
(
  set -e
  : "${MINIWAM_HF_ENDPOINT:?HF 镜像不可用，请按附录 A.2 中转}"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  export HF_HUB_CACHE="$MINIWAM_ROOT/pretrained/hf-cache"
  export HF_ENDPOINT="$MINIWAM_HF_ENDPOINT"
  export HF_HUB_DISABLE_XET=1
  export HF_HUB_DOWNLOAD_TIMEOUT=600
  export HF_HUB_ETAG_TIMEOUT=60
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

已经默认禁用 Xet。若仍报 CDN 连接失败、401/403、缺少固定 revision，按附录 A.2 在中转机重新下载这两个资产，只传入所需文件；不要让新机反复尝试海外域名。DINOv3 始终由 ModelScope 下载，独立于这些设置。[镜像端点说明](https://hf-mirror.com/)

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

镜像文件可达时下载；已按附录 A.2 放好四个压缩包时跳过这一块：

```bash
(
  set -e
  : "${MINIWAM_HF_ENDPOINT:?HF 镜像不可用，请按附录 A.2 中转}"
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  export HF_ENDPOINT="$MINIWAM_HF_ENDPOINT"
  export HF_HUB_DISABLE_XET=1
  export HF_HUB_DOWNLOAD_TIMEOUT=600
  export HF_HUB_ETAG_TIMEOUT=60
  mkdir -p datasets/LIBERO-fastwam
  for suite in libero_10 libero_spatial libero_object libero_goal; do
    hf download yuanty/LIBERO-fastwam "${suite}_no_noops_lerobot.tar.gz" \
      --repo-type dataset \
      --revision ee018b997c430bb12b5bf3c892d744798c5a2f91 \
      --local-dir datasets/LIBERO-fastwam
  done
)
```

两种获取方式都必须在新机校验全部四个包，再解压：

```bash
(
  set -e
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

## 7. W&B 默认离线记录，经中转机同步

```bash
python -c "import wandb; print(wandb.__version__)"
source migration_artifacts/runtime.env
test "$WANDB_MODE" = offline
```

版本应为 `0.27.2`。**新机不需要 W&B 登录或 API key**。`WANDB_MODE=offline` 将记录写入仓库的 `wandb/offline-run-*`，训练期间网页不会更新。

默认项目 `miniwam-sigreg`，可修改 `runtime.env` 的 `WANDB_PROJECT`。使用团队空间时额外设置 `export WANDB_ENTITY='your_team_or_username'`。

第 8 节保留 `--wandb --wandb_project "$WANDB_PROJECT"`，启用本地 W&B 记录。每个训练阶段有独立 run 名称，例如 `smoke_A_stage1`、`pilot_C_0_stage1`、`formal_C_seed42_stage2`；同步后可在网页按 `group=libero-sigreg-seed42` 筛选。

每 10 步自动记录 loss、action loss、latent loss、KL、重建、SIGReg、学习率和配置，本地也保留 `metrics.jsonl`。**保持 `protocol/shared.yaml` 原样**：它由 `prepare()` 锁定，手改其中 `use_wandb: false` 为 true 会阻止队列启动。命令行 `--wandb` 能启用日志，无需改协议。

全部训练/汇总 run 结束后，在**新服务器**打包离线记录；若只同步部分 run，只列出已结束的目录，不打包正在写入的 run：

```bash
tar -czf "$MINIWAM_INBOX/wandb-offline.tar.gz" wandb/offline-run-*
```

通过平台/内网/SFTP 将这个日志包取到**能访问 W&B 的中转机**。在装有相同 `wandb==0.27.2` 的环境中执行；替换日志包路径：

```bash
mkdir -p "$HOME/miniwam-wandb-sync"
cd "$HOME/miniwam-wandb-sync"
tar -xzf /path/to/wandb-offline.tar.gz
wandb login --relogin --verify
for run in wandb/offline-run-*; do
  [ -d "$run" ] || continue
  WANDB_MODE=online wandb sync "$run"
done
```

仅中转机需要浏览器打开 [W&B 授权页](https://wandb.ai/authorize)并在终端输入 key；不要写进 Git 或启动脚本。新机本地 `wandb/` 保留到确认同步成功。不要在新机设置 `WANDB_MODE=online` 或循环重试登录；本地进度使用第 9.1 节查看。[W&B 离线说明](https://docs.wandb.ai/models/track/offline)

## 8. 启动 SIGReg 四组实验

### 8.1 进入 tmux

```bash
tmux new -s miniwam
```

在 tmux 新会话中重新激活环境和变量；已有 Conda 路径不同时替换第一行，仓库位置不同时替换 cd：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/miniwam"
cd "$HOME/work/MiniWAM-release"
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

预期 `smoke_complete`，A/B/C/D 的 Stage 1、Stage 2 都有 `step_000100/`，本地 W&B 生成 8 个离线训练 run。检查阶段日志和 `metrics.jsonl` 后启动长训练；需要网页检查时，先同步这 8 个已结束的 run。

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
cd "$HOME/work/MiniWAM-release"
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

训练曲线默认保存在离线 run，诊断脚本默认只输出 JSON/CSV。阶段完成后可执行以下命令，将诊断表也写入一个离线 run，再按第 7 节一并同步：

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

先在新机验证无需显示器的 EGL 渲染：

```bash
python - <<'PY'
import mujoco

context = mujoco.GLContext(64, 64)
context.make_current()
model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=64, width=64)
renderer.update_scene(data)
image = renderer.render()
assert image.shape == (64, 64, 3), image.shape
renderer.close()
context.free()
print('Headless EGL rendering ready')
PY
```

若用户态库能加载但报 `EGL_NOT_INITIALIZED`、无 EGL 设备或缺 `libEGL_nvidia.so.0`，请提供方开放/挂载与宿主驱动匹配的 NVIDIA 图形库及 EGL vendor 配置；仅 CUDA 可用不代表渲染可用。不要尝试用用户目录安装内核驱动。渲染通过后再执行 rollout。

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

## 附录 A：国内镜像缺包时的中转下载

本附录只在镜像无法提供所需文件时使用。中转机必须能访问对应官方源，并能通过 SSH/SFTP、平台上传或共享盘将文件交给新机；不能靠修改镜像环境变量跨过被阻断的 CDN。所有资产都重新从网上获取，特征缓存和训练输出仍在新机生成。

### A.1 中转机制作完整 wheelhouse

使用**与新机相同 Linux x86_64/发行版、Python 3.10** 的中转机；不要求 GPU。加载中转机已有 Conda，沿用第 3.1 节的 `MINIWAM_DELIVERY` 和 `MINIWAM_RELAY_WORK`（新终端先恢复为当时的绝对路径）。新建专用构建环境；已有同名环境时使用另一个路径：

```bash
conda create -y -p "$HOME/.conda/envs/miniwam-wheelbuilder" --override-channels \
  -c https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/ \
  python=3.10 pip 'setuptools>=77' wheel pyyaml \
  cmake make ninja pkg-config c-compiler cxx-compiler
conda activate "$HOME/.conda/envs/miniwam-wheelbuilder"
export PYTHONNOUSERSITE=1
cd "$MINIWAM_RELAY_WORK/MiniWAM-release"
test "$(git rev-parse HEAD)" = "$(cat "$MINIWAM_DELIVERY/git-head.txt")"
mkdir -p "$MINIWAM_DELIVERY/wheelhouse"
python - <<'PY'
import os
from pathlib import Path
import yaml

environment = yaml.safe_load(Path('environment.yml').read_text())
requirements = next(item['pip'] for item in environment['dependencies']
                    if isinstance(item, dict) and 'pip' in item)
requirements = [item for item in requirements if not item.startswith('--')]
requirements.append('wandb==0.27.2')
path = Path(os.environ['MINIWAM_DELIVERY']) / 'wheelhouse/requirements-all.txt'
path.write_text('\n'.join(requirements) + '\n')
PY
python -m pip --isolated wheel \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu128/ \
  -r "$MINIWAM_DELIVERY/wheelhouse/requirements-all.txt" \
  --wheel-dir "$MINIWAM_DELIVERY/wheelhouse"
(
  cd "$MINIWAM_DELIVERY/wheelhouse"
  sha256sum ./*.whl requirements-all.txt > SHA256SUMS
)
scp -r "$MINIWAM_DELIVERY/wheelhouse" user@new-server:~/miniwam-migration/
```

先确认 `pip wheel` 完整成功，再生成校验清单并传输。它会下载运行依赖并构建只有源码分发的包，避免新机离线安装时临时拉取构建依赖；新机安装项目本身时由第 2 节的 `setuptools>=77` 提供 editable 构建。CUDA 运行依赖体积较大，wheelhouse 除 torch/torchvision 外还必须包含 NVIDIA 等传递依赖，不能只取两个 wheel。返回第 5.3 节方式二安装。

### A.2 中转机重新下载 UMT5 和原始数据

仅在 HF-Mirror 的对应文件不可达时使用。此处官方端点**只在中转机**使用；DINOv3 仍由新机从 ModelScope 下载。以下示例下载 UMT5 和四个原始数据包全套，缺少部分资产时可只保留对应下载命令/打包路径。

中转机使用 Python 3.10，沿用第 3.1 节的 `MINIWAM_DELIVERY`：

```bash
python -m pip --isolated install --index-url https://pypi.org/simple \
  'huggingface-hub==0.36.2'
export MINIWAM_RELAY_ASSETS="$MINIWAM_DELIVERY/assets"
mkdir -p "$MINIWAM_RELAY_ASSETS/pretrained/language" \
  "$MINIWAM_RELAY_ASSETS/datasets/LIBERO-fastwam" \
  "$MINIWAM_RELAY_ASSETS/migration_artifacts"
(
  set -e
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  export HF_ENDPOINT=https://huggingface.co
  export HF_HUB_CACHE="$MINIWAM_RELAY_ASSETS/pretrained/hf-cache"
  export HF_HUB_DISABLE_XET=1
  export HF_HUB_DOWNLOAD_TIMEOUT=600
  export HF_HUB_ETAG_TIMEOUT=60
  hf download google/umt5-xxl \
    --revision 66cb9e7e85526fe440a945569e42c72fb6cbc0ad \
    --include '*token*' '*.model'
  hf download Wan-AI/Wan2.1-T2V-14B models_t5_umt5-xxl-enc-bf16.pth \
    --revision b62153b54fce61f5908de23f6d8715e147de0daf \
    --local-dir "$MINIWAM_RELAY_ASSETS/pretrained/language"
  for suite in libero_10 libero_spatial libero_object libero_goal; do
    hf download yuanty/LIBERO-fastwam "${suite}_no_noops_lerobot.tar.gz" \
      --repo-type dataset --revision ee018b997c430bb12b5bf3c892d744798c5a2f91 \
      --local-dir "$MINIWAM_RELAY_ASSETS/datasets/LIBERO-fastwam"
  done
  python - <<'PY'
import os
from pathlib import Path

ref = Path(os.environ['HF_HUB_CACHE']) / 'models--google--umt5-xxl/refs/main'
ref.parent.mkdir(parents=True, exist_ok=True)
ref.write_text('66cb9e7e85526fe440a945569e42c72fb6cbc0ad')
PY
)
```

下载全部成功后，计算传输校验清单并打包；保留 tokenizer cache 的相对符号链接及其 `blobs/`，不要只复制 `snapshots/`：

```bash
python - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ['MINIWAM_RELAY_ASSETS'])
rows = []
for directory in ('pretrained', 'datasets'):
    for path in sorted((root / directory).rglob('*')):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        rows.append(f'{digest.hexdigest()}  {path.relative_to(root).as_posix()}\n')
assert rows, 'No downloaded assets to package'
(root / 'migration_artifacts/assets-SHA256SUMS').write_text(''.join(rows))
PY
tar -cf "$MINIWAM_DELIVERY/assets.tar" -C "$MINIWAM_RELAY_ASSETS" \
  pretrained datasets migration_artifacts/assets-SHA256SUMS
scp "$MINIWAM_DELIVERY/assets.tar" user@new-server:~/miniwam-migration/
```

打包目录应为本次下载专用，避免混入旧缓存和未完成的 `.incomplete`。全套下载约 15–16GiB，打包还需一份归档空间，CUDA wheelhouse 另计。

**新服务器：** 从仓库根目录导入并验证，无需联网：

```bash
(
  set -e
  cd "$MINIWAM_ROOT"
  tar -xf "$MINIWAM_INBOX/assets.tar"
  sha256sum -c migration_artifacts/assets-SHA256SUMS
)
```

然后返回第 6.2 节核对模型固定 SHA256 和离线加载，并执行第 6.3 节的**校验/解压块**及 6.4 的数据数量检查；不再执行 HF 下载块。确认所有检查通过后，传输归档可移出训练数据盘，保留实际模型、原始数据和完整 tokenizer cache。

## 附录 B：常见问题

| 现象 | 处理 |
| --- | --- |
| `cannot import name 'Imports'` | 按第 5.3 节从国内源或 wheelhouse 重装 `wandb==0.27.2`、`protobuf==4.25.3`，检查 import 和 pip check |
| CUDA 不可用 / no kernel image | 检查 GPU/驱动支持，确认 torch `+cu128`，运行第 5 节 CUDA 验证 |
| 没有 sudo / 无权写 Conda 目录 | 使用第 2 节用户前缀 `$HOME/.conda/envs/miniwam` 和用户包缓存；不使用 APT |
| GitHub 连接超时 | 从自己的 fork 在中转机制作 bundle，按第 3 节导入，保留 `.git` |
| 交大 PyTorch 502 / 缺少 wheel | 按附录 A.1 新下载完整 cu128 wheelhouse；不在新机回退官方源或更换版本 |
| DINOv3 下载失败 | 核对 ModelScope 网络和第 6.1.1 节固定版本，重跑下载脚本；已校验文件会复用 |
| DINOv3 SHA256 不匹配 | 将报错的 `.incomplete` 文件改名后重跑；保留脚本内固定校验值 |
| HF-Mirror 首页可开但下载失败 / HF 401、403 | 查看测速 `final_host`，镜像可能转到海外 CDN；按附录 A.2 中转所缺资产 |
| `libGL.so.1` / `libEGL.so.1` / GLib 缺失 | 按第 5.1 节安装 Conda 用户态库，并加载 runtime.env 中的动态库路径 |
| CUDA 可用但 EGL 渲染失败 | 按第 9.4 节实测，提供方需开放 NVIDIA 图形库、EGL vendor 配置和设备 |
| 离线找不到 DINO/tokenizer | 核对 HF_HUB_CACHE；DINO 运行下载脚本的 --check-only，tokenizer 核对 refs/main；按第 6.1 节补齐 |
| locked experiment artifact | 恢复原协议或使用新输出目录；W&B 使用 CLI 开关 |
| Changed command for completed phase | 解释器/路径/参数变化，新机使用独立输出目录，从预处理开始生成本机记录 |
| stats_id / manifest hash 不匹配 | 使用本次队列生成的缓存、统计和划分；不要混用基线缓存或修改 metadata 绕过 |
| 队列进程锁占用 | 确认是否已经有同目录队列运行；锁文件存在本身不等于被占用 |
| CUDA OOM | 排查其他 GPU 任务；改 batch 时统一所有组、记录统计口径并重新标定 |
| W&B 网页无曲线 | 离线模式属预期；查看本地 `metrics.jsonl` / `wandb/offline-run-*`，再由中转机同步 |
| W&B 网络超时 | 检查 runtime.env 为 offline，保留 --wandb；仅中转机登录/sync |
| 磁盘不足 | 查看 datasets/outputs 大小，扩容或转存完整 checkpoint；保留需要评估/恢复的文件 |

## 附录 C：安装完成后的最短启动命令

在 tmux 中，按实际路径执行：

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/miniwam"
cd "$HOME/work/MiniWAM-release"
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
- 发行方：[Miniforge](https://github.com/conda-forge/miniforge)、[Miniforge 国内镜像说明](https://mirrors.ustc.edu.cn/help/github-release.html)、[NVIDIA 兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)、[W&B 离线模式](https://docs.wandb.ai/models/track/offline)。

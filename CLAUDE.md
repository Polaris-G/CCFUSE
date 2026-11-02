# CCSFuse 复现项目

## 项目目标
复现 CCSFuse（Zhang et al., IEEE TGRS 2026, Art. no. 5400314，UAV RGB-IR 目标检测：CMFC + AFSF，基于 YOLOv5 双流）。
旧交接记载当时未找到官方代码，本次未重新核实其公开状态。详细方案见仓库根目录的 CCSFuse_复现方案.md，过程记录见 WORKLOG.md。
当前 Windows 本地项目为 `D:\project\CCFUSE`；旧记录中的 `D:\CCSFuse` 是历史代码/数据目录，不要混用。服务器项目为 `~/bzp/CCFUSE`。
原本地仓库 `D:\Projects\CCFUSE` 于本次会话中无法找到，已按用户要求从 GitHub 恢复到 `D:\project\CCFUSE`。
新会话先读根目录 [CCFUSE_新会话交接_2026-11-02.md](CCFUSE_新会话交接_2026-11-02.md)，再读本文件“当前进度”和 WORKLOG.md 最新记录，并核对实际 Git 状态。交接区分本地核验、用户服务器日志、历史记录与推测。历史命令保留作追溯，不构成新授权；论文原文未在本次本地接手中重新核对。

## 工作约定

- **每次完成项目任务后、最终回复前，都要同步更新 `CLAUDE.md` 和 `WORKLOG.md`**（用户于 2026-11-02 明确要求）。即使只交付服务器命令或解读用户日志，也要记录阶段进展。
- `CLAUDE.md` 维护当前约定、有效路径、配置、任务状态和下一步；`WORKLOG.md` 按日期追加实际改动、命令、检查结果、提交/上传状态及待办。保留历史记录，及时替换过期的“当前/下一步”说明。
- 明确区分“本地已执行”“用户服务器日志已确认”“仅提供命令”和“待确认”；没有启动或完成日志，不把指令交付写成训练已开始或已完成。
- 工作流程为 **Windows 本地修改与检查 → GitHub 上传 → 用户同步服务器 → 用户训练/评估**。服务器上的全部操作由用户执行，Codex 只提供命令/代码并分析输出，不直接 SSH 操作服务器。
- 用中文回复。
- 执行命令前先说明要做什么。
- 每完成一步先停下来，告诉用户如何验证，再继续下一步。
- 方案中所有【推断】项（论文未给出的设置）都要做成 yaml 或命令行可配置，并在代码注释中标明"论文未给出，推断"。
- 删除操作只删确认为空或确认无用的内容，删除前先核实；用户的原始数据只读。
- **长时间训练由用户在服务器 tmux 中运行**。给出完整命令、开跑前的检查项、续训命令，等用户贴回输出。本地按已授权范围执行配置核验及短前向/反向检查；服务器上的短检查和评估也由用户执行。
- 开始长时间训练前先让用户运行 `git status --short`，确认没有意外改动（见"踩过的坑"）。

## 环境（Windows 笔记本，历史记录）
下面的 `D:\` 路径只属于这台笔记本，服务器上不要用。
- GPU：NVIDIA RTX 3070 Ti Laptop，8 GB，算力 8.6；驱动 616.64（支持 CUDA 13.4）。
  - 桌面程序常驻占用 1–2 GB 显存，实际可分配约 7.25 GB。
  - "CUDA 系统内存回退"已关闭，显存不足时会正常报 OOM。
- conda：D:\Miniconda，**不在 PATH 中**。
- 环境：ccsfuse，路径 D:\Miniconda\envs\ccsfuse。
  - 激活：`D:\Miniconda\Scripts\activate` 后执行 `conda activate ccsfuse`。
  - 或直接调用 `D:\Miniconda\envs\ccsfuse\python.exe`（推荐，不受 VS Code 所选解释器影响）。
- 版本：Python 3.10.22，torch 2.5.1+cu124，torchvision 0.20.1+cu124，cuDNN 9.1，numpy 1.23.5。
  - 其余依赖的版本固定在 D:\CCSFuse\requirements-ccsfuse.txt。
- 环境自检：在 D:\CCSFuse 下运行 `python check_env.py`，应输出 ALL CHECKS PASSED。
- pip 缓存目录：D:\pip-cache（在 C:\Users\bzp02\AppData\Roaming\pip\pip.ini 中设置）。

## 服务器（Linux）
实验室机器，后续训练放在这里。用户 baozhipeng。没有 Slurm，长训练用 tmux。能直连 pypi.org。

- 系统：Ubuntu 24.04，20 核 CPU，62 GB 内存。
- GPU：1 张 RTX 5060 Ti 16 GB（Blackwell，sm_120）。驱动 595.99（支持到 CUDA 13.2），系统 nvcc 12.8。
- conda base：`/opt/miniforge3`；用户于 2026-11-02 回传 `conda info --envs` 确认当前 `ccsfuse` 环境实际位于 `/home/baozhipeng/.conda/envs/ccsfuse`。Python 3.10 的路径也见本次检查日志；依赖版本不要凭安装脚本重新推定或随意升级。
- 目录：
  - 项目：`~/bzp/CCFUSE`（`/home/baozhipeng/bzp/CCFUSE`）
  - 数据集：`~/bzp/datasets`，例如 `~/bzp/datasets/DroneVehicle/{rgb,ir}/images/{train,val,test}`、`~/bzp/datasets/LLVIP/...`
  - 权重：项目里的 `weights/`（`yolov5m.pt`、`yolov5s.pt`）
- data yaml 用相对仓库根目录的路径（`../datasets/...`）。命令都在仓库根目录执行，这样这些相对路径才指到 `~/bzp/datasets`。
- 安装（仓库根目录）：`bash setup_server.sh`
  - 用 `/opt/miniforge3` 建 Python 3.10 的 `ccsfuse` 环境。
  - 从 `https://download.pytorch.org/whl/cu128` 安装 torch 2.7.1、torchvision 0.22.1（cp310 的 wheel 已在该索引上；cu128 上还有更新的 2.8–2.11，这里钉第一版带 sm_120 的稳定组合，少改 YOLOv5 v5.0 的代码）。2.5.1+cu124 没有 sm_120。
  - 再装 `requirements-ccsfuse.txt`，并把 numpy 重新钉在 1.23.5（代码用了 `np.int`）。
- 激活：`source /opt/miniforge3/etc/profile.d/conda.sh && conda activate ccsfuse`
- 自检：`python check_env.py`，应输出 `ALL CHECKS PASSED`。显存下限是当前这张卡总显存的 70%（8 GB 卡约 5.6 GB，16 GB 卡约 11.2 GB），并检查当前算力是否在 `torch.cuda.get_arch_list()` 里。
- `import cv2` 若报缺 `libGL.so.1`：`sudo apt-get install -y libgl1`。

`train.py` 的默认 `--batch-size 4`、`--workers 2`、`--nbs 12` 仍是 8 GB 笔记本的设置。完整版显存大约 `0.35 + 0.775 × batch` GiB（allocated），batch 12 约 9.7 GiB，超过笔记本可分配的约 7.25 GB，所以默认 batch 保持 4。epoch、学习率、SGD 这些论文默认值没有改。服务器上要显式加上设备、batch 和 workers。

`nbs` / `accumulate`（只看实际在用的 `train_rgb_ir`）：

- `accumulate = max(round(nbs / batch_size), 1)`，有效 batch = `batch_size × accumulate`。
- `weight_decay` 再乘 `batch_size * accumulate / nbs`。这个系数为 1 时，衰减和名义 batch 一致。
- 笔记本：batch 4、nbs 12 → accumulate 3，有效 batch 12，weight_decay 系数 1。已有结果是这个口径。BN 每步看到的是 4 张图。
- 服务器：`--batch-size 12 --nbs 12` → accumulate 1，有效 batch 12，weight_decay 系数 1，和论文总 batch（2 卡 × 6）以及笔记本的有效 batch 同一口径。BN 每步看到的是 12 张图。
- 论文是 2 卡 DDP、每卡 batch 6，未写 SyncBN。两卡时每卡 BN 统计基于 batch 6；这里单卡 batch 12，BN 统计基于 12。
- `--batch-size 16 --nbs 16` 的有效 batch 是 16，和论文总 batch 12 不是同一口径。Table I 四组都用 12 / 12。

服务器推荐命令（有效 batch 12，weight_decay 系数 1）：

```bash
python train.py --cfg models/ccsfuse/<配置>.yaml --name <名字> \
    --batch-size 12 --workers 8 --device 0 --nbs 12
```

历史完整版 tmux 启动示例（不是当前 CMFC-only v2 指令；本轮指令见 WORKLOG.md 的 2026-11-02 记录）：

```bash
tmux new -s ccsfuse
source /opt/miniforge3/etc/profile.d/conda.sh && conda activate ccsfuse
cd ~/bzp/CCFUSE
git status --short
python train.py --cfg models/ccsfuse/yolov5m_ccsfuse.yaml --name ccsfuse_m_zinit \
    --batch-size 12 --workers 8 --device 0 --nbs 12
```

LLVIP（Adam，lr 0.0001，其余超参同 DroneVehicle；必须同时写 `--hyp` 和 `--adam`）：

```bash
python train.py --cfg models/ccsfuse/yolov5m_ccsfuse.yaml --name ccsfuse_m_llvip \
    --data data/multispectral/LLVIP_ccsfuse.yaml \
    --hyp data/hyp.ccsfuse_llvip.yaml --adam \
    --batch-size 12 --workers 8 --device 0 --nbs 12
```

脱离会话：Ctrl-b 然后 d。回来：`tmux attach -t ccsfuse`。续训：`python train.py --resume runs/train/<名字>/weights/last.pt`（在同一套 batch / nbs 下恢复）。

## 代码库
- 路径：当前 Windows 本地仓库为 `D:\project\CCFUSE`，服务器为 `~/bzp/CCFUSE`；`D:\CCSFuse` 为历史目录。本项目基于 CFT 官方实现修改。
  - 来源：https://github.com/DocF/multispectral-object-detection
  - commit：fb591c9b163177c0e950db08e213e24ddc912d41
- 基于 YOLOv5 v5.0 时期的代码：
  - 双流训练入口：train.py 中的 `train_rgb_ir`（train.py 里另有一个不用的单流 `train`，两者有大段重复代码，修改时注意改的是 `train_rgb_ir`）
  - 双流模型：models\yolo_test.py（train.py 导入的是它，不是 models\yolo.py）；yaml 中 `from: -4` 表示该层读取 IR 输入
  - 数据集类：utils\datasets.py 中的 `LoadMultiModalImagesAndLabels`
  - CFT 原有模型配置：models\transformer\*.yaml；CCSFuse 配置放在 models\ccsfuse\
- 预训练权重（ultralytics v5.0 release，COCO）：weights\yolov5s.pt（调试用）、weights\yolov5m.pt（正式训练用）。
- v5.0 主干结构：第 0 层 Focus，第 8 层 SPP；P3 / P4 / P5 来自第 4 / 6 / 9 层（均为 C3）；通道数 m 版 192 / 384 / 768，s 版 128 / 256 / 512。

### git
- GitHub 仓库：`https://github.com/Polaris-G/CCFUSE`。当前分支 `codex/cmfc-only-v2-ablation`；本轮从 `1e6e4de` 继续实现逐目标诊断。开始前本地与远端同哈希，main 为 `27b8214`；没有合并 main。本轮保留并纳入此前 CLAUDE.md、WORKLOG.md 的交接/纠错改动和新交接文档。本轮工具和文档已完成本地检查，准备提交/推送；服务器逐目标结果尚未回传。实际 HEAD 以 `git log -1` 为准。
- `server-linux` 是历史迁移分支记录。`.gitignore` 忽略 `__pycache__/`、`datasets/`、`weights/*.pt`、`runs/`、`*.cache`；不要提交原始数据、训练产物或权重。
- 未跟踪、不要提交：`.vscode/`（VS Code 设置）、论文 PDF。
- 提交使用实际本地 Git 配置。2026-11-02 已通过 `git var GIT_AUTHOR_IDENT` / `GIT_COMMITTER_IDENT` 确认身份并完成提交，不沿用旧文档里的临时姓名/邮箱覆盖配置。

| 提交 | 内容 |
|---|---|
| cba1f45 | 会话 1：复现方案、环境检查、数据转换工具、数据配置 |
| 87d523e | 第 1 步：双流 YOLOv5m + Add 基线，两路主干都加载 COCO 权重 |
| 5f70fc2 | 第 2 步：数据管线（成对随机裁剪、hsv_ir、check_aug.py） |
| bfa75fd | 第 3 步：基线结果记录（test 67.3）、续训测试 |
| 909c743 | 第 4 步：CMFC 模块及单元测试 |
| 76ba902 | 第 5 步：AFSF 模块及单元测试 |
| f13ec37 | 第 6 步：CMFC/AFSF 接入网络，完整版及两组消融的 yaml |
| 0b0a7c7 | CMFC `proj_init: zero`；记录完整版第一次结果（test 66.1）及诊断 |

### 改动过的 CFT 原有文件
- **train.py**（只改了 `train_rgb_ir` 和命令行参数）：
  - COCO 权重用 `load_coco_dual_stream` 加载，两路主干都能拿到（原来的 intersect_dicts 只覆盖 RGB 主干）；本地没有权重文件时直接报错，不调用 attempt_download。
  - 新增 `--nbs`、`--hyp-set KEY=VALUE`、`--pretrain-neck`；nesterov 和 warmup 最少步数改由 hyp 中的 `nesterov`、`warmup_min_iters` 控制；`--hyp-set` 还允许 `hsv_ir`；默认值即本机配置；hyp 和 data yaml 按 UTF-8 读取。
  - 优化器分组：LayerNorm 的 weight 不衰减（推断）；pg0/pg1/pg2 都没收集到的参数（CMFC 的 λ）补进不衰减组，日志打印 `Optimizer: N extra parameter tensors`。
- **models\yolo_test.py**：模型 yaml 按 UTF-8 读取；parse_model 注册 CMFC / CMFCInject / AFSF / PairAdd，模块构造支持 kwargs，从 yaml 顶层的 `cmfc:` / `cmfc_inject:` / `afsf:` 读取。
- **utils\datasets.py**：`LoadMultiModalImagesAndLabels` 的非 Mosaic 分支恢复成对的 `random_perspective_rgb_ir`；hyp `hsv_ir: none|independent|same`（缺省为 independent，即 CFT 原行为）；`augment_hsv` 增加 `gains=` 参数并返回增益。
- **test.py**：画图时只传 RGB 三通道（原来传 6 通道会在后台线程报错）；data yaml 按 UTF-8 读取。
- **utils\loss.py** 第 211 行：`clamp_(0, gain[3] - 1)` 改为 `int(gain[3]) - 1`（torch ≥1.12 下原写法报类型错误）。
- **utils\general.py** 的 `check_requirements`：导入不到 pkg_resources 时跳过检查（setuptools 84 已移除 pkg_resources）。

### 新增文件
- 模型与模块：
  - models\ccsfuse_modules.py：`CMFC`、`AFSF`、`CMFCInject`（回注方式 A/B）、`PairAdd`（把一对特征相加，用于 +CMFC 消融）。
  - utils\ccsfuse.py：`load_coco_dual_stream()`，按 yaml 中的 `pretrain_map` 把 COCO 主干权重复制到两路；`--pretrain-neck` 时也复制 neck。
  - models\ccsfuse\yolov5m_dual_add.yaml（基线，Table I 第 1 行；调试用 yolov5s_dual_add.yaml）
  - models\ccsfuse\yolov5m_ccsfuse.yaml（完整版，第 4 行）、yolov5m_cmfc_add.yaml（+CMFC，第 2 行）、yolov5m_afsf.yaml（+AFSF，第 3 行）
- 配置：data\hyp.ccsfuse_dronevehicle.yaml（DroneVehicle 超参，推断项已标注）；data\hyp.ccsfuse_llvip.yaml（LLVIP：Adam 用的 lr0=0.0001，其余同 DroneVehicle，训练时加 `--adam`）；data\multispectral\ 下的 DroneVehicle.yaml、DroneVehicle_test.yaml、LLVIP_ccsfuse.yaml、DroneVehicle_1pct.yaml（及其 txt 列表）。
- 工具（tools\）：prepare_dronevehicle.py、prepare_llvip.py、viz_pairs.py、check_loader.py、check_weights.py、check_model.py、make_subset.py、check_aug.py、test_cmfc.py、test_afsf.py；根目录 check_env.py、requirements-ccsfuse.txt。

## CCSFuse 模块的实现要点（推断项都在 yaml 中配置）
- **CMFC**（式 1–6）：每个模态各一套 Q/K/V；K/V 经 SR（Conv k=r, s=r + LayerNorm，PVT 写法）；输出 = (w_SIC·Softmax(s) + w_RIS·ReLU(s)²)·V，其中 s = QKᵀ/√d，(w_SIC, w_RIS) = softmax(λ1, λ2)，λ 初始化为 1；最后加残差。
  - yaml `cmfc:` 中的推断项：`num_heads: 8`、`out_proj: True`、`share_lambda: True`、`ris_norm: False`、`attn_fp32: True`、`branch_weights: null`（Table II 用 [1,0] / [0,1] / [0.5,0.5]）、旧版 `proj_init: zero`，本轮 v2 使用 **`proj_init: small`、`proj_std: 1.0e-3`**；每层的 sr_ratio 写在层参数里（stage 3/4/5 为 8/4/2）。
  - `cmfc_inject: A`（下一 stage 的输入 = f^CMFC）；B 为 stage 输出 + f^CMFC。
- **AFSF**（式 7–10）：每个模态 GAP → 1×1 卷积 → Sigmoid；拼接后经 MLP（2C→2C/4→2C，GELU），再按通道切成两半；两路各自乘权重后拼接，用 Conv(1×1)+BN+SiLU 压到 C。yaml `afsf:` 中为 `reduction: 4`、`sigmoid_after_mlp: False`。
- 完整版 yaml 的主干按 stage 交错：RGB s3、IR s3、CMFC_3、两路 Inject、RGB s4 ……；AFSF 直接取 CMFC 输出的一对特征。pretrain_map 为 rgb [0-4,13,14,20,21,22]、ir [5-9,15,16,23,24,25]。
- 规模（m 版，640×640）：

  | 配置 | 参数量 | GFLOPs（flop_counter） | 训练速度 | 显存（reserved） |
  |---|---|---|---|---|
  | 基线 Add | 33.43 M | 83.7 | 约 7.8 h / 45 epoch | 3.4 G |
  | +AFSF | 38.08 M | 86.5 | 估计 8–8.5 h | — |
  | +CMFC | 53.79 M | 94.5 | 估计约 12 h | — |
  | 完整版 | 58.45 M | 97.3 | 约 12–14 h（3.57 it/s 起） | 4.3–5.3 G |

  - **与论文的 33.50 M / 174.12 GFLOPs 对不上**，任何一种合理配置都无法同时吻合，因此不能用这两个数字反推结构。用户决定保持 m 版和方案默认的 CMFC，按**相对提升**评判。
  - 显存与 batch 的关系（完整版，AMP）：峰值 allocated 约为 0.35 + 0.775 × batch GiB；论文设置（每卡 6）约 6–7.5 GB/卡，单卡 batch 12 约 11–13.5 GB。

## CFT 加载器的约定
- 标签路径由图片路径推出：先把 `/` 和 `\` 都折成当前系统的分隔符，再把第一段 `images` 换成 `labels`，扩展名改为 `.txt`。列表用正斜杠或反斜杠都可以。
- RGB 和 IR 各自按文件名排序，然后按位置配对；两边的文件名必须完全一致。
- **训练 loss 只用 RGB 一侧的标签（labels_rgb）。**
- 每个样本是 6 通道张量（RGB + IR）。
- 加载器会在 labels 目录旁生成 `<split>.cache`，数据变动后会自动重建。
- `test.py` 只读取数据 yaml 中的 `val_rgb` / `val_ir`。
- 增强（第 2 步后）：不开 Mosaic；成对的随机平移和缩放（translate 0.1，scale 0.5）和水平翻转，两路几何变换完全一致；HSV 只作用于 RGB（`hsv_ir: none`）。

## 数据

Windows 上转换后的数据在 `D:\CCSFuse\datasets\`。服务器上同一份内容放在 `~/bzp/datasets/`（不要带 `D:\`）。原始数据仍只读；训练读的是转换后的 YOLO 目录。

### 原始数据（只读）
- D:\CCSFuse\datasets\DroneVehicle_raw\{train,val,test}\{split}{img,imgr,label,labelr}
  - img 为 RGB，imgr 为 IR，label / labelr 为对应的 xml 标注。
  - 原图 840×712。
- D:\CCSFuse\datasets\LLVIP_raw\LLVIP\{visible,infrared}\{train,test}\*.jpg
- D:\CCSFuse\datasets\LLVIP_raw\LLVIP\Annotations\*.xml
- D:\CCSFuse\datasets\LLVIP.zip：原始压缩包，用户尚未决定是否删除。

### 转换后的数据（YOLO 格式，cls xc yc w h 归一化）
- D:\CCSFuse\datasets\DroneVehicle\{rgb,ir}\{images,labels}\{train,val,test}\
- D:\CCSFuse\datasets\LLVIP\{rgb,ir}\{images,labels}\{train,test}\
- 转换统计：每个数据集目录下的 convert_stats.json。

### DroneVehicle 转换规则（tools\prepare_dronevehicle.py）
- 四边各裁掉 100 px 白边，840×712 变为 640×512，JPEG 质量 95。
- 框的处理：
  - polygon 旋转框：取四个顶点的 min/max 得到水平框；
  - bndbox：直接使用；
  - 所有框平移 −100 px 后裁剪到图像范围内，裁剪后宽或高 <2 px 的丢弃；
  - point 类型的标注丢弃。
- **使用 IR 标注（labelr），同一份标签同时写到 rgb\labels 和 ir\labels。**
- 类别 id（论文 Table I 顺序）：car=0，truck=1，freight car=2，bus=3，van=4。
- 原始类名的映射：`feright car` / `feright_car` / `feright` → freight car；`truvk` → truck；`*` → 丢弃。
- 运行命令：
  `D:\Miniconda\envs\ccsfuse\python.exe -I D:\CCSFuse\tools\prepare_dronevehicle.py --src D:\CCSFuse\datasets\DroneVehicle_raw --dst D:\CCSFuse\datasets\DroneVehicle`
  - 可选参数：`--label-source ir|rgb`、`--min-size`、`--quality`、`--limit`。
  - 如果 dst 已存在且非空，脚本会拒绝运行。
- 数量（对数 / 目标数）：

  | 划分 | 对数 | 目标数 |
  |---|---|---|
  | train | 17990 | 316,409 |
  | val | 1469 | 24,490 |
  | test | 8980 | 159,613 |

  - 共丢弃 44 个（point 38，`*` 1，过小 5）；空标签文件 53 个。
  - 类别极不均衡：test 集中 car 137,145 个，van 只有 4,282 个。

### LLVIP 转换规则（tools\prepare_llvip.py）
- 图像原样复制，不缩放（1280×1024，由加载器缩放到 640）。
- 两侧共用同一份标注；只有一个类，person=0。
- 宽或高 <1 px 的框丢弃（原始数据中有 5 个宽度为 0 的框）。
- 运行命令：
  `D:\Miniconda\envs\ccsfuse\python.exe -I D:\CCSFuse\tools\prepare_llvip.py --src D:\CCSFuse\datasets\LLVIP_raw\LLVIP --dst D:\CCSFuse\datasets\LLVIP`
- 数量：train 12025 对 / 34,130 个目标；test 3463 对 / 8,302 个目标；空标签文件 2 个。

### 数据配置文件
- 训练（用 val 验证）：data\multispectral\DroneVehicle.yaml
- 最终评估（val 键指向 test）：data\multispectral\DroneVehicle_test.yaml
- LLVIP（val 键指向 test）：data\multispectral\LLVIP_ccsfuse.yaml
- 过拟合检查（train 1% 子集，180 对，train 和 val 都指向它）：data\multispectral\DroneVehicle_1pct.yaml

## 常用命令
Windows 本地代码检查在 `D:\project\CCFUSE` 下运行，解释器为已核实的 `D:\Miniconda\envs\ccsfuse\python.exe`。当前数据 YAML 面向服务器的相邻 datasets 目录，本地检查应明确使用合成数据或另行核实的数据路径。
服务器：在 `~/bzp/CCFUSE` 下运行，先 `conda activate ccsfuse`，训练命令加上 `--batch-size 12 --workers 8 --device 0 --nbs 12`（见「服务器（Linux）」）。
- 训练（默认即基线和笔记本配置：yolov5m_dual_add.yaml、weights/yolov5m.pt、DroneVehicle.yaml、hyp.ccsfuse_dronevehicle.yaml、45 epoch、batch 4、nbs 12、workers 2，AMP 自动开启）：
  `python train.py --cfg models/ccsfuse/<配置>.yaml --name <名字>`
- 续训：`python train.py --resume runs/train/<名字>/weights/last.pt`（已测试：epoch、hyp-set、优化器、EMA 都能恢复）
- 历史 test 集评估命令（约 16–21 分钟；不作为本轮独立 val 的口径）：
  `python test.py --weights runs/train/<名字>/weights/best.pt --data data/multispectral/DroneVehicle_test.yaml --batch-size 8 --iou-thres 0.6 --task test --verbose --name <名字>_test`
  - 上述历史命令使用 NMS IoU 0.6。本轮与已有独立 val 对照时，明确使用 `--task val --conf-thres 0.001 --iou-thres 0.5`；训练内验证的默认 NMS IoU 0.6 与独立评估不同。
- 1% 子集过拟合检查（约 10 分钟）：
  `python train.py --cfg <配置> --data data/multispectral/DroneVehicle_1pct.yaml --epochs 30 --hyp-set lr0=0.01 warmup_epochs=1 warmup_min_iters=45 --name overfit_<名字>`
  - 服务器上同一条命令再加 `--batch-size 12 --workers 8 --device 0 --nbs 12`，放进 tmux。
- 检查工具（正斜杠在 Windows 和 Linux 上都能用）：
  - `python tools/check_model.py --cfg <配置> [--batch-size N]`：权重迁移（应为 rgb 306/306、ir 306/306）、融合层和 Detect 的形状、loss、显存、参数量、GFLOPs（thop 和 flop_counter 两种）、CMFC 分支权重。
  - `python tools/test_cmfc.py`、`python tools/test_afsf.py`：单元测试，应输出 ALL ... TESTS PASSED。
  - `python tools/check_aug.py --n 10 --out runs/viz/aug_pairs.jpg`：增强可视化和三项数值检查。
  - `python tools/check_loader.py --data <yaml>`、`python tools/viz_pairs.py --root <数据目录> --split train --n 10 --out <文件.jpg>`、`python tools/check_weights.py`、`python tools/make_subset.py --frac 0.01 --out data/multispectral/<名字>.yaml`。

## 超参数（data\hyp.ccsfuse_dronevehicle.yaml，可用 `--hyp-set` 覆盖）
- 论文给出：SGD、lr0 0.001、momentum 0.9、45 epoch、batch 12（2 卡 × 6）。
- 推断：lrf 0.01（YOLOv5 余弦退火，视作 One-Cycle）、nesterov False、weight_decay 5e-4、warmup 3 epoch 且最少 1000 步、warmup_bias_lr 0.01、mosaic 0、translate 0.1、scale 0.5、fliplr 0.5、HSV 用 YOLOv5 默认值、hsv_ir none。
- 本机：batch 4 + 梯度累积 3 步（nbs 12），weight_decay 缩放系数为 1。
- 服务器：`--batch-size 12 --nbs 12`，accumulate 1，系数 1。
- neck 和 head 不加载 COCO 权重（论文只说主干用 COCO 预训练），可用 `--pretrain-neck` 打开。
- LLVIP：`data/hyp.ccsfuse_llvip.yaml`，`lr0: 0.0001`，训练加 `--adam`（beta1 = momentum 0.9，beta2 0.999 为推断）。其余键与 DroneVehicle 相同。

## 论文训练设置对照

论文原文：Ubuntu 20.04、Python 3.8、PyTorch 2.0.1、两张 RTX 4090D；输入缩到 640×640；主干用 COCO 预训练；45 epoch；每卡 batch 6；DroneVehicle 用 SGD、momentum 0.9、初始学习率 0.001、One-Cycle；增强为水平翻转、随机裁剪、颜色变换。LLVIP 相同，优化器改为 Adam、学习率 0.0001。

| 项 | 论文 | 代码当前值 | 对照 |
|---|---|---|---|
| 系统 / Python / PyTorch | Ubuntu 20.04，Python 3.8，PyTorch 2.0.1 | 服务器 Ubuntu 24.04，Python 3.10，torch 2.7.1+cu128。笔记本 Python 3.10.22，torch 2.5.1+cu124 | 环境差异，不改。5060 Ti（sm_120）要 cu128 的 torch，2.0.1 没有这个算力 |
| GPU 与总 batch | 2×4090D，每卡 6，总 batch 12 | 服务器 `--batch-size 12 --nbs 12`（accumulate=1，有效 batch 12，weight_decay 系数 1）。`train.py` 默认 `--batch-size 4 --nbs 12`（accumulate 3，有效 batch 12） | 总 batch 与论文一致。默认 batch 保持 4：完整版 batch 12 约 9.7 GiB allocated，笔记本约 7.25 GB 可用，会 OOM |
| BN | 两卡 DDP，未写 SyncBN | 单卡 batch 12 | 论文两卡时每卡 BN 基于 batch 6；单卡 batch 12 的 BN 基于 12。笔记本已有结果的 BN 基于 batch 4 |
| 输入 | 640×640 | `--img-size 640 640`。letterbox：长边缩到 640，短边填充（DroneVehicle 裁边后 640×512，上下各填约 64 px） | 张量是 640×640，一致。填充而不是拉伸：论文未给出，推断 |
| 主干初始化 | COCO 预训练 | `weights/yolov5m.pt`，`load_coco_dual_stream` 复制到两路主干。neck 默认不复制 | 主干一致。neck 不加载：论文未给出，推断（`--pretrain-neck`） |
| Epoch | 45 | `--epochs 45` | 一致 |
| DroneVehicle 优化器 | SGD，momentum 0.9 | SGD，`momentum: 0.9`，`nesterov: False` | momentum 一致。nesterov：论文未给出，推断为 False |
| 初始学习率 | 0.001 | `lr0: 0.001` | 一致 |
| 学习率调度 | One-Cycle | 默认 `one_cycle(1, lrf, epochs)`：余弦乘子从 1 到 `lrf=0.01`，再乘 lr0，用 LambdaLR。`--linear-lr` 默认关闭。另有线性 warmup | 默认是 YOLOv5 的余弦 one-cycle，线性开关是关的。`lrf` 和 warmup 长度：论文未给出，推断。调度类是 LambdaLR，未使用会先升后降的 `torch.optim.lr_scheduler.OneCycleLR` |
| Warmup | 未单独写 | `warmup_epochs: 3`，`warmup_min_iters: 1000`，`warmup_momentum: 0.8`，`warmup_bias_lr: 0.01`。SGD momentum 在这段从 0.8 插到 0.9 | 论文未给出，推断 |
| weight_decay | 未写 | `0.0005`，再乘 `batch*accumulate/nbs`。服务器上系数为 1 | 论文未给出，推断 |
| 水平翻转 | 有 | `fliplr: 0.5`，两路同一次随机 | 有翻转，一致。概率 0.5：论文未给出，推断 |
| 随机裁剪 | 有 | `translate: 0.1`、`scale: 0.5`，`random_perspective_rgb_ir` 两路共用一个矩阵（Mosaic 关闭时） | 用平移+缩放实现。幅度：论文未给出，推断 |
| 颜色变换 | 有 | `hsv_h/s/v = 0.015/0.7/0.4`，`hsv_ir: none`（只动 RGB） | 有颜色变换，一致。强度和「只动 RGB」：论文未给出，推断 |
| Mosaic、mixup、shear、perspective、flipud、旋转 | 未写 | hyp 里全是 0。双流 `__getitem__` 的 mixup 代码已注释，把 `mixup` 调大也不会执行 | 已关闭 |
| 损失系数 | 未写（沿用 YOLOv5 损失） | box 0.05、cls 0.5、obj 1.0、iou_t 0.20、anchor_t 4.0 等 | 论文未给出，推断 |
| AMP、workers | 未写 | CUDA 上 AMP 默认开。workers 默认 2；服务器命令写 8 | 论文未给出，推断 |
| LLVIP | Adam，lr 0.0001，其余相同 | `data/hyp.ccsfuse_llvip.yaml` 的 `lr0: 0.0001`，其余键与 DroneVehicle 相同。命令加 `--adam`。`betas=(momentum, 0.999)`，beta1=0.9。warmup 不改 Adam 的 beta | 优化器和 lr 一致。beta2=0.999、eps、Adam 的 weight_decay：论文未给出，推断 |

## 历史实验结果（早期 Windows 实验，DroneVehicle test 集，mAP50 %）

| 实验 | runs\train\ | mAP50 | car | truck | freight car | bus | van | 相对基线 | 论文 |
|---|---|---|---|---|---|---|---|---|---|
| 基线 Add | baseline_add_m | **67.3** | 97.7 | 68.8 | 46.6 | 94.3 | 28.8 | – | 72.8 |
| 完整版（proj_init default） | ccsfuse_m | **66.1** | 97.6 | 68.9 | 42.3 | 94.4 | 27.4 | **−1.2** | 79.4（+6.6） |
| 完整版（proj_init zero） | ccsfuse_m_zinit | 待训练 | | | | | | | 79.4（+6.6） |
| +CMFC | cmfc_add_m | 未训练 | | | | | | | 76.8（+4.0） |
| +AFSF | afsf_m | 未训练 | | | | | | | 78.7（+5.9） |

- 基线比论文低 5.5 点，差距全部来自 van（51% 被判成 car）和 freight car（42% 被判成 truck）。用户决定接受 67.3 作为自己的基线，按相对提升评判（方案第 6 节）。
- 完整版第一次（ccsfuse_m）：val 曲线前 16 个 epoch 与基线持平，从 epoch 20 起落后，最后 val mAP50 0.646 对 0.670；训练 loss 一直略低于基线。λ 几乎没动（w_SIC/w_RIS 为 0.555/0.445、0.509/0.491、0.484/0.516）。
  - 诊断：初始化时 CMFC 的补偿量 ‖Δ‖/‖f‖ 就达到 0.50–0.66，经回注方式 A 直接破坏了 COCO 预训练的 stage4/5；ReLU² 分支的幅值大于 Softmax 分支。
  - 改法（用户选的方案 1）：`proj_init: zero`，输出投影零初始化，CMFC 初始为恒等映射（实测初始 ‖Δ‖/‖f‖ = 0.00；1% 过拟合正常）。备选方案 2：`ris_norm: True`。

## 当前进度（2026-11-02）

- 已收到 Add、CMFC-only v2、完整版 v2 三组服务器离线诊断，均为1469张 val 图、24490个GT、1468份预测TXT、缺1份。只有CMFC缺失文件名已明确为 `01322.jpg`，其他两组的缺失文件名尚未回传。用户已授权并完成本地逐目标对照工具；本地没有真实数据或服务器逐目标结果，未新增训练。旧交接文档为较早时间点快照，当前状态以本节和WORKLOG最新记录为准。
- 数值更正：按原交接基线 freight car/bus AP50:95=33.8/70.3，CMFC-only 相对差值为 +1.3/−1.2 个百分点；此前 WORKLOG 与派生 JSON 的 +1.8/−2.2 是混列转录错误，现已更正。总体指标不变。历史快照及历史解释不自动成为当前结论。

- 已收到基线、CMFC-only v2、AFSF-only、完整版 v2/v3 的独立 val 结果。前四个已有配置的数值来自 2026-11-02 交接，CMFC-only v2 来自用户本次输出；统一为 1469 张 val 图像、24490 个目标。用户将本次称为 test 结果，但这是 test.py 对 val 数据的评估，不是 8980 张 test 集。不要混入历史 test 或训练末轮指标，也不要重复启动已完成的实验。
- 本轮已完成 **CMFC-only v2** 的训练和 val 结果回传：`models/ccsfuse/yolov5m_cmfc_add_v2.yaml` 配套 `data/hyp.ccsfuse_dronevehicle_v2.yaml`。与完整版 v2 相比，仅 27/28/29 层由 AFSF 换为 PairAdd，保持 small 初始化、std=0.001、回注 A 和学习率倍数 10。
- 本地配置与检查已完成，代码提交 `5cb0b49` 已上传到 `codex/cmfc-only-v2-ablation`；19 项本地检查全部通过，使用真实 COCO 权重和合成数据。
- 用户已回传服务器真实数据短检查：两路 306/306，共 612/890；Focus 比对均为 True；train 17990、0 missing、33 empty；PairAdd/Detect 形状正确；loss 和梯度有限；参数 53.79M，GFLOPs 93.44（thop）/94.48（flop_counter），batch 2 AMP 峰值 1.85 GiB。本次服务器检查结果来自用户终端输出，不是 Codex 远程执行；用户未回传同步后的 `git rev-parse HEAD` 输出。
- 用户随后要求训练指令，已提供在 tmux `cmfc_only_v2` 中启动的命令：45 epoch、640×640、batch 12、workers 8、device 0、nbs 12，计划实验名 `cmfc_only_m_v2_b12`，默认只加载 COCO 主干。详细命令和续训方法见 WORKLOG.md。
- **正式训练已完成：用户回传 44/44 及 `45 epochs completed in 6.660 hours.`。** 末轮训练内 val 为 P/R 69.7/65.8%、mAP50/mAP75/mAP50:95 68.1/56.2/46.5%，1469 张图、24490 个目标；日志 `gpu_mem=13.7G`，不是已核实的峰值 allocated 显存。实际保存目录和服务器提交仍未从输出确认，计划名 `cmfc_only_m_v2_b12` 不等于已核实路径。不要重复启动这轮已完成训练。
- **CMFC-only v2 独立 val 已回传**：P/R 72.2/65.2%，mAP50/mAP75/mAP50:95 68.4/56.4/46.7%。按用户沿用此前 best.pt、conf=0.001、NMS IoU=0.5、batch 12、640 的指令理解；用户随后确认评估保存目录为 `runs/test/cmfc_only_m_v2_b12_val`（位于服务器）；完整启动命令、实际权重路径和服务器 HEAD 仍未直接回传。Codex 不能直接看到该服务器目录，仅读取用户提供的输出。
- 当前五组对比（%，CMFC-only 的评估参数按上述上下文理解）：

| 实验 | mAP50 | mAP75 | mAP50:95 |
|---|---:|---:|---:|
| Add 基线 | 68.8 | 56.1 | 46.8 |
| CMFC-only v2 | 68.4 | 56.4 | 46.7 |
| AFSF-only | 69.2 | 56.2 | 47.2 |
| 完整版 v2 | 70.7 | 59.9 | 49.2 |
| 完整版 v3 | 67.5 | 55.1 | 45.9 |

- 当前单次结果中，CMFC-only 相对基线为 -0.4/+0.3/-0.1 个百分点，总体接近基线；完整版 v2 比 CMFC-only 高 2.3/3.5/2.5 个百分点，仍是当前最好的一组。结果提示模块可能存在配合作用，不支持直接删除 CMFC，也不能据单次实验宣称已证明协同或 CMFC 普遍无效。
- 【用户日志】conf=0.25、匹配IoU=0.5时，van TP为253/219/281，FP为327/230/347，真实van被配为car的数量为347/356/349（Add/CMFC/full顺序）。这是共同的聚合错误模式；数量接近不代表同一批GT，也不能直接证明分类头、标注或类不平衡是根因。
- 【本地实现】新增 `tools/compare_detections.py`、`tools/paired_detection_render.py`、`tools/test_compare_detections.py`。读取三份已有summary.json和原始GT/预测，先对完整图像全部类别匹配，再按图片相对路径和非空GT行号（从0开始）对齐focus目标。默认六阈值，全量逐目标状态、成对TP gain/loss、共同van→car均从实际GT交集计算。
- 工具核对三组重算混淆矩阵及各类计数与旧报告一致、只接受旧报告已记录的缺TXT，保存当前输入SHA256并列出缺TXT图片的GT类别。旧报告没有原始输入内容哈希，所以同汇总数的内容变化未必被识别；也不能由TXT核实checkpoint或NMS参数。已有输出拒绝覆盖，禁止向输入目录写入。
- 输出 `summary.txt/json`、`paired_targets.csv`、`transitions.csv`、`selection.csv`、`README.txt`及 `images/*.png`。主图工作点为conf=0.25、匹配IoU=0.5，每组固定seed抽样10个；两行RGB/IR、四列GT/Add/CMFC/full。两行叠加的是同一组融合预测，不是单模态评估。候选最高van/car分数使用全部保存框，与阈值匹配分开记录；无候选记空而非0。组可重叠，样本不代表总体比例。
- 【本地验证】17项合成/CLI测试通过，包含同汇总但不同GT、gain/loss互抵、阈值变化、输入保护、报告不一致拒绝及八面板端到端；合成八面板已视觉检查。原诊断工具未修改；本地未运行真实DroneVehicle数据。
- **下一步**：用户同步GitHub实验分支，运行 `tools/compare_detections.py --reports`，依次传入Add、CMFC、full三份summary.json，使用 `--expected-focus 725 --samples-per-group 10`；完整命令见WORKLOG本轮记录。回传summary.txt和共同错误/模型分歧组的部分PNG，再决定是否改数据、训练或模块；不自动重训。
- 离线诊断默认置信度 0.05/0.25/0.5、匹配 IoU 0.5/0.75；先匹配同类 TP，再匹配剩余跨类框。输出固定阈值 TP/FP/FN、混淆矩阵、疑似定位问题及错例 CSV。该匹配口径是诊断启发式，不等于原 AP 匹配，也不要求重现原评估的 van R=22.7%；只观察已保存的 NMS 后预测。
- 历史单模型诊断工具曾通过 39 项合成边界/命令行测试；补充工具哈希和执行命令元数据后，5 项 CLI 检查再次通过。保持完整版 v2 作为当前参考；依据小差异做结构取舍需后续多 seed，当前不自动启动新训练。
- 口径提醒：训练末轮验证 NMS IoU 为 0.6，不能与上述独立 val 混比；现有 best.pt 按 `0.1*mAP50 + 0.9*mAP75` 选取，不代表最高 mAP50。继续保留旧实验，不顺带修改 seed、选优规则、优化器或调度。

- 本地恢复：完整 Git 历史和 CMFC-only v2 配置已从 `origin/codex/cmfc-only-v2-ablation` 的 `5cb0b49` 恢复；此前未提交的两份文档更新已重建，并补记训练完成。历史检查证据保存在本地 `runs/recovery_2026-11-02/`（Git 忽略）。本次未恢复未纳入 Git 的数据集、权重、训练产物或论文，未操作服务器。此前恢复、训练完成与 val 对比的文档修改随本轮诊断工具一起纳入实验分支提交；模型、配置、train.py/test.py 与训练基线 5cb0b49 一致，新增工具独立运行。
- 旧文档还记录用户手动清理 `D:\CCSFuse_wheels` 的待办；本轮未执行，也未核实该历史事项是否已完成。

## 踩过的坑（结论）
- **train.py 曾在工作区被意外改动**（2026-10-17，第 6 步的优化器分组被撤回，注释中还多了空格；用户不知情，推测是 VS Code 中误操作），已用 `git checkout -- train.py` 恢复。ccsfuse_m 训练在改动之前就已完成，不受影响。**开始长时间训练前先跑 `git status --short`**；旧版曾打印 `Optimizer: 3 extra parameter tensors`，分组为 188 / 182 / 146；这不是所有配置的统一要求。v2 的 lambda 直接进入 CMFC 专用参数组，应核对 `CMFC lr multiplier 10` 及参数无遗漏/重复，本地检查已通过。
- CFT 原来的权重加载（intersect_dicts 按键名）只给 RGB 主干加载 COCO 权重，IR 主干是随机初始化，但日志里的 "Transferred 306/812" 看上去正常。已用 load_coco_dual_stream 修复。
- 在预训练主干中间插入的模块，如果初始输出不为零，会从训练一开始就破坏预训练特征（见实验结果中 ccsfuse_m 的诊断）。新模块应考虑零初始化。
- 新版环境下 CFT 有三处直接崩溃，已修复（见"改动过的 CFT 原有文件"）：setuptools 84 没有 pkg_resources；loss 中 clamp_ 的 dtype 错误；test.py 把 6 通道图像传给画图函数（发生在后台线程，只打印 Traceback，不中断训练）。
- CFT 用 `open()` 读 yaml 时不指定编码，中文 Windows 默认按 GBK 解码。含中文注释的 yaml 必须按 UTF-8 读，已在 train/test/yolo_test 中修改。
- 用 PowerShell 写文件时：`Set-Content -Encoding utf8` 会加 BOM，应改用 `[IO.File]::WriteAllText(path, text, (New-Object Text.UTF8Encoding $false))`；在双引号字符串里，反引号是转义符，Markdown 的反引号会丢失。
- 学习率调度滞后一个 epoch（YOLOv5/CFT 原有行为，不改）：train.py 中 `scheduler.last_epoch = start_epoch - 1`，加上新版 torch 的 LambdaLR 在创建时已经执行过一次 step，第 e 个 epoch 用的是 lf(e−1)。不中断训练和续训都是这样；tensorboard 中 x/lr0 的记录即为实际值。
- results.txt 每行依次为：epoch、显存、box、obj、cls、total、目标数、图像尺寸，然后是 8 个指标：P、R、mAP50、mAP75、mAP50:95、val box/obj/cls。CFT 比 YOLOv5 多了 mAP75，mAP50 在第 11 列。
- 训练结束时 last.pt / best.pt 会被精简（去掉优化器，转成 fp16），读 λ 等参数可以直接用；想检查保存的优化器状态，只能在训练中途读取。
- test.py 默认会把每张图的预测写成 txt（`--save-txt` 默认 True，命令行关不掉），这是 test 集评估较慢的原因之一。
- lr0=0.001（论文值）时，1% 子集 30 epoch 的 mAP50 只到 0.12。原因是优化步数少（约 450 步），不是 bug；过拟合检查用 lr0=0.01。
- 1% 子集这样的小数据，YOLOv5 的 warmup 默认至少 1000 步，会占满整个训练。用 `--hyp-set warmup_epochs=1 warmup_min_iters=45` 缩短。
- 加载器读 txt 图片列表时，会先把 `/` 和 `\` 折成 `os.sep`，再把第一段 `images` 换成 `labels`。列表不必再写 Windows 反斜杠。`tools/make_subset.py` 写出 POSIX 绝对路径。cache 生成在 txt 旁边，不会覆盖完整训练集的 cache。
- VS Code 曾提示创建 D:\CCSFuse\.venv（uv 的 Python 3.15，没有 torch），用户同意后已删除。之后用户在 VS Code 中点"运行"时用错过这个 3.15 解释器（报 `No module named 'thop'`）。应拒绝 VS Code 创建虚拟环境的提示，把解释器选为 D:\Miniconda\envs\ccsfuse\python.exe，或直接写 python.exe 的完整路径。
- 在终端运行 `tools/*.py` 前要先进入仓库根目录（Windows 上是 `D:\project\CCFUSE`，服务器上是 `~/bzp/CCFUSE`），因为脚本的默认路径是相对路径。
- download.pytorch.org、阿里云、上海交大的镜像下载很慢且会断线。torch wheel 用南京大学镜像 https://mirrors.nju.edu.cn/pytorch/whl/cu124/ 下载，并用官方 SHA256 校验（已写在 requirements-ccsfuse.txt 中）。PyPI 用清华镜像。
- numpy 必须 <1.24：代码中用到了 `np.int`。
- 安装依赖要用 requirements-ccsfuse.txt 中写死的版本，加 `--only-binary :all:`。不写死版本时 pip 会长时间回溯。
- torch ≥2.6 的 `torch.load` 默认 `weights_only=True`，打不开 v5.0 的整模块 pickle。仓库里每处 `torch.load` 都写了 `weights_only=False`（2.5 也接受这个参数）。服务器因此使用 torch 2.7.1+cu128。不要改回默认，也不要再升到未核对过的版本。
- CFT 的 `attempt_download` 在本地没有权重文件时，会下载 GitHub 最新版权重，与这份代码不兼容。`train_rgb_ir` 已改为找不到本地文件就报错。
- CFT 原来的 hyp.scratch.yaml 中 mosaic=1.0；CCSFuse 用的 hyp.ccsfuse_dronevehicle.yaml 中 mosaic=0。
- utils\datasets.py 第 897 行的 `assert (a, b), ...` 恒为真，空数据集不会报错（会打印 SyntaxWarning，不影响运行）。加载时 IR 那一行打印的是 rgb 的 cache 路径（第 937 行的打印错误），实际加载用的是 IR 的 cache。
- 用 PowerShell 删除盘符根目录下的文件夹会被工具的安全规则拦截，需要用户手动删除。
- 在 Python 中读取下载的数据时用 `python -I`，脚本不要放在数据目录里。

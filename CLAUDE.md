# CCSFuse 复现项目

## 项目目标
复现 CCSFuse（Zhang et al., IEEE TGRS 2026, Art. no. 5400314，UAV RGB-IR 目标检测：CMFC + AFSF，基于 YOLOv5 双流）。
官方代码未公开。详细方案见 D:\CCSFuse\CCSFuse_复现方案.md，过程记录见 D:\CCSFuse\WORKLOG.md。

## 工作约定
- 用中文回复。
- 执行命令前先说明要做什么。
- 每完成一步先停下来，告诉用户如何验证，再继续下一步。
- 方案中所有【推断】项（论文未给出的设置）都要做成 yaml 或命令行可配置，并在代码注释中标明"论文未给出，推断"。
- 删除操作只删确认为空或确认无用的内容，删除前先核实；用户的原始数据只读。

## 环境
- GPU：NVIDIA RTX 3070 Ti Laptop，8 GB，算力 8.6；驱动 616.64（支持 CUDA 13.4）。
  - 桌面程序常驻占用 1–1.7 GB 显存，实际可分配约 7.25 GB。
  - "CUDA 系统内存回退"已关闭，显存不足时会正常报 OOM。
- conda：D:\Miniconda，**不在 PATH 中**。
- 环境：ccsfuse，路径 D:\Miniconda\envs\ccsfuse。
  - 激活：`D:\Miniconda\Scripts\activate` 后执行 `conda activate ccsfuse`。
  - 或直接调用 `D:\Miniconda\envs\ccsfuse\python.exe`。
- 版本：Python 3.10.22，torch 2.5.1+cu124，torchvision 0.20.1+cu124，cuDNN 9.1，numpy 1.23.5。
  - 其余依赖的版本固定在 D:\CCSFuse\requirements-ccsfuse.txt。
- 环境自检：在 D:\CCSFuse 下运行 `python check_env.py`，应输出 ALL CHECKS PASSED。
- pip 缓存目录：D:\pip-cache（在 C:\Users\bzp02\AppData\Roaming\pip\pip.ini 中设置）。

## 代码库
- 路径：D:\CCSFuse，即 CFT 官方实现。
  - 来源：https://github.com/DocF/multispectral-object-detection
  - commit：fb591c9b163177c0e950db08e213e24ddc912d41
- 基于 YOLOv5 v5.0 时期的代码：
  - 双流训练入口：D:\CCSFuse\train.py 中的 `train_rgb_ir`
  - 双流模型：D:\CCSFuse\models\yolo_test.py（train.py 导入的是它，不是 models\yolo.py）；yaml 中 `from: -4` 表示该层读取 IR 输入
  - 数据集类：D:\CCSFuse\utils\datasets.py 中的 `LoadMultiModalImagesAndLabels`
  - CFT 原有模型配置：models\transformer\*.yaml；CCSFuse 配置放在 models\ccsfuse\
- git：工作分支 `ccsfuse`，`main` 保持 CFT 原样。提交 cba1f45 是会话 1 的成果（方案、环境检查、数据工具和配置），87d523e 是第 1 步。
  - 新增 `.gitignore`：忽略 `__pycache__/`、`datasets/`、`weights/*.pt`、`runs/`、`*.cache`。
- **改动过的 CFT 原有文件**（第 1 步）：
  - train.py（仅 `train_rgb_ir` 和命令行参数）：
    - COCO 权重改用 `load_coco_dual_stream` 加载，两路主干都能拿到（原来的 intersect_dicts 只覆盖 RGB 主干）；本地没有权重文件时直接报错，不再调用 attempt_download。
    - 新增 `--nbs`、`--hyp-set`、`--pretrain-neck`；nesterov 和 warmup 最少步数改由 hyp 中的 `nesterov`、`warmup_min_iters` 控制；默认值改为本机配置（见"已核实的结论"）；hyp 和 data yaml 按 UTF-8 读取。
  - utils\loss.py 第 211 行：`clamp_(0, gain[3] - 1)` 改为 `int(gain[3]) - 1`。原写法在 torch ≥1.12 下报 int64/float 类型错误，训练无法运行。
  - utils\general.py 的 `check_requirements`：导入不到 pkg_resources 时跳过检查。setuptools 84 已移除 pkg_resources，原代码会导致 train.py 启动即崩溃。
  - test.py：画图时只传 RGB 三通道（原来传 6 通道，会在最后一个 epoch 的画图线程中报错）；data yaml 按 UTF-8 读取。
  - models\yolo_test.py：模型 yaml 按 UTF-8 读取。新配置中有中文注释，系统默认的 GBK 编码会读错。
- **改动过的 CFT 原有文件**（第 2 步）：
  - utils\datasets.py：`LoadMultiModalImagesAndLabels` 非 Mosaic 分支恢复成对的 random_perspective_rgb_ir；hyp `hsv_ir: none|independent|same`（缺省为 independent，即 CFT 原行为）；`augment_hsv` 增加 `gains=` 参数并返回增益。
  - train.py：`--hyp-set` 允许 `hsv_ir`。
- **改动过的 CFT 原有文件**（第 6 步）：
  - models\yolo_test.py：导入并在 parse_model 中注册 CMFC/CMFCInject/AFSF/PairAdd；模块构造支持 kwargs（从 yaml 顶层的 cmfc/cmfc_inject/afsf 读取）。
  - train.py（train_rgb_ir）：LayerNorm weight 不衰减；未分组参数（λ）补进 pg0。注意 train.py 里有两段相同的分组代码，第一段属于不用的单流 `train`，没有改。
- 新增文件（会话 3，第 4–6 步）：models\ccsfuse_modules.py（CMFC、AFSF、CMFCInject、PairAdd）；tools\test_cmfc.py、tools\test_afsf.py；models\ccsfuse\yolov5m_{ccsfuse,cmfc_add,afsf}.yaml。
- 新增文件（会话 3，第 2 步）：tools\check_aug.py（`python tools\check_aug.py --n 10 --out runs/viz/aug_pairs.jpg`，可加 `--hyp-set hsv_ir=same`）。
- 新增文件（会话 1）：check_env.py、requirements-ccsfuse.txt；tools\ 下的 prepare_dronevehicle.py、prepare_llvip.py、viz_pairs.py、check_loader.py、check_weights.py；data\multispectral\ 下的 DroneVehicle.yaml、DroneVehicle_test.yaml、LLVIP_ccsfuse.yaml。
- 新增文件（会话 2，第 1 步）：
  - models\ccsfuse\yolov5m_dual_add.yaml（以及调试用的 s 版 yolov5s_dual_add.yaml）：双流加 Add 融合，即 Table I 第 1 行的基线，nc=5。`pretrain_map` 写明哪些层对应 COCO 的哪些层。
  - data\hyp.ccsfuse_dronevehicle.yaml：论文超参。论文未给出的项都标注为"推断"。
  - utils\ccsfuse.py：`load_coco_dual_stream()`，按 `pretrain_map` 把 COCO 主干权重复制到 RGB 和 IR 两路；`--pretrain-neck` 时也复制 neck。
  - tools\check_model.py：建模型并迁移权重，用一个真实 batch 做前向和反向，打印各组的 Transferred 数、P3/P4/P5 和 Detect 的形状、loss、显存、参数量、GFLOPs。
  - tools\make_subset.py：按比例随机抽取配对子集，写成 txt 列表和 data yaml（train 与 val 指向同一份，用于过拟合检查）。
  - data\multispectral\DroneVehicle_1pct.yaml 和 DroneVehicle_1pct_train_{rgb,ir}.txt：1% 子集，180 对，seed 0。
- 预训练权重（ultralytics v5.0 release，COCO）：
  - D:\CCSFuse\weights\yolov5s.pt（调试用）
  - D:\CCSFuse\weights\yolov5m.pt（正式训练用）
- v5.0 主干结构：第 0 层 Focus，第 8 层 SPP。
  - P3 / P4 / P5 来自第 4 / 6 / 9 层（均为 C3）。
  - m 版通道数 192 / 384 / 768，s 版 128 / 256 / 512。

## CFT 加载器的约定
- 标签路径由图片路径推出：把第一个 `\images\` 替换成 `\labels\`，扩展名改为 `.txt`。
- RGB 和 IR 各自按文件名排序，然后按位置配对；两边的文件名必须完全一致。
- **训练 loss 只用 RGB 一侧的标签（labels_rgb）。**
- 每个样本是 6 通道张量（RGB + IR）。
- 加载器会在 labels 目录旁生成 `<split>.cache`，数据变动后会自动重建。
- `test.py` 只读取数据 yaml 中的 `val_rgb` / `val_ir`。

## 数据

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

### DroneVehicle 转换规则（D:\CCSFuse\tools\prepare_dronevehicle.py）
- 四边各裁掉 100 px 白边，840×712 变为 640×512，JPEG 质量 95。
- 框的处理：
  - polygon 旋转框：取四个顶点的 min/max 得到水平框；
  - bndbox：直接使用；
  - 所有框平移 −100 px 后裁剪到图像范围内，裁剪后宽或高 <2 px 的丢弃；
  - point 类型的标注丢弃。
- **使用 IR 标注（labelr），同一份标签同时写到 rgb\labels 和 ir\labels。**
- 类别 id（论文 Table I 顺序）：car=0，truck=1，freight car=2，bus=3，van=4。
- 原始类名的映射：
  - `feright car` / `feright_car` / `feright` → freight car
  - `truvk` → truck
  - `*` → 丢弃
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

### LLVIP 转换规则（D:\CCSFuse\tools\prepare_llvip.py）
- 图像原样复制，不缩放（1280×1024，由加载器缩放到 640）。
- 两侧共用同一份标注；只有一个类，person=0。
- 宽或高 <1 px 的框丢弃（原始数据中有 5 个宽度为 0 的框）。
- 运行命令：
  `D:\Miniconda\envs\ccsfuse\python.exe -I D:\CCSFuse\tools\prepare_llvip.py --src D:\CCSFuse\datasets\LLVIP_raw\LLVIP --dst D:\CCSFuse\datasets\LLVIP`
- 数量：

  | 划分 | 对数 | 目标数 |
  |---|---|---|
  | train | 12025 | 34,130 |
  | test | 3463 | 8,302 |

  - 空标签文件 2 个。

### 数据配置文件
- 训练（用 val 验证）：D:\CCSFuse\data\multispectral\DroneVehicle.yaml
- 最终评估（val 键指向 test）：D:\CCSFuse\data\multispectral\DroneVehicle_test.yaml
- LLVIP（val 键指向 test）：D:\CCSFuse\data\multispectral\LLVIP_ccsfuse.yaml
- 过拟合检查（train 1% 子集，180 对，train 和 val 都指向它）：D:\CCSFuse\data\multispectral\DroneVehicle_1pct.yaml

### 检查工具（在 D:\CCSFuse 下运行；必须用 ccsfuse 环境的 python，相对路径以 D:\CCSFuse 为准）
- `python tools\check_loader.py --data <yaml>`：用 CFT 加载器读取数据，检查配对、目标数和类别。
- `python tools\viz_pairs.py --root <数据目录> --split train --n 10 --out <文件名.jpg>`：可视化 RGB/IR 配对和标注框。
- `python tools\check_weights.py`：加载权重，打印主干各层的输出形状。
- `python tools\check_model.py [--cfg ... --weights ...]`：第 1 步检查，查看两路主干的权重迁移情况、P3/P4/P5 形状、loss、显存、参数量和 GFLOPs。
- `python tools\make_subset.py --frac 0.01 --out data\multispectral\<名字>.yaml`：生成配对子集。

## 已核实的结论（会话 2，开始第 1 步之前）
- **预训练权重**：CFT 原来的加载方式（intersect_dicts）只覆盖 RGB 主干，Transferred 306/812，IR 主干 0/306。改用 load_coco_dual_stream 后为 rgb 306/306、ir 306/306、Transferred 612/812，两路的 Focus 权重都与 COCO 第 0 层逐元素相同。neck 和 head 默认不加载 COCO 权重（论文只说主干用 COCO 预训练）。
- **增强**（CFT 加载器，第 2 步要改）：
  - Mosaic、仿射变换、翻转对 RGB/IR 是同步的。Mosaic 时两路共用拼接中心、图块和变换矩阵 M，实测两路灰色填充区域重合 99.1%；水平翻转实测两路逐像素一致。
  - HSV 会作用到 IR，而且两路的随机增益各自独立（datasets.py 第 1230–1231 行分别调用两次 augment_hsv）。
  - **关闭 Mosaic 后没有随机裁剪或缩放**：非 Mosaic 分支的 random_perspective 被注释掉了，实测两次读取结果完全相同。
- **模型规模**：双流 YOLOv5m 加 Add 融合，33.43 M 参数（论文 CCSFuse 为 33.50 M），84.32 GFLOPs（按 2×MACs 计，论文 174.12）。参数量支持 m 版的推断，GFLOPs 的差距留到第 6 步加入 CMFC/AFSF 后再对比。
- **超参数**（data\hyp.ccsfuse_dronevehicle.yaml，可用 `--hyp-set` 覆盖）：
  - 论文给出：lr0 0.001、momentum 0.9、SGD、45 epoch、关闭 Mosaic（论文未提 Mosaic）。
  - 推断：lrf 0.01（YOLOv5 余弦退火到 lr0×lrf，视作 One-Cycle）、nesterov False、weight_decay 5e-4、warmup 3 epoch 且最少 1000 步、warmup_bias_lr 0.01、translate 0.1、scale 0.5、HSV 用 YOLOv5 默认值。
- **训练命令**（默认即本机配置）：`D:\Miniconda\envs\ccsfuse\python.exe train.py`，等价于 `--cfg models/ccsfuse/yolov5m_dual_add.yaml --weights weights/yolov5m.pt --data data/multispectral/DroneVehicle.yaml --hyp data/hyp.ccsfuse_dronevehicle.yaml --epochs 45 --batch-size 4 --nbs 12 --workers 2`，AMP 自动开启。
  - 梯度累积 round(12/4) = 3 步；weight_decay 缩放系数为 1（日志显示 Scaled weight_decay = 0.0005）。

## 当前进度
- 已完成：环境配置；克隆代码库；整理并转换 DroneVehicle、LLVIP；下载并验证 v5.0 的 s/m 权重（会话 1）。
- **方案第 7 节第 1 步已完成**（commit 87d523e）：双流 YOLOv5m 基线，Add 融合。
  - check_model.py 的输出：Transferred 612/812（rgb 306/306，ir 306/306）；融合后的 P3/P4/P5 为 (4,192,80,80) / (4,384,40,40) / (4,768,20,20)；Detect 为 (4,3,{80,40,20},{80,40,20},10)。
    - loss 和梯度均为有限值；前向加反向峰值显存 2.24 GiB；33.43 M 参数。
  - 1% 子集过拟合，30 epoch，batch 4，训练集同时用作验证集：
    - 论文 lr0=0.001：loss 0.274→0.177，mAP50 0.12，见 runs\train\overfit_add_m；
    - lr0=0.01：loss 0.275→0.104，mAP50 0.47，car 类 0.97，见 runs\train\overfit_add_m_lr01；
    - 每 epoch 约 14 秒（含验证），训练显存 3.3 GB。
  - 用户已在本地运行 check_model.py。
- **方案第 7 节第 2 步（数据管线）已完成**（commit 5f70fc2）（会话 3，详见 WORKLOG）：
  - 非 Mosaic 分支恢复 random_perspective_rgb_ir（两路共用矩阵）；新增 hyp `hsv_ir`，默认 none；新增 tools\check_aug.py。
  - check_aug.py 三项检查全部通过；可视化图在 runs\viz\aug_pairs.jpg。
  - 1% 过拟合复跑（lr0=0.01）：loss 0.269→0.127，mAP50 0.456（无增强时为 0.104 / 0.472），见 runs\train\overfit_add_m_lr01_aug。
- 断点续训已用 1% 子集测试通过（强制结束后执行 `python train.py --resume runs\train\<name>\weights\last.pt`，epoch、hyp-set、优化器、EMA 都能恢复）。
- **第 3 步（完整训练集基线）已训练完**（用户在 VS Code 中运行，约 7.8 h）：runs\train\baseline_add_m。
  - test 集 mAP50 **67.3**（论文 72.8）：car 97.7、truck 68.8、freight car 46.6、bus 94.3、van 28.8。差距全部来自 van（51% 被判成 car）和 freight car（42% 被判成 truck）。详见 WORKLOG。
  - test 评估命令：`python test.py --weights <best.pt> --data data/multispectral/DroneVehicle_test.yaml --batch-size 8 --iou-thres 0.6 --task test --verbose --name <名字>`（必须显式写 `--iou-thres 0.6`，test.py 命令行默认 0.5；运行约 16 分钟）。
  - 用户决定：接受 67.3 作为自己的基线，后续按相对提升评判（方案第 6 节的判定标准）。
- **第 4 步（CMFC）已实现，单元测试全部通过**（commit 909c743）：models\ccsfuse_modules.py 中的 `CMFC`，测试为 `python tools\test_cmfc.py`（应输出 ALL CMFC TESTS PASSED）。
  - m 版 3 个 CMFC 共 20.37 M 参数，加上后总参数约 54 M，与论文的 33.50 M 矛盾（详见 WORKLOG）。用户决定：保持 m 版和方案默认 CMFC（卷积 SR + 输出投影），按相对提升评判；轻量 SR 暂不加，若 +CMFC 提升明显不足再作为排查手段。
- **第 5 步（AFSF）已实现，单元测试全部通过**（commit 76ba902）：models\ccsfuse_modules.py 中的 `AFSF`（`reduction=4`，`sigmoid_after_mlp=False`），测试为 `python tools\test_afsf.py`（应输出 ALL AFSF TESTS PASSED）。3 个 AFSF 共 4.65 M 参数；m 版 CCSFuse 合计约 58.45 M。
- **第 6 步（接入网络）已完成**（详见 WORKLOG）：
  - 模型配置：`models\ccsfuse\yolov5m_ccsfuse.yaml`（完整版）、`yolov5m_cmfc_add.yaml`（+CMFC）、`yolov5m_afsf.yaml`（+AFSF）。推断项在 yaml 顶层的 `cmfc:`、`cmfc_inject:`（A/B）、`afsf:` 中设置；Table II 用 `cmfc: branch_weights: [1, 0]` 等。
  - 参数量：完整版 58.45 M、+CMFC 53.79 M、+AFSF 38.08 M。GFLOPs（flop_counter）：97.3 / 94.5 / 86.5。
  - 速度约为基线的 1/1.6。推算完整训练：完整版约 12.5 h，+CMFC 约 12 h，+AFSF 约 8–8.5 h（都由用户自己运行）。
  - train.py 会把 λ 补进优化器的不衰减组（日志中有 "Optimizer: 3 extra parameter tensors"）。
  - test.py（attempt_load + Conv/BN 融合）能正常评估新模型；推理 39 ms/张，基线为 11 ms/张，主要因为注意力在 fp32 下计算。
- **第 7 步进行中**：用户决定按"完整版 → +CMFC → +AFSF"的顺序跑消融（先看最关键的结果），每组结束后用 test 集评估。
  - 完整版 runs\train\ccsfuse_m 已训练完：**test mAP50 66.1，比基线低 1.2**（论文 +6.6）；freight car −4.3，van −1.4，其余持平。val 曲线从 epoch 20 起落后。λ 几乎没动（约 0.5/0.5）。
  - 诊断：CMFC 初始化时补偿量就达到原特征的 50–66%（‖Δ‖/‖f‖），通过回注方式 A 破坏了 COCO 预训练的 stage4/5；ReLU² 分支幅值大于 Softmax 分支。候选改法：输出投影零初始化（推荐）、ris_norm。详见 WORKLOG。
  - **用户选了方案 1**：CMFC 新增 proj_init: zero（输出投影零初始化，初始为恒等映射），完整版和 +CMFC 的 yaml 已改。下一步：用户重训完整版 --name ccsfuse_m_zinit。
- **长时间训练一律交给用户在自己电脑上运行**（用户要求）：给出完整命令、监控方法、续训命令，等用户贴回结果。
- 第 6 步时注意：CMFC 的 λ 是单独定义的 nn.Parameter，train.py 的优化器分组（pg0/pg1/pg2）收集不到它，需要单独加进优化器。
- 后台训练：无（2026-10-15 04:00 核实，没有 python 进程在运行）。
- 用户待办：手动删除 D:\CCSFuse_wheels（约 2.4 GB）。

## 踩过的坑（结论）
- CFT 原来的权重加载（intersect_dicts 按键名）只给 RGB 主干加载 COCO 权重，IR 主干（第 10–19 层）是随机初始化，但日志里的 "Transferred 306/812" 看上去正常。已用 load_coco_dual_stream 修复。
- 新版环境下 CFT 有三处直接崩溃，已修复（详见"代码库"）：setuptools 84 没有 pkg_resources；loss 中 clamp_ 的 dtype 错误；test.py 把 6 通道图像传给画图函数。其中画图的错误发生在后台线程，只打印 Traceback，不会中断训练。
- CFT 用 `open()` 读 yaml 时不指定编码，中文 Windows 默认按 GBK 解码。含中文注释的 yaml 必须按 UTF-8 读，已在 train/test/yolo_test 中修改。
- 学习率调度滞后一个 epoch（YOLOv5/CFT 原有行为，不改）：train.py 第 684 行 `scheduler.last_epoch = start_epoch - 1`，加上新版 torch 的 LambdaLR 在创建时已经执行过一次 step，第 e 个 epoch 用的是 lf(e−1)。不中断训练和续训都是这样；tensorboard 中 x/lr0 的记录即为实际值。
- results.txt 每行 8 个指标依次为：P、R、mAP50、mAP75、mAP50:95、val box/obj/cls。CFT 比 YOLOv5 多了 mAP75。
- lr0=0.001（论文值）时，1% 子集 30 epoch 的 mAP50 只到 0.12。原因是优化步数少（约 450 步），不是 bug；用 lr0=0.01 可以快速过拟合。
- 1% 子集这样的小数据跑 YOLOv5 时，warmup 默认至少 1000 步，会占满整个训练。用 `--hyp-set warmup_epochs=1 warmup_min_iters=45` 缩短。
- 加载器读 txt 图片列表时，标签按路径中的 `\images\` 替换得到，所以列表必须写 Windows 反斜杠路径（make_subset.py 已处理）。cache 生成在 txt 旁边，不会覆盖完整训练集的 cache。
- git 没有配置作者身份。提交时用 `git -c user.name=bzp02 -c user.email=miho1983511@gmail.com commit ...` 临时指定，不改全局配置。
- VS Code 曾提示创建 D:\CCSFuse\.venv（uv 的 Python 3.15，只有 pip，没有 torch），用户同意后已删除。以后 VS Code 再提示创建虚拟环境时应拒绝，并把解释器选为 D:\Miniconda\envs\ccsfuse\python.exe。
- 在终端运行 tools\*.py 前要先 `cd D:\CCSFuse`，因为脚本的默认路径是相对路径。
- download.pytorch.org、阿里云、上海交大的镜像下载很慢且会断线。torch wheel 用南京大学镜像 https://mirrors.nju.edu.cn/pytorch/whl/cu124/ 下载，并用官方 SHA256 校验（已写在 requirements-ccsfuse.txt 中）。PyPI 用清华镜像。
- numpy 必须 <1.24：代码中用到了 `np.int`。
- 安装依赖要用 requirements-ccsfuse.txt 中写死的版本，加 `--only-binary :all:`。不写死版本时 pip 会长时间回溯。
- torch 不要升级到 ≥2.6：v5.0 的权重是整个模块的 pickle，需要 `torch.load(..., weights_only=False)`，而 2.6 起默认值改为 True，加载会失败。
- CFT 的 `attempt_download` 在本地没有权重文件时，会下载 GitHub 最新版权重，与这份代码不兼容。`train_rgb_ir` 已改为找不到本地文件就报错；train.py 默认用 weights/yolov5m.pt。
- CFT 原来的 hyp.scratch.yaml 中 mosaic=1.0。CCSFuse 用的 hyp.ccsfuse_dronevehicle.yaml 中 mosaic=0。
- D:\CCSFuse\utils\datasets.py 第 897 行的 `assert (a, b), ...` 恒为真，空数据集不会报错（会打印 SyntaxWarning，不影响运行）。
- 用 PowerShell 删除盘符根目录下的文件夹会被工具的安全规则拦截，需要用户手动删除。
- train.py 曾在工作区被意外改动（2026-10-17，第 6 步的优化器分组被撤回，用户不知情），已用 git checkout 恢复。**每次开始长时间训练前先跑 git status，确认没有意外改动**；训练日志开头应有 Optimizer: 3 extra parameter tensors（含 CMFC 的模型）。
- 在 Python 中读取下载的数据时用 `python -I`，脚本不要放在数据目录里。

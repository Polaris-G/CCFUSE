# CCSFuse 复现项目

## 项目目标
复现 CCSFuse（Zhang et al., IEEE TGRS 2026, Art. no. 5400314，UAV RGB-IR 目标检测：CMFC + AFSF，基于 YOLOv5 双流）。
官方代码未公开。详细方案见 D:\CCSFuse\CCSFuse_复现方案.md，过程记录（含每一步的完整数据）见 D:\CCSFuse\WORKLOG.md。
论文 PDF 在仓库根目录（未纳入 git）；ccsfuse 环境中没有 PDF 库，读不了，以复现方案中标为【论文】的内容为准。

## 工作约定
- 用中文回复。
- 执行命令前先说明要做什么。
- 每完成一步先停下来，告诉用户如何验证，再继续下一步。
- 方案中所有【推断】项（论文未给出的设置）都要做成 yaml 或命令行可配置，并在代码注释中标明"论文未给出，推断"。
- 删除操作只删确认为空或确认无用的内容，删除前先核实；用户的原始数据只读。
- **长时间训练一律交给用户在自己电脑（VS Code 终端）上运行**：给出完整命令、开跑前的检查项、续训命令，等用户贴回输出。我只运行短任务：单元测试、check_model、1% 子集过拟合（约 10 分钟）、test 集评估（约 20 分钟）。
- 开始长时间训练前先让用户运行 `git status --short`，确认没有意外改动（见"踩过的坑"）。

## 环境
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

## 代码库
- 路径：D:\CCSFuse，即 CFT 官方实现。
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
- 工作分支 `ccsfuse`，`main` 保持 CFT 原样。`.gitignore` 忽略 `__pycache__/`、`datasets/`、`weights/*.pt`、`runs/`、`*.cache`。
- 未跟踪、不要提交：`.vscode/`（VS Code 设置）、论文 PDF。
- git 没有配置作者身份，提交时用 `git -c user.name=bzp02 -c user.email=miho1983511@gmail.com commit ...`。

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
- 配置：data\hyp.ccsfuse_dronevehicle.yaml（论文超参，推断项已标注）；data\multispectral\ 下的 DroneVehicle.yaml、DroneVehicle_test.yaml、LLVIP_ccsfuse.yaml、DroneVehicle_1pct.yaml（及其 txt 列表）。
- 工具（tools\）：prepare_dronevehicle.py、prepare_llvip.py、viz_pairs.py、check_loader.py、check_weights.py、check_model.py、make_subset.py、check_aug.py、test_cmfc.py、test_afsf.py；根目录 check_env.py、requirements-ccsfuse.txt。

## CCSFuse 模块的实现要点（推断项都在 yaml 中配置）
- **CMFC**（式 1–6）：每个模态各一套 Q/K/V；K/V 经 SR（Conv k=r, s=r + LayerNorm，PVT 写法）；输出 = (w_SIC·Softmax(s) + w_RIS·ReLU(s)²)·V，其中 s = QKᵀ/√d，(w_SIC, w_RIS) = softmax(λ1, λ2)，λ 初始化为 1；最后加残差。
  - yaml `cmfc:` 中的推断项：`num_heads: 8`、`out_proj: True`、`share_lambda: True`、`ris_norm: False`、`attn_fp32: True`、`branch_weights: null`（Table II 用 [1,0] / [0,1] / [0.5,0.5]）、**`proj_init: zero`**；每层的 sr_ratio 写在层参数里（stage 3/4/5 为 8/4/2）。
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
- 标签路径由图片路径推出：把第一个 `\images\` 替换成 `\labels\`，扩展名改为 `.txt`。
- RGB 和 IR 各自按文件名排序，然后按位置配对；两边的文件名必须完全一致。
- **训练 loss 只用 RGB 一侧的标签（labels_rgb）。**
- 每个样本是 6 通道张量（RGB + IR）。
- 加载器会在 labels 目录旁生成 `<split>.cache`，数据变动后会自动重建。
- `test.py` 只读取数据 yaml 中的 `val_rgb` / `val_ir`。
- 增强（第 2 步后）：不开 Mosaic；成对的随机平移和缩放（translate 0.1，scale 0.5）和水平翻转，两路几何变换完全一致；HSV 只作用于 RGB（`hsv_ir: none`）。

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

## 常用命令（都在 D:\CCSFuse 下运行，python 指 D:\Miniconda\envs\ccsfuse\python.exe）
- 训练（默认即基线和本机配置：yolov5m_dual_add.yaml、weights/yolov5m.pt、DroneVehicle.yaml、hyp.ccsfuse_dronevehicle.yaml、45 epoch、batch 4、nbs 12、workers 2，AMP 自动开启）：
  `python train.py --cfg models/ccsfuse/<配置>.yaml --name <名字>`
- 续训：`python train.py --resume runs\train\<名字>\weights\last.pt`（已测试：epoch、hyp-set、优化器、EMA 都能恢复）
- test 集评估（约 16–21 分钟）：
  `python test.py --weights runs/train/<名字>/weights/best.pt --data data/multispectral/DroneVehicle_test.yaml --batch-size 8 --iou-thres 0.6 --task test --verbose --name <名字>_test`
  - **必须显式写 `--iou-thres 0.6`**（test.py 命令行默认 0.5，与训练时的验证口径不一致）。
- 1% 子集过拟合检查（约 10 分钟）：
  `python train.py --cfg <配置> --data data/multispectral/DroneVehicle_1pct.yaml --epochs 30 --hyp-set lr0=0.01 warmup_epochs=1 warmup_min_iters=45 --name overfit_<名字>`
- 检查工具：
  - `python tools\check_model.py --cfg <配置> [--batch-size N]`：权重迁移（应为 rgb 306/306、ir 306/306）、融合层和 Detect 的形状、loss、显存、参数量、GFLOPs（thop 和 flop_counter 两种）、CMFC 分支权重。
  - `python tools\test_cmfc.py`、`python tools\test_afsf.py`：单元测试，应输出 ALL ... TESTS PASSED。
  - `python tools\check_aug.py --n 10 --out runs/viz/aug_pairs.jpg`：增强可视化和三项数值检查。
  - `python tools\check_loader.py --data <yaml>`、`python tools\viz_pairs.py --root <数据目录> --split train --n 10 --out <文件.jpg>`、`python tools\check_weights.py`、`python tools\make_subset.py --frac 0.01 --out data\multispectral\<名字>.yaml`。

## 超参数（data\hyp.ccsfuse_dronevehicle.yaml，可用 `--hyp-set` 覆盖）
- 论文给出：SGD、lr0 0.001、momentum 0.9、45 epoch、batch 12（2 卡 × 6）。
- 推断：lrf 0.01（YOLOv5 余弦退火，视作 One-Cycle）、nesterov False、weight_decay 5e-4、warmup 3 epoch 且最少 1000 步、warmup_bias_lr 0.01、mosaic 0、translate 0.1、scale 0.5、fliplr 0.5、HSV 用 YOLOv5 默认值、hsv_ir none。
- 本机：batch 4 + 梯度累积 3 步（nbs 12），weight_decay 缩放系数为 1。
- neck 和 head 不加载 COCO 权重（论文只说主干用 COCO 预训练），可用 `--pretrain-neck` 打开。

## 实验结果（DroneVehicle test 集，mAP50 %）

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

## 当前进度
- 第 1–6 步已完成（环境、数据、基线、CMFC、AFSF、接入网络）。
- **第 7 步（Table I 消融）进行中**：用户决定按"完整版 → +CMFC → +AFSF"的顺序跑。
  - **下一步：用户重训完整版（零初始化）**，代码版本为 0b0a7c7：
    `D:\Miniconda\envs\ccsfuse\python.exe train.py --cfg models/ccsfuse/yolov5m_ccsfuse.yaml --name ccsfuse_m_zinit`
    - 开头要核对：`YOLOv5 0b0a7c7`；CMFC 参数中有 `'proj_init': 'zero'`；`Optimizer: 3 extra parameter tensors`；`rgb 306/306, ir 306/306`。
    - 中途参照：ccsfuse_m 在 epoch 24 的 val mAP50 为 0.584，基线为 0.602。
    - 训练结束后由我在 test 集上评估，并读出最终的 λ。
  - 之后：若零初始化有效，再跑 +CMFC（`yolov5m_cmfc_add.yaml`，已是 proj_init zero）和 +AFSF（`yolov5m_afsf.yaml`，不含 CMFC，随时可以跑）；若仍不提升，试方案 2（ris_norm），并结合 +AFSF 的结果定位是哪个模块的问题。
- 之后的步骤（方案第 7 节）：第 8 步 Table II 权重策略消融（`branch_weights`）；第 9 步 LLVIP（Adam，lr 1e-4）；第 10 步记录 λ 的变化曲线。
- 服务器迁移：用户暂不考虑。若迁移后设置不变（batch 4、nbs 12、单卡），已有结果可以直接用；若改成论文的 batch 12 或多卡，四组都要重训。Linux 上要改 data yaml 中的 `D:/` 路径和 1% 子集 txt 中的反斜杠。
- 用户待办：手动删除 D:\CCSFuse_wheels（约 2.4 GB）。

## 踩过的坑（结论）
- **train.py 曾在工作区被意外改动**（2026-10-17，第 6 步的优化器分组被撤回，注释中还多了空格；用户不知情，推测是 VS Code 中误操作），已用 `git checkout -- train.py` 恢复。ccsfuse_m 训练在改动之前就已完成，不受影响。**开始长时间训练前先跑 `git status --short`**；含 CMFC 的模型，训练日志开头必须有 `Optimizer: 3 extra parameter tensors`，分组为 188 / 182 / 146。
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
- 加载器读 txt 图片列表时，标签按路径中的 `\images\` 替换得到，所以列表必须写 Windows 反斜杠路径（make_subset.py 已处理）。cache 生成在 txt 旁边，不会覆盖完整训练集的 cache。
- VS Code 曾提示创建 D:\CCSFuse\.venv（uv 的 Python 3.15，没有 torch），用户同意后已删除。之后用户在 VS Code 中点"运行"时用错过这个 3.15 解释器（报 `No module named 'thop'`）。应拒绝 VS Code 创建虚拟环境的提示，把解释器选为 D:\Miniconda\envs\ccsfuse\python.exe，或直接写 python.exe 的完整路径。
- 在终端运行 tools\*.py 前要先 `cd D:\CCSFuse`，因为脚本的默认路径是相对路径。
- download.pytorch.org、阿里云、上海交大的镜像下载很慢且会断线。torch wheel 用南京大学镜像 https://mirrors.nju.edu.cn/pytorch/whl/cu124/ 下载，并用官方 SHA256 校验（已写在 requirements-ccsfuse.txt 中）。PyPI 用清华镜像。
- numpy 必须 <1.24：代码中用到了 `np.int`。
- 安装依赖要用 requirements-ccsfuse.txt 中写死的版本，加 `--only-binary :all:`。不写死版本时 pip 会长时间回溯。
- torch 不要升级到 ≥2.6：v5.0 的权重是整个模块的 pickle，需要 `torch.load(..., weights_only=False)`，而 2.6 起默认值改为 True，加载会失败。
- CFT 的 `attempt_download` 在本地没有权重文件时，会下载 GitHub 最新版权重，与这份代码不兼容。`train_rgb_ir` 已改为找不到本地文件就报错。
- CFT 原来的 hyp.scratch.yaml 中 mosaic=1.0；CCSFuse 用的 hyp.ccsfuse_dronevehicle.yaml 中 mosaic=0。
- utils\datasets.py 第 897 行的 `assert (a, b), ...` 恒为真，空数据集不会报错（会打印 SyntaxWarning，不影响运行）。加载时 IR 那一行打印的是 rgb 的 cache 路径（第 937 行的打印错误），实际加载用的是 IR 的 cache。
- 用 PowerShell 删除盘符根目录下的文件夹会被工具的安全规则拦截，需要用户手动删除。
- 在 Python 中读取下载的数据时用 `python -I`，脚本不要放在数据目录里。

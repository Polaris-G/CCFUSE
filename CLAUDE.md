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
  - 数据集类：D:\CCSFuse\utils\datasets.py 中的 `LoadMultiModalImagesAndLabels`
  - 模型配置：D:\CCSFuse\models\transformer\*.yaml
- **原有的 git 跟踪文件均未修改。** 新增文件如下：
  - D:\CCSFuse\check_env.py
  - D:\CCSFuse\requirements-ccsfuse.txt
  - D:\CCSFuse\tools\prepare_dronevehicle.py
  - D:\CCSFuse\tools\prepare_llvip.py
  - D:\CCSFuse\tools\viz_pairs.py
  - D:\CCSFuse\tools\check_loader.py
  - D:\CCSFuse\tools\check_weights.py
  - D:\CCSFuse\data\multispectral\DroneVehicle.yaml
  - D:\CCSFuse\data\multispectral\DroneVehicle_test.yaml
  - D:\CCSFuse\data\multispectral\LLVIP_ccsfuse.yaml
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

### 检查工具（在 D:\CCSFuse 下运行）
- `python tools\check_loader.py --data <yaml>`：用 CFT 加载器读取数据，检查配对、目标数和类别。
- `python tools\viz_pairs.py --root <数据目录> --split train --n 10 --out <文件名.jpg>`：可视化 RGB/IR 配对和标注框。
- `python tools\check_weights.py`：加载权重，打印主干各层的输出形状。

## 当前进度
- 已完成：
  - 环境配置；
  - 克隆代码库；
  - 整理并转换 DroneVehicle、LLVIP，均通过数量核对、可视化和加载器检查；
  - 下载并验证 v5.0 的 s/m 权重。
  - 方案第 7 节第 1 步：双流 YOLOv5 baseline（加法融合），git 分支 `ccsfuse`。
    - 模型配置 models\ccsfuse\yolov5{m,s}_dual_add.yaml（含 `pretrain_map`）；超参 data\hyp.ccsfuse_dronevehicle.yaml。
    - train.py 默认值已改为本机配置：m 版、DroneVehicle、batch 4、`--nbs 12`（累积 3 步）、workers 2、45 epoch；新增 `--hyp-set k=v`、`--pretrain-neck`。
    - 检查：`python tools\check_model.py`（权重迁移、P3/P4/P5 形状、显存、参数量）；`python tools\make_subset.py` 生成子集。
    - 结果：Add 基线 m 版 33.43 M 参数、84.3 GFLOPs（2×MACs），batch 4 训练显存约 3.3 GB；1% 数据过拟合，lr0=0.01 时 loss 0.275→0.104。
- **下一步：方案第 7 节第 2 步**，数据管线（非 Mosaic 分支恢复成对的随机裁剪；HSV 只作用于 RGB，做成可配置）。
- 用户待办：手动删除 D:\CCSFuse_wheels（约 2.4 GB，torch wheel 已安装完毕；该删除操作被工具的安全规则拦截）。

## 踩过的坑（结论）
- CFT 原来的权重加载（intersect_dicts 按键名）只给 RGB 主干加载 COCO 权重，IR 主干（第 10–19 层）是随机初始化。已用 utils\ccsfuse.py 的 `load_coco_dual_stream` 修复。
- 双流模型在 models\yolo_test.py 中，`from: -4` 表示该层读取 IR 输入。
- setuptools 84 没有 pkg_resources，已让 `check_requirements` 跳过。utils\loss.py 的 `clamp_` 在新版 torch 下报 dtype 错误，已修复。
- results.txt 每行 8 个指标依次为：P、R、mAP50、mAP75、mAP50:95、val box/obj/cls。
- lr0=0.001（论文值）时，1% 子集 30 epoch 的 mAP50 只到 0.12，原因是优化步数少（约 450 步），不是 bug；用 lr0=0.01 可以快速过拟合。
- git 没有配置作者身份，提交时用 `git -c user.name=bzp02 -c user.email=...` 临时指定。
- download.pytorch.org、阿里云、上海交大的镜像下载很慢且会断线。torch wheel 用南京大学镜像 https://mirrors.nju.edu.cn/pytorch/whl/cu124/ 下载，并用官方 SHA256 校验（已写在 requirements-ccsfuse.txt 中）。PyPI 用清华镜像。
- numpy 必须 <1.24：代码中用到了 `np.int`。
- 安装依赖要用 requirements-ccsfuse.txt 中写死的版本，加 `--only-binary :all:`。不写死版本时 pip 会长时间回溯。
- torch 不要升级到 ≥2.6：v5.0 的权重是整个模块的 pickle，需要 `torch.load(..., weights_only=False)`，而 2.6 起默认值改为 True，加载会失败。
- **训练时必须显式传入本地 v5.0 权重路径。** 本地文件缺失时，CFT 的 `attempt_download` 会下载 GitHub 最新版权重，与这份代码不兼容。
- CFT 训练时默认开启 Mosaic（hyp['mosaic']）。方案默认关闭 Mosaic，搭 baseline 时需要处理。
- D:\CCSFuse\utils\datasets.py 第 897 行的 `assert (a, b), ...` 恒为真，空数据集不会报错（会打印 SyntaxWarning，不影响运行）。
- 用 PowerShell 删除盘符根目录下的文件夹会被工具的安全规则拦截，需要用户手动删除。
- 在 Python 中读取下载的数据时用 `python -I`，脚本不要放在数据目录里。

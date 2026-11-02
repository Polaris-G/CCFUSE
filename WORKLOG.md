# CCSFuse 复现工作记录

会话时间：2026-10-14 01:33 – 2026-10-15 02:15。按时间顺序记录；所有数字均来自命令输出，并已在 2026-10-15 重新核实。

---

## 1. 检查硬件与已有工具

- **做了什么**：运行 `nvidia-smi`，并检查 conda、git 是否可用。
- **结果**：
  - GPU：RTX 3070 Ti Laptop，8 GB；驱动 616.64，支持 CUDA 13.4。
  - 桌面程序常驻占用 1.0–1.7 GB 显存。
  - git 2.55 可用。
  - conda 不在 PATH 中，在 `D:\Miniconda` 找到了 Miniconda。
- **我替你做的选择**：
  - 项目放在 `D:\CCSFuse`。理由：D 盘剩余 537 GB，数据集较大。
  - conda 一律用完整路径调用，没有改动 PATH。

## 2. 创建 conda 环境 `ccsfuse`

- **做了什么**：
  `D:\Miniconda\Scripts\conda.exe create -n ccsfuse python=3.10 -c conda-forge --override-channels -y`
- **为什么**：
  - Python 3.10 是你指定的版本。
  - 使用 conda-forge 频道，可以避开 Anaconda 默认频道的服务条款（ToS）确认提示。
- **结果**：成功，环境位于 `D:\Miniconda\envs\ccsfuse`，Python 3.10.22。

## 3. 安装 PyTorch（第一次失败）

- **做了什么**：用 pip 从 `download.pytorch.org/whl/cu124` 安装 `torch==2.5.1`、`torchvision==0.20.1`。
- **为什么选这个版本**（替你做的选择）：
  - **cu124**：驱动支持 CUDA 13.4，可以向下兼容运行 cu124。
  - **torch 2.5.1**：从 2.6 起，`torch.load` 默认 `weights_only=True`，会导致 CFT 代码加载 `.pt` 权重时报错；2.5.1 没有这个问题，且与论文使用的 2.0.1 API 兼容。
- **结果**：失败。下载 2.5 GB 的 wheel 时连续出现 5 次 SSL 中断（`UNEXPECTED_EOF_WHILE_READING`），最后报 `WinError 32`（文件被占用）。
- **解决办法**：
  1. 从官方索引获取两个 wheel 的 SHA256。
  2. 测试各下载源的速度：官方源约 0.55 MB/s，阿里云约 0.5 MB/s，上海交大约 0.6 MB/s，**南京大学镜像约 22 MB/s**。
  3. 改用南京大学镜像，用 curl 断点续传下载到 `D:\CCSFuse_wheels`。
  4. 用官方 SHA256 校验通过（`sha256sum -c` 两项均为 OK）。

## 4. 克隆代码库

- **做了什么**：
  `git clone https://github.com/DocF/multispectral-object-detection.git D:\CCSFuse`
- **为什么**：这是方案第 1 节建议的 CFT 官方实现，即双流 YOLOv5（v5.0 时期）。我把它直接克隆为项目根目录。
- **结果**：成功，commit `fb591c9b163177c0e950db08e213e24ddc912d41`（2024-05-19，"Update README.md"）。
- **发现**：
  - 代码中多处使用了 `np.int`，这个写法在 numpy 1.24 中已被删除。
  - 仓库**没有** DroneVehicle 的配置文件，只有 FLIR、LLVIP、VEDAI 的。

## 5. 安装依赖（第一次卡住，第二次成功）

- **第一次**：把本地 wheel、`requirements.txt` 和 `"numpy<1.24"` 放在同一条 pip 命令里安装。
  - **结果**：运行 10 分钟仍未完成。原因是新版 scipy、opencv、pandas 要求 numpy≥1.25，pip 在反复下载旧版本做回溯。
  - **处理**：停止 pip，并结束残留的 pip 进程。
- **第二次**：新建 `D:\CCSFuse\requirements-ccsfuse.txt`，把一组与 numpy 1.23.5 兼容的版本全部写死，加上 `--only-binary :all:`（不从源码编译），使用清华 PyPI 镜像。
- **结果**：成功，退出码 0。最终版本如下：

  | 包 | 版本 |
  |---|---|
  | torch | 2.5.1+cu124 |
  | torchvision | 0.20.1+cu124 |
  | numpy | 1.23.5 |
  | scipy | 1.10.1 |
  | opencv-python | 4.8.1.78 |
  | matplotlib | 3.7.5 |
  | pandas | 2.0.3 |
  | seaborn | 0.12.2 |
  | Pillow | 10.4.0 |
  | PyYAML | 6.0.2 |
  | tqdm | 4.66.5 |
  | tensorboard | 2.14.0 |
  | thop | 0.1.1.post2209072238 |
  | pycocotools | 2.0.8 |
  | protobuf | 7.36.2（随依赖装入；tensorboard 已测试能正常导入） |

## 6. 环境检查脚本 `check_env.py`

- **做了什么**：新建 `D:\CCSFuse\check_env.py`，检查以下内容：
  - 各依赖的版本，以及 numpy 是否 <1.24；
  - CUDA 和 cuDNN 是否可用；
  - 显存：驱动报告的可用量，以及按 256 MB 分块实际申请直到 OOM 的可分配量；
  - AMP：fp16 autocast + GradScaler 跑 5 步训练；
  - CMFC 尺寸（N=6400，M=100）下 ReLU² 注意力分数在 fp16 和 fp32 下的数值；
  - 能否导入 CFT 的 `models.yolo` 和 `utils.datasets`。
- **结果**：全部通过（ALL CHECKS PASSED）。
  - cuDNN 90100，支持 bf16。
  - AMP 5 步训练峰值显存 347 MB。
- **问题**：可分配显存测出 **14.5 GB**，超过物理显存的 8 GB。说明驱动开启了"CUDA 系统内存回退"：显存不够时会悄悄借用系统内存，表现为训练变慢，而不是报 OOM。
- **处理**：
  - 在 check_env.py 中加入检测：可分配量大于物理显存时打印 WARN。
  - 请你在 NVIDIA 控制面板中为 `D:\Miniconda\envs\ccsfuse\python.exe` 关闭该功能。**现已核实关闭**，见末尾"不一致"一节。

## 7. 记忆文件

- 新建 `C:\Users\bzp02\.claude\projects\C--Users-bzp02\memory\ccsfuse-env.md`，并在同目录的 `MEMORY.md` 中加了一行索引。
- 后来又补充了两条：pip 缓存目录、权重与数据路径。

## 8. 清理 C 盘、把 pip 缓存改到 D 盘（应你的要求）

- **核实**：安装过程在 C 盘留下了约 594 MB：
  - pip 缓存 350 MB；
  - `%TEMP%` 中 31 个 `pip-*` 临时文件夹，共 244 MB，是我中途停掉 pip 后残留的。
- **删除前确认**：当前没有 pip 进程在运行；缓存和临时文件夹全部是当天产生的。
- **删除**：
  - 31 个 `pip-*` 临时文件夹，删除后剩余 0 个；
  - 运行 `pip cache purge`，删除 228 个文件，共 366.7 MB。
- **未删除**：`D:\CCSFuse_wheels`。我的删除命令被工具的安全规则拦截（不允许删除盘符根目录下的文件夹），我没有绕过，留给你手动删除。
  - 删除前，我已把下载地址和官方 SHA256 写进 `requirements-ccsfuse.txt`，以便日后重装。
- **pip 缓存目录**：`pip config --user set global.cache-dir D:\pip-cache`，写入 `C:\Users\bzp02\AppData\Roaming\pip\pip.ini`。
- **刻意没有改的**：`TMPDIR`。pip 解包时仍然使用 C 盘的 Temp，正常结束时会自动清理。

## 9. 整理数据集目录（你提供了参考脚本）

- **移动前核实**：
  - DroneVehicle：113,756 个文件，即 4 × 28,439；
  - LLVIP_raw：46,466 个文件。
- **在你的脚本基础上加的保护**：
  - 移动前检查目标位置，已存在就停止；
  - 空文件夹从最里层逐层删除，删除前确认里面没有任何内容；
  - 删除时**不用 `-Recurse`**，万一文件夹不是空的，命令会报错而不会删掉内容；
  - 统计数量时包含隐藏文件。
- **结果**：
  - DroneVehicle 移动后仍为 113,756 个文件；
  - 删除了 7 个空文件夹（`Drone Vehicle` 及其子文件夹）和 1 个中转用的 `LLVIP_tmp`；
  - `LLVIP.zip`（3.73 GB）移到 `D:\CCSFuse\datasets\`，大小不变；
  - `README.md` 随文件夹一起移到了 `LLVIP_raw\LLVIP\` 下。
- **数量核对**：全部一致，没有不匹配。
  - DroneVehicle 的 train/val/test 中，4 个文件夹分别都是 17990 / 1469 / 8980；
  - LLVIP train/test = 12025 / 3463，Annotations = 15488。

## 10. DroneVehicle 转换

### 统计原始数据

只读扫描，脚本放在临时目录。结果如下：

- **尺寸**：全部 840×712。
- **白边**：四边各 100 px，抽样图像的灰度均为 255；第 100 列开始是图像内容。
- **框格式**：polygon 887,905 个，**bndbox 65,184 个**，**point 75 个**。
- **类名**：car / truck / bus / van，以及 `feright car`、`feright_car`、`feright`、`truvk`、`*`。
- **标注**：有 53 个 IR 标注文件没有目标；有坐标落在白边区域内（x 范围 −3 到 843）。

### 查看 CFT 的数据加载器

- 训练 loss 使用的是 `labels_rgb`，即 RGB 一侧的标签。
- RGB 和 IR 是各自排序后按位置配对的。
- 标签路径的约定是把路径中的 `\images\` 替换为 `\labels\`。

### 替你做的选择（论文未写明）

| 项目 | 选择 | 理由 |
|---|---|---|
| 裁剪 | 每边裁掉 100 px，得到 640×512 | 已核实白边宽度正好是 100 px |
| 框格式 | 旋转框取顶点 min/max 得到水平框，再裁剪到图像范围；bndbox 直接使用 | YOLOv5 只支持水平框 |
| 丢弃规则 | point 和 `*` 丢弃；裁剪后宽或高 <2 px 的框丢弃 | point 无法得到框，`*` 类别不明 |
| 标注来源 | 使用 IR 标注，同一份写到 rgb 和 ir 两侧 | CFT 只用 RGB 一侧的标签训练，必须写到 RGB 侧才能真正用上 IR 标注；两侧标签一致也能保证配对不错位 |
| 类别 id | 按论文 Table I 的顺序：car 0，truck 1，freight car 2，bus 3，van 4 | — |
| 图像保存 | JPEG 质量 95 | 裁剪后必须重新编码 |

### 新建的文件

- `D:\CCSFuse\tools\prepare_dronevehicle.py`
- `D:\CCSFuse\tools\viz_pairs.py`
- `D:\CCSFuse\tools\check_loader.py`
- `D:\CCSFuse\data\multispectral\DroneVehicle.yaml`：用 val 做验证
- `D:\CCSFuse\data\multispectral\DroneVehicle_test.yaml`：val 指向 test 划分，因为 CFT 的 `test.py` 只读取 `val_rgb`/`val_ir`

### 验证

1. **试转换**：每个划分转 20 对，手算核对 00001 的第一个框，结果一致。
2. **可视化**：看了 3 对试转换样本和 10 对全量输出样本。白边已去除，框在 IR 图上贴合。部分白天的 RGB 图上框略有偏移，原因是 RGB 和 IR 本身没有对齐（即论文 Fig. 5 所说的现象），不是转换错误。
3. **全量转换结果**：train/val/test = 17990 / 1469 / 8980 对，四个文件夹数量一致，共 6.79 GB。

   | 划分 | car | truck | freight car | bus | van | 合计 |
   |---|---|---|---|---|---|---|
   | train | 270,348 | 15,833 | 11,193 | 11,334 | 7,701 | 316,409 |
   | val | 20,588 | 1,470 | 918 | 789 | 725 | 24,490 |
   | test | 137,145 | 8,657 | 5,062 | 4,467 | 4,282 | 159,613 |

   - 丢弃：point 38 个，`*` 1 个，裁剪后过小 5 个。
   - 空标签 53 个，与原始数据一致。
   - 每个划分的守恒关系都成立：原始框数 − 丢弃数 = 保留数。
4. **CFT 加载器读取**：三个划分配对顺序一致，5 个类别都在，输入张量为 6 通道 640×640。

- **问题**：最后一次加载器检查时会话中断，结果没有记录下来。之后已重新运行，全部通过。

## 11. LLVIP 转换

### 统计原始数据

- 文件名三方一一对应，train 和 test 无重叠。
- 只有 `person` 一个类，42,437 个目标，全部是 bndbox。
- 图像全部 1280×1024，没有越界的框。
- 有 2 个文件没有目标。
- 有 **5 个宽度为 0 的框**：020118、020236、090437、091180、170350，看起来是标注时误点产生的，无法修正。

### 替你做的选择

- 丢弃宽或高 <1 px 的框，即上面这 5 个；
- 图像原样复制，不重新编码；
- 不在离线阶段缩放，交给加载器缩放到 640；
- LLVIP 没有 val 划分，验证和最终测试都用 test。

### 新建的文件

- `D:\CCSFuse\tools\prepare_llvip.py`
- `D:\CCSFuse\data\multispectral\LLVIP_ccsfuse.yaml`：仓库里已有作者的 `LLVIP.yaml`，所以另起文件名，不覆盖原文件。

### 结果

- 手算核对 010001 的第一个框，结果一致；复制后的图像与原图逐字节相同。
- 全量：train/test = 12025 / 3463 对，共 3.77 GB。
- 目标数：34,130 + 8,302 = 42,432，等于 42,437 减去丢弃的 5 个。
- 加载器读取正常，配对顺序一致。
- 可视化了 4 对，正常。

## 12. COCO 预训练权重

- **发现**：如果本地找不到权重文件，CFT 的 `utils\google_utils.py` 中的 `attempt_download` 会下载 GitHub **最新**的 release，而最新版与 v5.0 的代码不兼容。
- **做了什么**：从 ultralytics/yolov5 的 **v5.0** release 下载 `yolov5s.pt` 和 `yolov5m.pt`，保存到 `D:\CCSFuse\weights\`。
- **校验**：文件大小与 GitHub API 返回的官方大小一致（43,098,607 / 14,795,158 字节）。官方没有提供哈希值，所以只能核对大小。
- **新建**：`D:\CCSFuse\tools\check_weights.py`。
- **结果**：两个权重都能正常加载。
  - s 版：7.28 M 参数；m 版：21.38 M 参数。
  - m 版主干的 P3 / P4 / P5 来自第 4 / 6 / 9 层，通道数为 192 / 384 / 768。
  - v5.0 的主干结构：第 0 层是 Focus，第 8 层是 SPP，第 9 层是 C3。这与方案中按 v7.0 写的"第 9 层是 SPPF"不同。

## 13. 本次总结（即本文件和 CLAUDE.md）

用命令核实了磁盘状态：git 状态、文件清单、数据数量、环境版本、`check_env.py` 的输出。之后写了 `D:\CCSFuse\WORKLOG.md` 和 `D:\CCSFuse\CLAUDE.md`。

---

## 文件清单（本次会话新建或修改，已核实存在）

### 代码库内：`D:\CCSFuse`

CFT 原有的、受 git 跟踪的文件**均未修改**（`git status` 中只有未跟踪的新文件）。

| 路径 | 用途 |
|---|---|
| `D:\CCSFuse\`（整个目录） | `git clone` 得到的 CFT 代码库 |
| `D:\CCSFuse\check_env.py` | 环境检查：版本、CUDA、显存、系统内存回退、AMP、fp32 注意力、代码库导入 |
| `D:\CCSFuse\requirements-ccsfuse.txt` | 写死版本的依赖清单，附 torch wheel 的下载地址、SHA256 和安装命令 |
| `D:\CCSFuse\tools\prepare_dronevehicle.py` | DroneVehicle 转换：裁白边、旋转框转水平框、IR 标注写到两侧 |
| `D:\CCSFuse\tools\prepare_llvip.py` | LLVIP 转换：VOC 格式转 YOLO 格式，图像原样复制 |
| `D:\CCSFuse\tools\viz_pairs.py` | 把 RGB/IR 配对和标注框画在一起，便于目视检查 |
| `D:\CCSFuse\tools\check_loader.py` | 用 CFT 的 `LoadMultiModalImagesAndLabels` 读取数据，检查配对、目标数和类别 |
| `D:\CCSFuse\tools\check_weights.py` | 加载 v5.0 权重，打印主干各层的输出形状 |
| `D:\CCSFuse\data\multispectral\DroneVehicle.yaml` | DroneVehicle 训练配置，用 val 做验证 |
| `D:\CCSFuse\data\multispectral\DroneVehicle_test.yaml` | DroneVehicle 最终评估配置，val 键指向 test |
| `D:\CCSFuse\data\multispectral\LLVIP_ccsfuse.yaml` | LLVIP 配置，val 键指向 test |
| `D:\CCSFuse\weights\yolov5s.pt` | COCO 预训练权重，v5.0，调试用 |
| `D:\CCSFuse\weights\yolov5m.pt` | COCO 预训练权重，v5.0，正式训练用 |
| `D:\CCSFuse\WORKLOG.md` | 本文件 |
| `D:\CCSFuse\CLAUDE.md` | 供新会话自动读取的项目事实 |
| `D:\CCSFuse\__pycache__\`、`models\__pycache__\`、`utils\__pycache__\` | 运行 Python 时自动生成的字节码缓存 |

### 数据：`D:\CCSFuse\datasets`

| 路径 | 用途 |
|---|---|
| `D:\CCSFuse\datasets\DroneVehicle_raw\` | 原始 DroneVehicle：由你下载，我整理了目录层级 |
| `D:\CCSFuse\datasets\LLVIP_raw\LLVIP\` | 原始 LLVIP：由你下载，我整理了目录层级 |
| `D:\CCSFuse\datasets\LLVIP.zip` | LLVIP 原始压缩包，从 `LLVIP_raw\LLVIP\` 移到这里，尚未删除 |
| `D:\CCSFuse\datasets\DroneVehicle\{rgb,ir}\{images,labels}\{train,val,test}\` | 转换后的 DroneVehicle |
| `D:\CCSFuse\datasets\DroneVehicle\convert_stats.json` | DroneVehicle 转换统计 |
| `D:\CCSFuse\datasets\DroneVehicle\viz_train_10pairs.jpg` | 10 对 train 样本的可视化 |
| `D:\CCSFuse\datasets\LLVIP\{rgb,ir}\{images,labels}\{train,test}\` | 转换后的 LLVIP |
| `D:\CCSFuse\datasets\LLVIP\convert_stats.json` | LLVIP 转换统计 |
| `D:\CCSFuse\datasets\LLVIP\viz_train_4pairs.jpg` | 4 对 train 样本的可视化 |
| `D:\CCSFuse\datasets\{DroneVehicle,LLVIP}\{rgb,ir}\labels\*.cache`（10 个） | CFT 加载器自动生成的标签缓存；数据有变动时会自动重建 |
| `D:\CCSFuse\datasets\convert_dronevehicle.log` | DroneVehicle 全量转换日志 |
| `D:\CCSFuse\datasets\convert_llvip.log` | LLVIP 全量转换日志 |

### 代码库以外

| 路径 | 用途 |
|---|---|
| `D:\Miniconda\envs\ccsfuse\` | conda 环境，约 5.1 GB |
| `D:\CCSFuse_wheels\` | torch / torchvision 的 wheel、SHA256SUMS、pip_install.log，约 2.4 GB，**待你手动删除** |
| `D:\pip-cache\` | 新的 pip 缓存目录，目前为空 |
| `C:\Users\bzp02\AppData\Roaming\pip\pip.ini` | pip 用户配置，内容为 `cache-dir = D:\pip-cache` |
| `C:\Users\bzp02\.conda\environments.txt` 等 3 个小文件 | conda 自动写入的环境登记文件，不是我手动创建的 |
| `C:\Users\bzp02\.claude\projects\C--Users-bzp02\memory\ccsfuse-env.md` | Claude 的记忆：环境中的坑和路径 |
| `C:\Users\bzp02\.claude\projects\C--Users-bzp02\memory\MEMORY.md` | Claude 记忆索引，新增一行 |
| `C:\Users\bzp02\AppData\Local\Temp\claude\...\scratchpad\survey_dv.py`、`survey_dv2.py`、`survey_llvip.py` | 一次性只读统计脚本，放在会话临时目录 |
| `C:\Users\bzp02\AppData\Local\Temp\claude\...\scratchpad\dv_try\`、`llvip_try\`、`viz_try.jpg` | 试转换输出，放在会话临时目录，可随时删除 |

---

## 会话记录与磁盘实际状态的不一致（以磁盘为准）

1. **系统内存回退已关闭**。会话中我一直把它列为"待你处理"；2026-10-15 重新运行 `check_env.py`，可分配量为 7.25 GB，小于 8 GB，且没有 WARN，说明已经关闭。
2. **`CCSFuse_复现方案.md` 已在 `D:\CCSFuse\` 下**（文件时间 2026-10-13 23:01）。会话中我曾把"复制方案文件"列为待办，实际已经存在。
3. **C 盘 pip 缓存目录没有完全清空**。会话中我说已经清理干净，实际还剩 `C:\Users\bzp02\AppData\Local\pip\cache\selfcheck\` 下的 1 个文件（不到 1 KB，写入时间 2026-10-14 02:01）。这是 pip 记录版本检查的文件，`pip cache purge` 不会删除它，不影响使用。
4. **DroneVehicle 转换后的大小**：会话中报告为 6.75 GB，现在是 6.79 GB。多出的部分是之后运行加载器检查时生成的 6 个 `.cache` 文件，约 34 MB。
5. **`D:\CCSFuse_wheels` 仍然存在**，与会话记录一致，仍是待办，不算不一致，只是提醒。

---

# 会话 2（2026-10-15）

会话时间：2026-10-15 约 02:16 – 04:05。时间取自本机时钟，与会话 1 相同；git 和 shell 显示的时区为 -0400。按时间顺序记录，数字均来自命令输出。

## 1. 读方案，核实开工前的四项（02:16 – 02:21）

- **做了什么**：
  - 读了 CCSFuse_复现方案.md、CLAUDE.md、WORKLOG.md；
  - 阅读 CFT 的 train.py（`train_rgb_ir`）、models\yolo_test.py、utils\datasets.py；
  - 写了两个只读探查脚本，放在会话临时目录：`probe_transfer.py`、`probe_aug.py`。
- **结果**：
  1. **预训练权重**：train.py 实际导入的是 models\yolo_test.py 的 Model，yaml 中 `from: -4` 表示读 IR 输入。CFT 的 intersect_dicts 按键名匹配，结果 Transferred 306/812：RGB 主干 306/306，**IR 主干 0/306**，neck 0/200。原因是 IR 主干的键 `model.10`–`model.19` 在 COCO 权重中属于 neck，形状对不上。
  2. **增强**：
     - Mosaic、仿射变换、翻转对两路同步。实测 Mosaic 加仿射后两路灰色填充区域重合 99.1%（差的部分是图像中本来就有的 114 灰度像素）；fliplr=1 时两路都与镜像后的原图逐像素一致。
     - HSV 对 IR 也起作用，且两路的随机增益各自独立。实测 IR 亮度被缩放到原来的 0.67–1.11 倍。
     - **关闭 Mosaic 后没有随机裁剪**：非 Mosaic 分支的 random_perspective 被注释掉了，实测两次读取结果相同。
  3. **规模**：仓库只有 s 和 l 两个规模的 add 配置。临时建 m 版双流模型（depth 0.67、width 0.75、nc=5）得到 33.43 M 参数，接近论文的 33.50 M。
  4. **超参**：与论文不一致。原值为 lr0 0.01、momentum 0.937、nesterov True、mosaic 1.0、epochs 100、batch 16、workers 8，梯度累积按 nbs=64 计算。
- **额外发现**：train.py 的优化器分组只收集模块的 `.weight` 和 `.bias`。CMFC 的 λ 是单独定义的 nn.Parameter，收集不到，第 4–6 步需要处理。
- 向你汇报后暂停，你回复"继续"。

## 2. 基线提交（02:21 – 02:22）

- **替你做的选择**：
  - 新建 `.gitignore`，忽略 `__pycache__`、`datasets/`、`weights/*.pt`、`runs/`、`*.cache`。原仓库没有 `.gitignore`。
  - 新建分支 `ccsfuse`，`main` 保持 CFT 原样，便于对比和回退。
  - 本机 git 没有配置作者身份。我没有修改全局配置，只在这次提交中用 `-c user.name=bzp02 -c user.email=<你的账号邮箱>` 临时指定。
- **结果**：提交 cba1f45，包含会话 1 新建的文件和三个文档。

## 3. 第 1 步：双流 YOLOv5m Add 基线（02:23 – 02:45）

- **新建配置**：
  - models\ccsfuse\yolov5m_dual_add.yaml 和 s 版：结构与 CFT 的 add 配置相同，改为 nc=5，新增 `pretrain_map`；
  - data\hyp.ccsfuse_dronevehicle.yaml。
- **新建 utils\ccsfuse.py 的 `load_coco_dual_stream`**：把 COCO 第 0–9 层同时复制到 RGB 主干（第 0–9 层）和 IR 主干（第 10–19 层）。
- **修改 train.py**：
  - 权重加载：checkpoint 的层数与模型不同时，走新的映射逻辑；
  - 本地没有权重文件时直接报错；
  - 新增 `--nbs`、`--hyp-set`、`--pretrain-neck`；
  - 默认值改为本机配置。
- **替你做的选择**（论文未给出，均已在 yaml 或代码中标注"推断"）：

  | 项目 | 取值 | 理由 |
  |---|---|---|
  | lrf | 0.01 | YOLOv5 余弦退火到 lr0×lrf，视作 One-Cycle |
  | nesterov | False | 论文只写 SGD，没写 Nesterov |
  | warmup_bias_lr | 0.01 | 保持 YOLOv5 "bias 初始 lr 是 lr0 的 10 倍"的比例 |
  | weight_decay、warmup 3 epoch、HSV、translate 0.1、scale 0.5 | YOLOv5 默认值 | — |
  | neck 是否加载 COCO 权重 | 默认不加载（`--pretrain-neck` 可开启） | 论文只说主干用 COCO 预训练 |
  | 梯度累积 | `--nbs 12`，即 batch 4 累积 3 步，等效 batch 12 | 对应论文的 6/GPU × 2 |
  | workers | 2 | Windows |
  | 1% 子集 | seed 0，180 对；train 和 val 指向同一份 | 用于过拟合检查 |

- **遇到的问题**：
  1. **CFT 的 bug**：utils\loss.py 第 211 行 `clamp_(0, gain[3] - 1)` 在 torch 2.5 下报 int64/float 类型错误，loss 无法计算。改为 `int(gain[3]) - 1`。
  2. **编码**：新 yaml 有中文注释，CFT 用 `open()` 读取时没有指定编码，中文 Windows 默认按 GBK 解码。改为 UTF-8 读取，涉及 train.py（hyp 和 data）、test.py、yolo_test.py。
  3. **我自己脚本的 bug**：tools\check_model.py 中 forward hook 有返回值，替换了模块输出；thop 统计时又复制了 hook。均已修正，不影响训练代码。
  4. **第一次启动训练失败**：setuptools 84 已删除 pkg_resources，`check_requirements` 导入失败。我没有降级 setuptools，而是让这个检查在导入失败时跳过，因为环境已由 requirements-ccsfuse.txt 和 check_env.py 保证。
  5. **最后一个 epoch 画图报错**：test.py 把 6 通道图像传给 plot_images。报错发生在后台线程，不中断训练。改为只传 RGB 三通道。
- **结果**：
  - check_model.py：
    - Transferred 612/812（rgb 306/306，ir 306/306），两路 Focus 权重都与 COCO 相同；
    - P3/P4/P5 为 (4,192,80,80) / (4,384,40,40) / (4,768,20,20)；
    - Detect 为 (4,3,{80,40,20},{80,40,20},10)；
    - loss 和梯度均为有限值；
    - 峰值显存 2.24 GiB（这是 max_memory_allocated）；
    - 33.43 M 参数，84.32 GFLOPs（按 2×MACs 计）。
  - 1% 过拟合（30 epoch，batch 4，`--hyp-set warmup_epochs=1 warmup_min_iters=45`）：

    | 运行 | 学习率 | loss | mAP50 | 用时 |
    |---|---|---|---|---|
    | runs\train\overfit_add_m | lr0=0.001 | 0.2736→0.1772 | 0.119 | 0.113 h |
    | runs\train\overfit_add_m_lr01（诊断用） | lr0=0.01 | 0.2749→0.1037 | 0.472，car 类 0.97 | 0.126 h |

    - 训练时显存（日志中的 memory_reserved）为 3.32 GB；日志显示 Scaled weight_decay = 0.0005，Using 2 dataloader workers。
  - **结论**：管线能够学习。论文 lr 下进展慢，是因为优化步数只有约 450 步。
- **提交**：87d523e。

## 4. 你在本地验证（约 02:45 – 04:00）

- 你说找不到 tools\check_model.py。
  - 我核实：文件存在，分支为 ccsfuse，文件已提交。
  - 推测原因：终端不在 D:\CCSFuse，或资源管理器没有刷新，或切到了 main 分支。
  - 我给出了先 `cd D:\CCSFuse` 再运行的命令。
- 你运行后，我看不到你终端的输出，于是在本地重跑了一次，供你对照。结果与上面一致；loss 每次取随机增强样本，这次为 4.311。

## 5. 收尾：核实磁盘状态，更新文档（04:00 – 04:05）

- **核实**：
  - git 在 ccsfuse 分支，没有未提交的改动；
  - 没有 python 进程，也没有后台训练；
  - runs 下两个过拟合目录都有 30 行 results.txt，以及 best.pt 和 last.pt。
- **新发现**：D:\CCSFuse\.venv（见下文"不一致"第 1 条）。
- **更新了 CLAUDE.md**，并追加本节。

---

## 本次新建或修改的文件

| 路径 | 类型 | 用途 |
|---|---|---|
| `.gitignore` | 新建 | 忽略缓存、数据、权重、runs |
| `models\ccsfuse\yolov5m_dual_add.yaml` | 新建 | 双流 YOLOv5m + Add 基线，含 pretrain_map |
| `models\ccsfuse\yolov5s_dual_add.yaml` | 新建 | 同上，s 版，调试用 |
| `data\hyp.ccsfuse_dronevehicle.yaml` | 新建 | 论文超参，推断项已标注 |
| `utils\ccsfuse.py` | 新建 | load_coco_dual_stream：COCO 主干权重复制到两路 |
| `tools\check_model.py` | 新建 | 第 1 步检查：权重迁移、形状、loss、显存、参数量、GFLOPs |
| `tools\make_subset.py` | 新建 | 生成配对子集的 txt 列表和 data yaml |
| `data\multispectral\DroneVehicle_1pct.yaml`、`DroneVehicle_1pct_train_{rgb,ir}.txt` | 新建 | 1% 子集（180 对） |
| `train.py` | 修改（CFT 原有） | 双流权重映射、禁止自动下载、--nbs/--hyp-set/--pretrain-neck、nesterov 和 warmup 最少步数可配置、默认值、UTF-8 |
| `test.py` | 修改（CFT 原有） | 画图只传 RGB；UTF-8 |
| `models\yolo_test.py` | 修改（CFT 原有） | 模型 yaml 按 UTF-8 读取 |
| `utils\loss.py` | 修改（CFT 原有） | clamp_ 的 dtype 修复 |
| `utils\general.py` | 修改（CFT 原有） | 没有 pkg_resources 时跳过依赖检查 |
| `CLAUDE.md`、`WORKLOG.md` | 修改 | 文档 |
| `runs\overfit_add_m.log`、`runs\overfit_add_m_lr01.log`、`runs\train\overfit_add_m*\` | 运行产物 | 两次过拟合的日志、曲线、权重（git 忽略） |
| `data\multispectral\DroneVehicle_1pct_train_{rgb,ir}.cache` | 运行产物 | 加载器生成的标签缓存（git 忽略） |
| 会话临时目录 `scratchpad\probe_transfer.py`、`probe_aug.py` | 临时文件 | 第 1 节的只读探查脚本，不在项目中 |

## 会话记录与磁盘实际状态的不一致（以磁盘为准）

1. **D:\CCSFuse\.venv 不是本会话创建的**。
   - 磁盘上的情况：创建时间 2026-10-15 03:56，uv 的 Python 3.15，14 MB，没有 torch。
   - 推测：时间正好在你打开 check_model.py 之后，很可能是 VS Code 的 Python 扩展自动创建的。
   - 它带有 `*` 的 .gitignore，git 不跟踪。
   - **处理结果**：你确认它是 VS Code 提示创建的，同意删除。删除前我核实了：它只含 Python 3.15 和 pip，共 894 个文件，没有 torch，也没有你的文件；项目中没有任何配置引用它；没有 python 进程在运行。随后用 `rm -rf ./.venv` 删除，其他目录不受影响。
2. **CLAUDE.md 原来写的"原有的 git 跟踪文件均未修改"已经过时**：本次修改了 5 个 CFT 原有文件，已在 CLAUDE.md 中更正。
3. **后台监视任务显示 "stopped / 无完成记录"**：这是我用来查看训练进度的监视任务（bw0osy5hw）。两次训练实际都已完成：各 30 行 results.txt，日志末尾有 "30 epochs completed"，best.pt 和 last.pt 都在。
4. **两个显存数字口径不同**：
   - 第 1 步汇报中说"训练显存约 3.3 GB"，取自训练日志的 memory_reserved（3.32G）；
   - check_model.py 报告的 2.24 GiB 是 max_memory_allocated。
   - 两者都没错，只是统计口径不同。

## 会话 3：第 2 步，数据管线（2026-10-15）

### 改动
| 文件 | 类型 | 内容 |
|---|---|---|
| `utils\datasets.py` | 修改（CFT 原有） | `LoadMultiModalImagesAndLabels.__getitem__` 非 Mosaic 分支恢复 `random_perspective_rgb_ir`（两路共用同一矩阵，translate/scale 来自 hyp，推断）；新增 hyp `hsv_ir: none\|independent\|same`（缺省时为 independent，保持 CFT 行为）；`augment_hsv` 新增 `gains=` 参数并返回所用增益 |
| `data\hyp.ccsfuse_dronevehicle.yaml` | 修改 | 新增 `hsv_ir: none`（推断） |
| `train.py` | 修改 | `--hyp-set` 允许 `hsv_ir` |
| `tools\check_aug.py` | 新建 | 增强可视化（左 RGB 右 IR，带框）和三项数值检查 |

### 验证
`python tools\check_aug.py --n 10 --out runs/viz/aug_pairs.jpg`：
- [1] 同一样本读两次，10/10 不同；增强全关时 10/10 相同；
- [2] hsv_ir=none：IR 与"只做几何变换"的 IR 10/10 逐像素相同；RGB 10/10 被 HSV 改变；
- [3] RGB 的灰边在 IR 中也是灰边：覆盖率最小 0.9998（6 个样本，4 个放大后无灰边的样本跳过）。
  - 最初用 IoU 作指标，最小值为 0。原因是放大样本没有灰边，且 IR 灰度图中本来就有大量值为 114 的像素，并不是错位。
- 可视化图：两路裁剪、缩放、翻转一致，框贴合 IR 目标；RGB 上少数框略偏，来自 DroneVehicle 本身的 RGB/IR 配准误差（标签取自 IR）。

### 1% 子集过拟合复跑（lr0=0.01，30 epoch，warmup_epochs=1，warmup_min_iters=45）
runs\train\overfit_add_m_lr01_aug，日志 runs\overfit_aug.log：
| | loss 首→末 | mAP50 | mAP50:95 |
|---|---|---|---|
| 无增强（第 1 步） | 0.275→0.104 | 0.472 | 0.228 |
| 新增强 | 0.269→0.127 | 0.456 | 0.211 |
- loss 持续下降；加了增强后训练集更难拟合，末尾 loss 略高，符合预期。30 epoch 共 0.107 h，显存 3.45G（reserved）。

## 会话 3：断点续训测试（2026-10-15）

方法：1% 子集，5 epoch（lr0=0.01，warmup_epochs=1，warmup_min_iters=45），跑完 2 个 epoch 后用 taskkill /T /F 强制结束（模拟睡眠或崩溃），再执行 `python train.py --resume runs\train\resume_test\weights\last.pt`。

结果：**续训可用**。
- 从 epoch 2 接着跑，"3 epochs completed"，results.txt 共 5 行，退出码 0；
- `--hyp-set` 的覆盖值（lr0=0.01、hsv_ir=none）通过 opt.yaml 保留下来；Transferred 812/812，优化器、EMA、调度器都已恢复；
- 学习率：续训后第 e 个 epoch 的学习率是 lr0×lf(e−1)，看上去慢了一个 epoch。但在不中断的 30 epoch 训练（overfit_add_m_lr01_aug）中，从 epoch 2 起也是 lf(e−1)。
  - 原因：train.py 第 684 行 `scheduler.last_epoch = start_epoch - 1`，加上新版 torch 的 LambdaLR 在创建时已经执行过一次 step，所以 scheduler.step() 总是滞后一个 epoch。epoch 1 恰好由 warmup 的最后一步（ni=nw）设成 lf(1)，因此看不出来。
  - 这是 YOLOv5/CFT 原有的行为，续训与不中断训练一致，**保持不改**。影响：最后一个 epoch 的学习率是 lf(E−2) 而不是 lf(E−1)，对 45 epoch 的训练影响很小。
- 测试产物（resume_test、resume_test2 目录和日志）已删除。

## 会话 3：第 3 步，完整训练集基线（2026-10-15）

训练由用户在 VS Code 终端运行：`python train.py --name baseline_add_m`（默认配置，45 epoch，batch 4，nbs 12，lr0 0.001）。04:38→12:24，约 7.8 h，每 epoch 约 10 分钟（5.5 it/s）。
- val 集 mAP50：0.072（epoch 0）→ 0.670（epoch 44），到最后仍在缓慢上升（每 epoch +0.001~0.002），没有过拟合。

test 集评估（我运行，约 16 分钟，主要耗时在 test.py 默认写出 8979 个 txt）：
`python test.py --weights runs/train/baseline_add_m/weights/best.pt --data data/multispectral/DroneVehicle_test.yaml --batch-size 8 --iou-thres 0.6 --task test --verbose --name baseline_add_m_test`
- 注意：test.py 命令行的 `--iou-thres` 默认是 0.5（CFT 改的），训练时的验证和 YOLOv5 标准都用 0.6，这里显式指定 0.6。

| | mAP50 | car | truck | freight car | bus | van |
|---|---|---|---|---|---|---|
| 本复现 baseline（test） | **67.3** | 97.7 | 68.8 | 46.6 | 94.3 | 28.8 |
| 论文 Table I 第 1 行 | 72.8 | – | – | – | – | – |
| 论文 +CMFC（参考） | 76.8 | 96.5 | 68.8 | 64.8 | 93.2 | 60.6 |

- 差距 5.5 点，**全部来自 freight car 和 van**；car、truck、bus 与论文 +CMFC 行持平甚至更高。
- 混淆矩阵（runs\train\baseline_add_m\confusion_matrix.png，val 集）：van 51% 被判成 car（只有 20% 判对，召回率 0.17）；freight car 42% 被判成 truck（35% 判对）。漏检比例低，说明问题出在分类，不在定位。
- 其他：mAP50:95 0.456，推理速度 11.3 ms/张（batch 8）。

## 会话 3：第 4 步，CMFC 实现与单元测试（2026-10-15）

新增 `models\ccsfuse_modules.py`（CMFC）和 `tools\test_cmfc.py`。不改 common.py；文件名不用 models\ccsfuse.py，因为 models\ccsfuse\ 已经是 yaml 目录。
- 结构（式 1–6）：每个模态各一套 Q/K/V（论文：W_rgb ≠ W_ir）；Q 用全分辨率 token，K/V 用经 SR 缩减后的 token；scores = QKᵀ/√d；输出 = w_SIC·Softmax(scores) + w_RIS·ReLU(scores)²，再乘 V；(w_SIC, w_RIS) = softmax(λ1, λ2)，λ 初始化为 1；最后加残差 f。
- 推断项均为构造参数：`num_heads=8`（可设 'auto' = C/64）、`sr_ratio`（方案默认 8/4/2）、`out_proj=True`、`share_lambda=True`、`ris_norm=False`、`attn_fp32=True`。SR = Conv(k=r, s=r) + LayerNorm（PVT 写法）。
- forward 接受 `(f_rgb, f_ir)` 或 `[f_rgb, f_ir]`（YOLO 图中的调用方式），返回 `(f_rgb_cmfc, f_ir_cmfc)`。接入网络（第 6 步）时：回注方式 A 需要一个按下标取出的模块；方式 B 正好对应 CFT 已有的 Add2。
- λ 是直接定义的 nn.Parameter，train.py 的 pg0/pg1/pg2 收集不到，第 6 步要单独加进优化器。

`python tools\test_cmfc.py` 全部通过：
- 形状：(2,192,80,80)、(2,384,40,40)、(2,768,20,20) 输入输出形状一致；
- 初始化 w=0.5/0.5；share_lambda 开关生效；
- λ=[50,−50]（w_SIC≈1）时，与 F.scaled_dot_product_attention 写的普通交叉注意力一致（相对误差 9e-8）；λ=[−50,50] 时与手写的 ReLU²·V 一致；
- AMP fp16 下前向和反向都是有限值（attn_fp32 开或关、输入放大 30 倍都测过）；
- 只对 RGB 输出求梯度时，IR 输入和 λ 也有梯度；
- 训练规模（batch 4，三个 stage，AMP）下前向加反向额外显存约 1.1 GiB。

### 参数量矛盾（重要，待用户决定）
3 个 CMFC（m 版，方案默认配置）共 **20.37 M** 参数：SR 卷积约 14.2 M（c²r²，三个 stage 各约 2.36 M，两路各一套），Linear 约 6.2 M。
| 规模 | 基线 | +CMFC（卷积 SR + 投影） | +CMFC（SR 无参数、无投影） |
|---|---|---|---|
| s | 11.29 M | 20.34 M | 13.36 M |
| m | 33.43 M | 53.79 M | 38.08 M |
（以上都还没算 AFSF：m 版约 4–5 M，s 版约 2 M。）
论文 CCSFuse 为 33.50 M、174.12 GFLOPs。m 基线已有 33.43 M，留给 CMFC 和 AFSF 的只有约 0.07 M，而一个 768×768 的 Linear 就有 0.59 M。GFLOPs 粗算：m 基线 84，加 CMFC 后约 94，离 174 也很远。**论文的效率数字无法和任何合理配置同时吻合，不能用来反推 CMFC 的结构。**

## 会话 3：第 5 步，AFSF 实现与单元测试（2026-10-15）

`models\ccsfuse_modules.py` 中新增 `AFSF`，测试为 `tools\test_afsf.py`。
- 式 (7)：w_m = Sigmoid(Conv1×1(GAP(f_m)))，RGB 和 IR 各一个独立的 1×1 卷积（C→C，带 bias，推断）；
- 式 (8)：拼接成 B×2C，经过 FC1（2C→2C/reduction）、GELU、FC2（→2C），再按通道切成两半，前 C 个为 w̃_rgb，后 C 个为 w̃_ir（推断）；
- 式 (9)：f_m × w̃_m；
- 式 (10)：YOLOv5 Conv（Conv+BN+SiLU）把 2C 压到 C（推断）。
- 推断项均为构造参数：`reduction=4`、`sigmoid_after_mlp=False`（默认忠实原文，即 MLP 后不加激活，权重可以为负）。

`python tools\test_afsf.py` 全部通过：
- 三个 stage 形状一致；
- 用模块自身的各层手算式 (7)–(10)，与输出一致（误差 4.7e-6）；
- 默认设置下 w̃ 在初始化时的范围约为 [−0.30, 0.29]（会出现负值，并会把特征幅值压小，由 fuse 中的 BN 重新归一）；`sigmoid_after_mlp=True` 时权重在 (0,1) 内；
- 梯度能回传到两路输入和全部参数；AMP 下（输入放大 30 倍）结果为有限值；
- 三个 stage 的 CMFC→AFSF 串联都能跑通，梯度能回传到两路输入和 λ。
- 参数量：3 个 AFSF 共 4.65 M；m 版 CCSFuse 合计约 **58.45 M**（基线 33.43 + CMFC 20.37 + AFSF 4.65），论文为 33.50 M（用户已决定接受这一差异，见第 4 步）。

## 会话 3：第 6 步，接入网络（2026-10-15）

改动：
- `models\ccsfuse_modules.py`：CMFC 新增 `branch_weights`（None 为可学习 λ；给定 (w_SIC, w_RIS) 时权重固定且不创建 λ，用于 Table II）；新增 `CMFCInject`（回注方式 A 为 f^CMFC，B 为 stage 输出 + f^CMFC）和 `PairAdd`（把一对特征相加，用于 +CMFC 消融）。
- `models\yolo_test.py`：parse_model 注册 CMFC/CMFCInject/AFSF/PairAdd；推断项从 yaml 顶层的 `cmfc:`、`cmfc_inject:`、`afsf:` 读取，每层只写 sr_ratio 或分支下标；模块构造支持 kwargs。
- `train.py`（train_rgb_ir）：LayerNorm 的 weight 归入不衰减组（推断）；没被 pg0/1/2 收集的参数（CMFC λ）补进不衰减组，日志打印 "Optimizer: N extra parameter tensors"。
- `tools\check_model.py`：融合类型默认加上 PairAdd；新增 flop_counter GFLOPs（thop 统计不到 CMFC 内部的矩阵乘法）；打印 CMFC 分支权重。
- 新 yaml（models\ccsfuse\）：`yolov5m_ccsfuse.yaml`（完整版，Table I 第 4 行）、`yolov5m_cmfc_add.yaml`（第 2 行）、`yolov5m_afsf.yaml`（第 3 行）。完整版主干按 stage 交错：RGB s3、IR s3、CMFC_3、两路 Inject、RGB s4 ……；AFSF 直接取 CMFC 输出的一对特征；pretrain_map 为 rgb [0-4,13,14,20,21,22]、ir [5-9,15,16,23,24,25]、neck [30..43]。

check_model.py（batch 4，AMP，真实 batch）：
| 配置 | 参数量 | GFLOPs（thop / flop_counter） | 峰值显存 |
|---|---|---|---|
| 基线 Add | 33.43 M | 84.3 / 83.7 | 2.24 GiB |
| +CMFC | 53.79 M | 93.4 / 94.5 | 3.37 GiB |
| +AFSF | 38.08 M | 87.2 / 86.5 | 2.34 GiB |
| 完整版 | 58.45 M | 96.3 / 97.3 | 3.45 GiB |
- 四个配置都是 rgb 306/306、ir 306/306；融合层和 Detect 形状正确；loss 和梯度有限；λ 初始为 0.5/0.5。

1% 子集过拟合，完整版（lr0=0.01，30 epoch，warmup 1 epoch / 45 步），runs\train\overfit_ccsfuse_m：
- 日志显示 "3 extra parameter tensors"，即 3 个 λ 进了优化器；COCO 权重迁移正常；
- loss 0.273→0.125（基线同设置为 0.269→0.127）；mAP50 0.384（基线 0.456；新增约 2500 万随机初始化参数，小数据上偏慢，不作为判断依据）；
- 训练后 λ：CMFC_3 w_SIC/w_RIS = 0.54/0.46，CMFC_4 0.51/0.49，CMFC_5 0.50/0.50，说明 λ 在更新；
- 速度：0.171 h，基线 0.107 h，约慢 1.6 倍；显存 reserved 5.18 G。推算完整训练：完整版约 12.5 h，+CMFC 约 12 h，+AFSF 约 8–8.5 h。

## 会话 3：完整版显存与 batch 的关系（2026-10-18，用于评估服务器配置）

check_model.py（完整版 yolov5m_ccsfuse，AMP，前向加反向峰值 allocated）：batch 2 / 4 / 6 = 1.90 / 3.45 / 5.00 GiB，线性：约 0.35 + 0.775 × batch GiB。
- 训练时还要加上 SGD momentum 和 EMA（各约 0.22 GiB，共 0.44 GiB）；PyTorch 缓存使 reserved 比 allocated 多 10–35%（本机 batch 4 实测 reserved 4.29–5.18G）。
- 预测（AMP）：论文设置（每卡 6，2 卡）→ 每卡 allocated 约 5.4 GiB，reserved 约 6–7.5 GB；单卡 batch 12 → allocated 约 10.1 GiB，reserved 约 11–13.5 GB。不开 AMP 时激活显存约多 1.5–2 倍。

## 会话 3：第 7 步，完整版 CCSFuse 结果（2026-10-20）

训练：用户在 VS Code 中运行 `train.py --cfg models/ccsfuse/yolov5m_ccsfuse.yaml --name ccsfuse_m`，45 epoch，于 2026-10-16 10:37 完成。
test 集评估（我运行，21 分钟）：runs\test\ccsfuse_m_test，日志 runs\ccsfuse_m_test.log。

| | mAP50 | car | truck | freight car | bus | van | mAP50:95 |
|---|---|---|---|---|---|---|---|
| 基线 Add（test） | 67.3 | 97.7 | 68.8 | 46.6 | 94.3 | 28.8 | 45.6 |
| **CCSFuse 完整版（test）** | **66.1** | 97.6 | 68.9 | 42.3 | 94.4 | 27.4 | 44.8 |
| 差值 | **−1.2** | −0.1 | +0.1 | −4.3 | +0.1 | −1.4 | −0.8 |
| 论文差值（79.4 − 72.8） | +6.6 | | | | | | |

- val 曲线：前 16 个 epoch 与基线持平，从 epoch 20 起落后，最后 0.646 对 0.670（−2.4）。训练 loss 一直略低于基线，说明拟合训练集更好、泛化更差。
- λ 几乎没学动：w_SIC/w_RIS 在 stage3/4/5 分别为 0.555/0.445、0.509/0.491、0.484/0.516。

### 诊断：CMFC 在初始化时就严重扰乱了预训练特征
用 4 张 val 图测 ‖Δ‖/‖f‖（Δ 为 CMFC 加到特征上的补偿量）：
| CMFC | COCO 初始化时 rgb / ir | 训练后 rgb / ir |
|---|---|---|
| stage3 | 0.59 / 0.50 | 0.52 / 1.19 |
| stage4 | 0.58 / 0.48 | 0.56 / 0.45 |
| stage5 | 0.65 / 0.66 | 0.66 / 0.34 |
- 回注方式 A 下，补偿量直接成为 COCO 预训练 stage4/5 的输入。随机初始化的输出投影从第一步起就加入了相当于原特征 50–66% 的扰动，预训练特征被破坏。
- 两个分支的权重各占一半，但 ReLU² 分支的输出幅值普遍大于 Softmax 分支（没有归一化）。
- 候选改法（都是论文未给出的实现细节，需要重训约 13 h）：(1) 输出投影零初始化，使 CMFC 初始为恒等映射（推荐）；(2) `ris_norm: True` 平衡两个分支的幅值。

## 会话 3：方案 1，CMFC 输出投影零初始化（2026-10-20）

- `CMFC(proj_init='default'|'zero')`（推断）：设为 zero 时，proj_rgb/proj_ir 的 weight 和 bias 都初始化为 0，CMFC 初始为恒等映射。构造函数默认仍是 default，ccsfuse_m 的权重不受影响。
- yaml：完整版和 +CMFC 的 `cmfc:` 中加入 `proj_init: zero`。
- test_cmfc.py 新增一项：初始时输出 == 输入；proj 立即有梯度，Q/K/V 在一次更新后有梯度。全部通过。
- 真实网络（COCO 初始化）中三个 CMFC 的 ‖Δ‖/‖f‖ = 0.00。
- 1% 子集过拟合（runs\train\overfit_ccsfuse_m_zinit）：loss 0.274→0.125，mAP50 0.382，与之前的 0.384 相当，零初始化不妨碍学习。注意：这次是在被意外改动的 train.py 上跑的（见下），λ 没有参与训练，只作为能否正常学习的检查。

### train.py 被意外改动（已恢复）
- 2026-10-17 01:26，train.py 在工作区被改动（用户确认不是有意操作，推测是 VS Code 中误操作或撤销过头）：第 6 步的优化器分组代码被撤回（λ 不进优化器、LayerNorm 被施加 weight decay），注释中还多了两个空格（`# SyncBa  tchNorm`）。
- 不影响 ccsfuse_m：它在 10-15 启动、10-16 结束，启动日志中有 "3 extra parameter tensors"，λ 也从 1.0 变成了 1.11/0.889 等值，确实参与了训练。
- 用 `git checkout -- train.py` 恢复后，1 epoch 检查中日志重新出现 "3 extra parameter tensors"，分组为 188/182/146。

## CMFC 开启实验配置 v2（2026-10-29）
- 原因：best.pt 中 proj mean|w| 约 2e-4~3e-3，lambda softmax 仍约 0.5，CMFC 基本未打开。
- 改动：投影使用 small 初始化（std=1e-3），CMFC 参数学习率倍率设为 10。
- 建议实验名：`ccsfuse_m_v2`。

## 本地 CMFC-only v2 配置准备与检查（2026-11-02）

- 工作目录：`D:\Projects\CCFUSE`；从 `main/27b8214` 创建 `codex/cmfc-only-v2-ablation`。开始时工作区干净，`git ls-remote origin refs/heads/main` 返回同一提交。
- 新增 `models/ccsfuse/yolov5m_cmfc_add_v2.yaml`。解析后的配置与完整版 `yolov5m_ccsfuse_v2.yaml` 相比，仅第 27/28/29 层由 AFSF 改为 PairAdd；主干、CMFC、回注、head、anchors 和 pretrain_map 完全相同。保留但不使用顶层 afsf 参数。
- 沿用 v2 推断设置：`proj_init: small`、`proj_std: 1.0e-3`、`cmfc_inject: A`；配套使用现有 `data/hyp.ccsfuse_dronevehicle_v2.yaml`，`cmfc_lr_mult: 10.0`。旧配置与训练代码未改动。
- 本地实际环境：`D:\Miniconda\envs\ccsfuse\python.exe`，Python 3.10.22、torch 2.5.1+cu124、numpy 1.23.5，RTX 3070 Ti Laptop GPU。
- 结构检查通过：3 个 CMFC、3 个 PairAdd、0 个 AFSF，未融合参数量 53,793,312。六个输出投影的初始化 std 均接近 0.001。
- 只读使用已有真实 COCO 权重 `D:\CCSFuse\weights\yolov5m.pt`；两路均为 306/306，共加载 612 个 state_dict 张量。逐张量比对全部一致，RGB/IR Focus 权重也分别与 COCO 第 0 层一致；未加载 neck。
- 单独执行现有 `train_rgb_ir` 的优化器分组代码块，483 个参数张量均且仅被收集一次；CMFC 参数全部进入倍数 10 分组，3 个 lambda 均在不衰减组。没有调用训练函数或 optimizer.step()。
- 合成输入 batch 2、640x640：FP32 推理输出为 `(2,25200,10)`，Detect 三层分别为 `(2,3,80,80,10)`、`(2,3,40,40,10)`、`(2,3,20,20,10)`，全部有限；PairAdd 特征尺寸为 `(2,192,80,80)`、`(2,384,40,40)`、`(2,768,20,20)`。
- 合成目标上的 CUDA AMP 检测损失反向通过：loss 约 1.815743（仅冒烟检查数值，不是实验指标），所有参数都有有限梯度，两路输入和三个 lambda 均有非零梯度。该短检查峰值 allocated 显存约 1.846 GiB，不代表长训练显存。
- 本次检查使用真实预训练权重、合成输入和目标，没有读取真实训练数据。真实数据 batch 检查尚待服务器执行。出现现有的 datasets.py 恒真 assert 和 torch.meshgrid 警告，未阻碍以上检查。
- 临时检查脚本、完整日志和 JSON 结果保存在本地 `C:\Users\bzp02\Documents\ChatGPT\Coding 2\artifacts\ccfuse_cmfc_v2\`，未加入项目仓库。
- 本阶段未提交、未推送、未连接服务器、未启动训练。后续服务器命令提供给用户，由用户执行。

## GitHub 上传、服务器短检查与训练指令交接（2026-11-02）

### 已执行：本地提交和上传

- 在 `D:\Projects\CCFUSE` 使用实际 Git 身份提交 `WORKLOG.md` 与 `models/ccsfuse/yolov5m_cmfc_add_v2.yaml`，提交为 `5cb0b49e35081d655e7103368f5000cd062695e1`（Add CMFC-only v2 ablation configuration）。
- 已推送 `origin/codex/cmfc-only-v2-ablation` 并通过 `git ls-remote` 核实本地/远端哈希一致；推送后工作区干净。未创建或合并 PR，未改动 main。

### 用户日志确认：服务器检查

- 服务器操作全部由用户执行，Codex 提供指令并分析回传输出，没有直接 SSH 连接服务器。
- 同步前用户回传：`main/27b8214`、工作区干净、origin 为 `https://github.com/Polaris-G/CCFUSE.git`；`ccsfuse` 激活于 `/home/baozhipeng/.conda/envs/ccsfuse`，base 为 `/opt/miniforge3`。GPU 为 RTX 5060 Ti（16311 MiB），驱动 595.99.02，当时没有相关训练/测试/检查进程输出，GPU 仅少量桌面占用。
- 已提供 `git fetch origin`、`git switch --track origin/codex/cmfc-only-v2-ablation` 及提交号核对指令，目标提交为上述 `5cb0b49`。用户随后运行新配置并贴回短检查输出；未贴回同步后的 `git rev-parse HEAD`，因此服务器实际 HEAD 哈希未独立确认。
- 用户实际运行的真实数据短检查命令（仓库根目录、ccsfuse 环境）：

```bash
python tools/check_model.py \
  --cfg models/ccsfuse/yolov5m_cmfc_add_v2.yaml \
  --weights weights/yolov5m.pt \
  --data data/multispectral/DroneVehicle.yaml \
  --hyp data/hyp.ccsfuse_dronevehicle_v2.yaml \
  --batch-size 2 --img-size 640
```

- 真实预训练加载：rgb 306/306、ir 306/306，共 612/890；两路 Focus 与 COCO 第 0 层权重比较均为 True。`CFT intersect_dicts would transfer 157/890` 是旧方式对照，不是本次实际加载结果。
- 数据：两路 train 均扫描到 17990，0 missing、33 empty。本次 batch 为 `(2,6,640,640)`，20 个目标。IR 行显示 RGB cache 路径是已知打印问题。
- 第 27/28/29 层均为 PairAdd，尺寸分别为 `(2,192,80,80)`、`(2,384,40,40)`、`(2,768,20,20)`；Detect 为 `(2,3,80,80,10)`、`(2,3,40,40,10)`、`(2,3,20,20,10)`。
- loss 1.999，box/obj/cls 为 0.122/0.065/0.814；loss finite=True、grads finite=True。此为单 batch 检查数值，不是训练精度。
- 参数 53.79M；640 输入 GFLOPs 为 thop 93.44、flop_counter 94.48；batch 2 AMP 前向/反向峰值 allocated 显存 1.85 GiB，不能据此声称 batch 12 已验证。
- 三个 CMFC 的分支初始权重均为 0.50/0.50，属正常初始化。torch.meshgrid 未来参数警告未阻碍本次运行。
- 判断：用户回传的真实数据短检查通过；只证明该 batch 计算正常，不代表长训练已经成功或精度提升。

### 仅已交付：正式训练指令

- 用户在短检查通过后明确索要训练指令，已提供以下命令。**没有收到启动/首个 epoch 日志，实际启动状态、保存目录、训练进度与 batch 12 显存仍待用户输出确认。**
- 先在服务器执行 `tmux new -s cmfc_only_v2`；进入会话后 `conda activate ccsfuse`、`cd ~/bzp/CCFUSE`，检查 `git status --short` 无意外改动，并核对代码提交。

```bash
python train.py \
  --cfg models/ccsfuse/yolov5m_cmfc_add_v2.yaml \
  --weights weights/yolov5m.pt \
  --data data/multispectral/DroneVehicle.yaml \
  --hyp data/hyp.ccsfuse_dronevehicle_v2.yaml \
  --epochs 45 --img-size 640 640 \
  --batch-size 12 --workers 8 --device 0 --nbs 12 \
  --name cmfc_only_m_v2_b12
```

- 启动日志核对 small/std=0.001、CMFC lr multiplier 10、两路 306/306 和 `Logging results to`。实验名为计划值，出现目录后缀时以实际日志为准；训练前还应确认同名实验是否已经存在，避免重复启动。
- tmux 脱离：Ctrl+B 后 D；返回：`tmux attach -t cmfc_only_v2`。仅在原进程已停止、该实验尚未完成时，用实际目录续训：`python train.py --resume runs/train/cmfc_only_m_v2_b12/weights/last.pt`。
- 下一步：分析用户提供的启动日志和首个 epoch 输出，再更新当前训练状态。训练完成后按独立 val 的 `--conf-thres 0.001 --iou-thres 0.5 --task val` 口径评估。

### 文档维护约定

- 用户要求每次完成项目任务后更新 `CLAUDE.md` 和 `WORKLOG.md`；已写入 CLAUDE.md 的工作约定。
- 本次更新两份文档，补齐已上传版本、服务器环境与真实数据检查、训练指令交付和待确认事项；替换 CLAUDE.md 过期的当前进度、Git 身份/分支说明及不适用于 v2 的日志要求，保留历史结果。
- 本次仅修改文档，检查文档差异和格式；没有改动模型、训练代码或实验配置，没有重复运行模型检查或操作服务器。本次文档改动暂未提交/推送。

## CMFC-only v2 训练完成，独立 val 待执行（2026-11-02）

- 来源：用户粘贴的服务器终端输出。已显示 epoch 44/44、1500/1500 batches 和 `45 epochs completed in 6.660 hours.`；据此记录 45 个 epoch 已完成，用时 6.660 小时。
- 末轮训练损失：box 0.02888、obj 0.05147、cls 0.00541、total 0.08576；img_size 640。训练日志显示 gpu_mem 13.7G，不将它写成已核实的峰值 allocated 显存。
- 末轮训练内 val：1469 图像、24490 个目标。下面均为百分数，属于训练内验证，不是 best.pt 的独立评估：

| 类别 | P | R | mAP50 | mAP75 | mAP50:95 |
|---|---:|---:|---:|---:|---:|
| all | 69.7 | 65.8 | 68.1 | 56.2 | 46.5 |
| car | 86.6 | 96.0 | 97.5 | 82.1 | 67.2 |
| truck | 63.0 | 61.5 | 64.3 | 46.0 | 40.0 |
| freight car | 47.7 | 59.5 | 52.8 | 41.9 | 34.9 |
| bus | 92.3 | 88.5 | 93.1 | 86.6 | 68.8 |
| van | 59.1 | 23.3 | 33.0 | 24.3 | 21.3 |

- 训练内 NMS IoU 为 0.6；已有四组独立 val 使用 IoU 0.5、conf 0.001 和各自 best.pt。现有 best.pt 按 0.1*mAP50 + 0.9*mAP75 选取，不保证来自最后一轮或取得最高 mAP50。暂不将本次末轮数值并入独立评估对照表，也不据此断言模块优劣。
- 用户只提供末轮与训练完成输出，实际 `Logging results to` 和服务器 HEAD 尚未回传。计划目录为 `runs/train/cmfc_only_m_v2_b12`；有后缀时以实际目录替换，不能声称该路径已确认。
- 已交付以下独立 val 指令（仅提供，由用户在服务器执行；结果尚未回传）：

```bash
python test.py \
  --data data/multispectral/DroneVehicle.yaml \
  --weights runs/train/cmfc_only_m_v2_b12/weights/best.pt \
  --batch-size 12 --img-size 640 \
  --conf-thres 0.001 --iou-thres 0.5 \
  --task val --device 0 \
  --name cmfc_only_m_v2_b12_val
```

- 记录本次结果时发现旧本地仓库路径不存在，因此先将原始数值暂存为 pending_training_completion_2026-11-02.json；随后在下述项目恢复任务中正式补入两份文档。

## 本地仓库恢复到 D:/project/CCFUSE（2026-11-02）

- 用户报告本地文件夹丢失，要求在 D 盘新建 project 文件夹恢复项目。原 D:/Projects/CCFUSE 与 D:/CCSFuse 无法找到，新目标 D:/project/CCFUSE 原本不存在。
- 只读核对 GitHub：main 为 27b8214，codex/cmfc-only-v2-ablation 为 5cb0b49e35081d655e7103368f5000cd062695e1。新建 D:/project 后克隆完整仓库（非浅克隆、非 single-branch），检出实验分支；远端其他分支与完整 Git 历史仍可访问，origin 保持 https://github.com/Polaris-G/CCFUSE.git。
- 恢复代码的实际命令：

```powershell
git clone --branch codex/cmfc-only-v2-ablation -- https://github.com/Polaris-G/CCFUSE.git D:/project/CCFUSE
git -C D:/project/CCFUSE fsck --full
```

- 克隆后 HEAD 为上述 5cb0b49，工作区干净，git fsck --full 通过。新配置解析结果与提交内容一致，并仍只在三处融合层区别于完整版 v2；没有重跑训练或 GPU 检查。
- 此前未上传的 CLAUDE.md / WORKLOG.md 修改从当前聊天工作区的 update_docs.py 重建，保留当时历史状态；另补回脚本之外的路径修正及历史待办。当前有效路径改为 D:/project/CCFUSE，历史操作路径不做全局替换。
- 同时把用户已回传的 45 epoch 训练完成结果写入文档，当前下一步为 best.pt 独立 val，不能再把本轮训练记为“尚未启动确认”。
- 回收站 Shell 元数据和可读的 D 盘回收记录未找到匹配旧路径的可用副本，另有一处访问受限，不能声称全盘排除。恢复来源是 GitHub 和留存记录，不是磁盘级找回。
- 本次恢复了已提交代码/配置及能够重建的文档、检查记录；未恢复 Git 忽略的数据集、权重、训练产物或论文。未连接或修改服务器，服务器文件仍由用户检查/管理。
- 历史检查日志、19 项 PASS 结果、原检查脚本、交接文档及训练完成记录副本保存到本地 runs/recovery_2026-11-02/（被 Git 忽略）。历史检查脚本保留旧绝对路径，仅作证据，不应直接当作新目录中的可运行检查脚本。
- 本次恢复的文档修改尚未提交/推送。最终核对仅 CLAUDE.md、WORKLOG.md 存在工作区差异，代码与配置保持 GitHub 5cb0b49 原样；文档快照另存聊天工作区供再次恢复。

## CMFC-only v2 独立 val 回传与消融对比（2026-11-02）

- 来源：用户本次粘贴的评估终端输出。虽然用户称为 test 结果，日志标记 val、图像 1469、目标 24490，与验证集一致；不是 DroneVehicle 的 8980 张 test 集。两路 val 扫描均为 1469 found、0 missing、2 empty。
- 按此前交付的命令，将其记录为 CMFC-only v2 的 best.pt 独立 val：batch 12、640、conf=0.001、NMS IoU=0.5、task val。该归类依据连续对话，输出片段未直接包含完整启动命令、权重路径、保存目录或服务器 HEAD；不声称已独立核实这些元数据。计划权重路径仍为 runs/train/cmfc_only_m_v2_b12/weights/best.pt，实际路径可能带后缀。
- 原输出数值换算为百分数如下；P/R 为脚本选定工作点的值，不等于全阈值表现：

| 类别 | P | R | AP50 | AP75 | AP50:95 |
|---|---:|---:|---:|---:|---:|
| all | 72.2 | 65.2 | 68.4 | 56.4 | 46.7 |
| car | 91.1 | 96.1 | 97.9 | 82.3 | 67.5 |
| truck | 63.2 | 60.4 | 64.2 | 45.9 | 40.0 |
| freight car | 49.9 | 58.7 | 53.2 | 42.0 | 35.1 |
| bus | 93.4 | 88.3 | 93.2 | 86.6 | 69.1 |
| van | 63.2 | 22.7 | 33.6 | 25.1 | 21.7 |

### 独立 val 总体对比

- 前四个已有配置来自交接中的独立 val 记录，本次新增 CMFC-only v2。以下均为百分数，末轮训练内的 68.1/56.2/46.5 不放入此表：

| 实验 | mAP50 | mAP75 | mAP50:95 |
|---|---:|---:|---:|
| Add 基线 | 68.8 | 56.1 | 46.8 |
| CMFC-only v2 | 68.4 | 56.4 | 46.7 |
| AFSF-only | 69.2 | 56.2 | 47.2 |
| 完整版 v2 | 70.7 | 59.9 | 49.2 |
| 完整版 v3 | 67.5 | 55.1 | 45.9 |

- CMFC-only v2 相对 Add 基线：mAP50 -0.4、mAP75 +0.3、mAP50:95 -0.1 个百分点。
- 相对 AFSF-only：-0.8/+0.2/-0.5 个百分点；相对完整版 v2：-2.3/-3.5/-2.5；相对完整版 v3：+0.9/+1.3/+0.8。
- 分类差值（CMFC-only v2 减 Add 基线，单位为百分点）：

| 类别 | ΔAP50 | ΔAP75 | ΔAP50:95 |
|---|---:|---:|---:|
| car | -0.1 | +0.1 | +0.0 |
| truck | -0.8 | +3.1 | +0.9 |
| freight car | +1.1 | +2.7 | +1.3 |
| bus | -2.1 | -1.8 | -1.2 |
| van | +0.0 | -3.0 | -1.4 |

### 结论与下一步

- 当前单次结果中，CMFC 单独加入未带来明确的总体增益；freight car 改善，bus 下降，van AP50 与基线相同且更高 IoU 下退步。不能把 -0.4 或 -0.1 个百分点解释为稳定统计结论。
- 完整版 v2 仍是当前最好的一组，比 CMFC-only 高 2.3/3.5/2.5 个百分点；相对 CMFC-only 的 AP50 增益在 freight car 为 +4.6、van +4.1、bus +2.1。结果与两个模块可能有配合作用的解释一致，但尚未证明机制，不支持据单模块结果直接删除 CMFC。
- van 在此次评估工作点召回为 22.7%，需结合已有 PR 曲线、混淆矩阵和错例区分漏检、分类混淆与定位问题；不要仅凭一个 P/R 工作点断言所有阈值表现。
- 下一步建议先归档实际权重与输出目录、分析已有 bus/van 错例，保留完整版 v2 作为当前参考。若要据小差异做结构取舍，再用重复 seed 验证。未新增训练、未修改模型或学习率，服务器仍由用户操作。
- 本轮已同步更新 CLAUDE.md 的当前状态和对比表，并在此追加完整指标；保留之前恢复的未提交文档改动。本轮只做文档和数值核对，不重复运行模型测试；文档改动仍未提交/推送。

## bus/van 离线错误诊断工具与服务器指令（2026-11-02）

- 用户要求下一步，并补充 `Results saved to runs/test/cmfc_only_m_v2_b12_val`，确认这是服务器上的实际评估目录。Codex 只能访问本地 D:/project/CCFUSE 和用户贴回的输出，不能直接看到服务器文件；服务器同步和诊断运行继续由用户执行。
- 只读审计发现：独立 test.py CLI 默认 save_txt/save_conf=True，导出 labels/<image_stem>.txt，每行为 `class xc yc w h confidence`，坐标还原到原图后归一化。CLI 默认 plots=False 且未提供 --plots，因此该独立评估不保证存在 PR/混淆图，优先复用现有 TXT。
- 原 ConfusionMatrix 的调用/实现会跳过部分零预测、零 GT 或无匹配情况；本轮不修改它、test.py、训练逻辑或既有指标，而新增独立离线工具。

### 本地实现与检查

- 新增 tools/diagnose_detections.py：只依赖现有 NumPy/PyYAML 及标准库，不加载 torch/权重，不调用模型、不读写图片内容、不写入任何输入目录。遍历完整 val 图片列表，读取对应 RGB 侧真值和导出预测，空 GT/空预测均计数；图片只用于列举完整样本及错例路径。
- 输入核对包括图像/目标总数、5/6 列格式、类别/数值/归一化坐标、重复 stem、重复行、错分割的额外预测文件以及每图超过当前 NMS max_det=300 的异常。GT 要求正面积；原图裁剪后可能出现的零面积预测保留为 FP。已存在的输出目录拒绝覆盖。
- 因 test.py 对零预测图片不生成 TXT，缺 TXT 默认为错误；只有在用户确认导出完整时才使用 --allow-missing-predictions，将缺文件按零预测计入。本次用户已提供完整评估指标与结束保存目录；该显式参数仍不代替完整性来源核实。不能用 save_hybrid 导出或重复 --exist-ok 追加污染的预测做此诊断。
- 默认对 confidence 0.05/0.25/0.5 与匹配 IoU 0.5/0.75 的 6 种组合统计。先按同类 IoU 降序贪心一对一配对 TP，再对剩余框进行跨类别配对；每个真值和保留预测均计入矩阵。固定阈值 FP/FN 与混淆类别按这一诊断定义计算，不替代原 AP，也不一定等于 test.py 平均 F1 最大工作点的 P/R。
- 未匹配真值按最佳同类/跨类框的 IoU 标记 assignment_conflict、localization_candidate、class_localization_candidate、no_overlap_candidate；定位候选的最低 IoU 默认 0.1。这些是排查线索，不是已证实的错误机制。不能据 NMS 后 TXT 还原被抑制或低于导出阈值的候选。
- 输出 summary.txt、summary.json、threshold_summary.csv、每种阈值的 confusion_*.csv，以及 bus/van 的 examples.csv。混淆矩阵行是预测类别、列是真实类别，额外行/列分别表示未匹配真值/预测；错例按文件名顺序每类/错误/阈值最多 20 条，不声称代表全体分布。JSON 保存工具 SHA256、实际命令、数据/预测路径、覆盖情况及全部类别计数。
- 新增 tools/test_detection_diagnostics.py：39 项测试通过，包含零 GT/预测/完全无匹配、TP 优先、错类、疑似定位、重复预测、阈值边界、零面积预测、输入不变、完整 CLI 输出、拒绝覆盖、错分割/不完整导出等。独立审阅发现的错例关联与近邻阈值文件名冲突已修复并有回归用例；最终增加来源元数据后另跑 5 项 CLI 测试也全部通过。
- 验证使用合成标签/预测和模拟图片文件名，没有本地真实数据验证，也未在服务器运行。测试日志保存在当前聊天工作区 artifacts/ccfuse_diagnostics/。

### 交付给用户的服务器命令

- 以下命令由用户从 ~/bzp/CCFUSE、已激活的 ccsfuse 环境执行。先确认 git status --short 没有意外改动，再切到实验分支并快进同步本轮提交；不要求用户改服务器模型代码。

```bash
git switch codex/cmfc-only-v2-ablation
git pull --ff-only origin codex/cmfc-only-v2-ablation
python tools/diagnose_detections.py \
  --data data/multispectral/DroneVehicle.yaml \
  --pred-dir runs/test/cmfc_only_m_v2_b12_val/labels \
  --output runs/diagnostics/cmfc_only_m_v2_b12_val \
  --expected-images 1469 --expected-labels 24490 \
  --allow-missing-predictions
```

- 预期输出目录为 runs/diagnostics/cmfc_only_m_v2_b12_val；如果目录已存在，选择新的输出目录，不覆盖旧报告。运行后回传 summary.txt（命令本身也会打印该摘要）。若缺 6 列预测、输入校验失败或文件不完整，先回传错误，不自动重新推理或训练。
- 下一步根据用户实际诊断摘要比较不同 confidence/IoU 下的 bus/van 漏检、错类与定位线索，再决定是否检查原图错例或运行其他已完成实验的同口径离线诊断。本次尚未获得任何服务器诊断结果。
- 本轮将此前未提交的恢复/实验文档更新与上述两个工具文件一起提交到 codex/cmfc-only-v2-ablation，通过 GitHub 交付；提交和推送结果以本轮实际 Git 返回的哈希为准。没有合并 main、重训或直接连接服务器。

## 新会话交接与分类差值纠正（2026-11-02）

- 用户要求为新会话提供项目目标、进度、路径、数据/训练配置、已完成/待办、实验结果、问题、失败方案、下一步及不可违反要求，并明确证据来源和 Git 状态。新增根目录 CCFUSE_新会话交接_2026-11-02.md，副本保存在当前聊天工作区 artifacts/ccfuse_handoff/；同步更新 CLAUDE.md 的入口、Git 和当前状态。
- 本地查询确认当前分支 codex/cmfc-only-v2-ablation，HEAD 1e6e4de390eac2137d7778a1d5086f6636ce5ef6（Add offline detection diagnostics and record CMFC-only results）。git ls-remote 返回远端同名分支同哈希，main 为 27b8214c5308b7d2b7ab94d5b4b71e5bedd2d4d3；写交接前工作区干净。此前诊断工具和恢复/结果文档已经上传，不能继续说它们未提交。
- 再次核对本地权重 weights/yolov5m.pt、相邻 D:/project/datasets/DroneVehicle 不存在，仓库内未找到 PDF；恢复没有包括这些非 Git 产物。解释器 D:/Miniconda/envs/ccsfuse/python.exe 仍存在。没有重跑真实权重、GPU 或数据检查。
- 交接区分本地核验、用户服务器日志、历史记录与推测。训练完成及独立 val 指标、服务器评估目录已经用户确认；实际 checkpoint 路径、训练目录、完整评估命令、服务器 HEAD 和离线诊断结果仍待确认。服务器命令只是已交付，不代表已执行。下一步仍由用户执行 CPU 离线诊断并回传 summary.txt，不新增训练。
- 发现并纠正此前分类差值转录错误：原始 D:/codex/CCFUSE_本地开发交接_2026-11-02.md 的 baseline freight car AP50:95=33.8、bus=70.3。此前误用了 v3 的 freight=33.3 和 v2 的 bus=71.3，导致差值误记为 +1.8/−2.2。按用户 CMFC-only 的 35.1/69.1，正确差值为 +1.3/−1.2 个百分点；已更正上方表格。总体指标及其他分类差值不变；这是记录纠错，没有重新评估。
- 同步纠正聊天工作区 artifacts/ccfuse_cmfc_v2/independent_val_2026-11-02.json 的派生差值与已确认评估目录、record_independent_val.py 的两项基线常量，并附纠错说明。历史脚本未重跑，旧快照保留作历史证据；不要由旧快照重新引入错误。
- CLAUDE.md 开头“官方代码未公开”改为历史记载且本次未重新核实；其他历史论文解释、早期误差归因与未验证结构设想在新交接中标明边界。未把未运行方案列为失败实验。
- 本次仅交接/文档和派生记录纠错，没有修改模型、训练代码或配置，没有 SSH、推理、训练、提交、推送或合并 PR。完成后的 Git 变化为修改 CLAUDE.md、WORKLOG.md，以及新增未跟踪 CCFUSE_新会话交接_2026-11-02.md；均未暂存，HEAD 保持 1e6e4de。
- 检查范围：文档内容和原交接数字复核、代码配置静态核对、Markdown 围栏/差异/路径与最终 Git 状态；本次不重复模型或诊断单元测试。新会话提交或执行后续步骤前应再次检查实际状态。


## 三组离线诊断回传与逐目标对照工具（2026-11-02）

**来源与范围。** 用户先回传 CMFC-only v2 的诊断摘要和 JSON 提取结果，随后回传 Add 基线与完整版 v2 的诊断摘要和 conf=0.25、匹配 IoU=0.5 的完整混淆矩阵。用户提供的命令和输出确认本轮诊断读取以下服务器目录：

| 角色 | 预测 TXT 目录 | 已生成诊断目录 |
|---|---|---|
| Add 基线 | `runs/test/exp5/labels` | `runs/diagnostics/add_baseline_val` |
| CMFC-only v2 | `runs/test/cmfc_only_m_v2_b12_val/labels` | `runs/diagnostics/cmfc_only_m_v2_b12_val` |
| 完整版 v2 | `runs/test/exp6/labels` | `runs/diagnostics/ccsfuse_v2_val` |

以上对应关系依据交接和本轮用户执行的命令；本地没有直接读取服务器预测或 checkpoint。原始模型评估的完整启动参数、实际权重来源仍需保留既有待确认标识，不能把离线诊断命令当作原始推理参数的独立验证。

三组均打印 1469 图、24490 个 GT、1468 份预测 TXT、缺失 1 份 TXT。CMFC-only 的 `missing_prediction_files` 明确为 `['01322.jpg']`，其 GT 内容尚未回传；其他两组缺失文件名尚未回传。缺失数相同不等于缺失图片相同，也不单独证明导出完整。

**van 的固定阈值对照。** 下表每格为 `TP / FP / FN`，不是 AP，也不是 `test.py` 原始选定工作点的 P/R：

| conf | 匹配 IoU | Add | CMFC-only v2 | 完整版 v2 |
|---:|---:|---:|---:|---:|
| 0.05 | 0.5 | 513 / 3474 / 212 | 496 / 2854 / 229 | 495 / 2458 / 230 |
| 0.05 | 0.75 | 448 / 3539 / 277 | 422 / 2928 / 303 | 439 / 2514 / 286 |
| 0.25 | 0.5 | 253 / 327 / 472 | 219 / 230 / 506 | 281 / 347 / 444 |
| 0.25 | 0.75 | 232 / 348 / 493 | 196 / 253 / 529 | 262 / 366 / 463 |
| 0.5 | 0.5 | 55 / 19 / 670 | 47 / 17 / 678 | 100 / 32 / 625 |
| 0.5 | 0.75 | 54 / 20 / 671 | 41 / 23 / 684 | 95 / 37 / 630 |

conf=0.25、匹配 IoU=0.5 的进一步对照：

| 指标 | Add | CMFC-only v2 | 完整版 v2 |
|---|---:|---:|---:|
| van 精确率 / 召回率（%） | 43.6 / 34.9 | 48.8 / 30.2 | 44.7 / 38.8 |
| 真实 van → car | 347 | 356 | 349 |
| 真实 van → 其他错误类别（不含 car） | 79 | 93 | 54 |
| 未匹配 van GT（混淆矩阵末行） | 46 | 57 | 41 |
| bus TP / FP / FN | 744 / 127 / 45 | 710 / 67 / 79 | 738 / 94 / 51 |

CMFC-only 的预测框分类已由用户额外提取：conf=0.05、IoU=0.5 时，van 的 2854 个 FP 中有 2728 个 `other_overlap`（95.6%），另有 classification=16、background=104、duplicate=6；conf=0.25 时 230 个 FP 中有 other_overlap=179、classification=47、background=1、duplicate=3。

**解释边界与本轮决策。** 三组的 van→car 总量接近，说明这是共同的聚合错误模式，不能据此认定为 CMFC 特有故障，更不能推定是同一批 GT。完整版相对 Add 在 conf=0.25 时净多 28 个 van TP，但 van→car 总数增加 2；其他分类错总数减少 25、未匹配减少 5。这只是总量平衡，不是逐目标转移证据。CMFC-only 相对 Add 少 34 个 TP、少 97 个 FP，体现该工作点的精度/召回取舍。保留完整版 v2 为参考，进入同图同目标对照，不直接修改 loss 权重、NMS、模块或学习率。

诊断先匹配同类 TP，再在剩余框中跨类配对；`classification` 不是模型内部分类头错误的直接测量。已被正确框匹配的 car 上多出的错误 van 框可以进入 `other_overlap`，所以不能把 van 行的 classification=16/47 解释为全部异类误报量。`other_overlap` 只说明存在达到候选重叠下限的 GT，不等于背景、定位错误或跨类重复中的任意单一原因。本地评估代码允许多类别预测并按类 NMS，可解释部分同位置跨类框，但真实占比尚未测得。

**本轮工具实现。** 用户回复“下一步吧”，授权实现离线三模型逐目标对照。新增 `tools/compare_detections.py`、`tools/paired_detection_render.py`、`tools/test_compare_detections.py`，读取既有三份 `summary.json` 及其引用的 GT/预测路径，复用 `diagnose_detections.py` 的匹配语义。按图片相对路径和 GT 行号对齐全部 focus GT，保存六组阈值的状态、关联类别/分数/IoU、共同错误和模型成对 TP 得失；候选分数与实际匹配分开记录，不能将候选分数直接视为匹配结论。

输入核验覆盖三报告的 names/数据路径/图像与 GT 计数一致性、输入重算结果与旧报告一致、缺失预测仅限已记录列表、既有输出拒绝覆盖和不写输入目录。汇总计数使用全部目标；图像按可复现抽样挑出少量目标，主工作点 conf=0.25、匹配 IoU=0.5。输出 PNG 两行 RGB/IR、四列 GT/Add/CMFC/Full，在同一目标上下文中显示匹配状态和框。没有调用模型、权重、GPU、训练或额外推理。

**实际本地验证。** 在本地ccsfuse环境执行 `python tools/test_compare_detections.py`，17项测试通过。覆盖逐GT交集、同总量互换错误的gain/loss、六场景全量CSV、低于当前阈值的候选分数、无候选为空、缺TXT元数据、新缺失/报告和源输入变化拒绝、输出不覆盖、输入目录保护与输入哈希不变、八面板PNG端到端及去重。本地最初一次测试因沙箱Temp目录权限失败，切到正常临时目录后测试通过；没有把环境失败算成算法失败。实际查看了聊天工作区 `artifacts/ccfuse_paired/render_smoke/comparison.png` 合成八面板，文字、上下文裁剪和框位置可读；渲染模块另外完成空预测/边界GT、缺IR、尺寸不符和拒绝覆盖等轻量检查。验证使用合成 GT、预测和图像；没有本地真实 DroneVehicle 图片/权重/预测，不能声称已得到真实逐目标交集或真实并排图。

**仅交付给用户的服务器命令。** 下列输入顺序固定为 Add、CMFC-only、完整版 v2；从已激活的 `ccsfuse` 环境和服务器仓库根目录执行。先查看 `git status --short --branch`，有意外改动时不要强制重置；正常时同步实验分支。以下命令未由助手在服务器执行：

```bash
cd ~/bzp/CCFUSE
git status --short --branch
git pull --ff-only origin codex/cmfc-only-v2-ablation
git log -1 --oneline

python tools/compare_detections.py \
  --reports \
    runs/diagnostics/add_baseline_val/summary.json \
    runs/diagnostics/cmfc_only_m_v2_b12_val/summary.json \
    runs/diagnostics/ccsfuse_v2_val/summary.json \
  --output runs/diagnostics/van_three_model_comparison \
  --expected-focus 725 \
  --samples-per-group 10
```

命令已按最终CLI核对，`--help`正常。若输出目录已存在，读取已有报告或更换新的输出名，不覆盖历史结果。回传 `summary.txt`，并提供共同 van→car、Add 检对但 CMFC 检错、完整版检对但其他模型检错等组的若干 PNG。根据全量转移和图像证据决定下一步，不能由抽样图推断总体比例。

**工作区与交付。** 起点为 `codex/cmfc-only-v2-ablation`、`1e6e4de`，远端分支相同；main为`27b8214`。保留并纳入此前CLAUDE/WORKLOG交接与数值纠错，以及未跟踪的新交接文档；后者增加快照过期提示，不改写历史。本轮新增三个工具/测试文件，修改上述三份文档；模型、train.py、test.py、数据/实验配置及原诊断工具未改。此记录与工具一起准备提交和通过GitHub交付；实际上传结果在核对后补记。没有SSH、服务器推理或训练。

**来源和输出补充。** 重算核对旧报告能发现汇总不一致，旧报告无原始输入哈希，不能证明每个历史文件字节未变。新报告保存当前输入摘要；候选分数来自保存的NMS后框，不还原被抑制候选。CSV中GT行号按非空行从0开始，groups可重叠，gain/loss分别计数不能用净差代替。输出同时包含README.txt和抽样清单，实际图片在images目录；渲染错误会记入summary.json并返回非零退出码，保留已完成计数供排查。

### 本轮 GitHub 交付核验

- 工具及文档提交 `725f420d2048d891ac934e3e3d5b1efbf541ba61` 已成功推送到 `origin/codex/cmfc-only-v2-ablation`，并用 `git ls-remote` 核对该远端分支与本地一致；main 仍为 `27b8214c5308b7d2b7ab94d5b4b71e5bedd2d4d3`，未合并。
- 本段及 CLAUDE 的上传状态是核验之后补记；后续文档提交不改变上述工具内容。服务器仍由用户同步和运行，未取得真实逐目标结果。


## 服务器逐目标对照完成与真实图片检查待办（2026-11-02）

- 【用户日志】服务器 `git status --short --branch` 显示实验分支工作区干净；随后同步到 `cdf5850`（本地/远端跟踪分支相同），运行三报告逐目标对照工具成功。不是助手SSH执行。
- 用户终端确认1469图、24490 GT、725 van；六组阈值的原报告矩阵和类别统计均重算一致；生成72张并排PNG，render_errors=0。输出目录为 `/home/baozhipeng/bzp/CCFUSE/runs/diagnostics/van_three_model_comparison`。PNG和完整CSV尚未发送给助手，不能称已看过真实错例。
- 三组缺TXT均为 `01322.jpg`，GT=0、gt_classes为空。因此该图不贡献本次GT/FN，不能解释van漏检；缺TXT和空GT本身仍不足以证明原评估一定完整，若原本有未导出的预测仍可能涉及FP。

以下pairwise为实际逐GT交集；每格格式为 `共同TP / 新增TP / 丢失TP / 共同miss / 共同van→car`，方向从左模型到右模型。不是AP，也不是因果或模块即时修复证据。

| conf | 匹配IoU | 三组共同van→car | Add→CMFC | Add→Full | CMFC→Full |
|---:|---:|---:|---|---|---|
| 0.05 | 0.5 | 153 | 462 / 34 / 51 / 178 / 160 | 465 / 30 / 48 / 182 / 167 | 463 / 32 / 33 / 197 / 179 |
| 0.05 | 0.75 | 110 | 379 / 43 / 69 / 234 / 119 | 391 / 48 / 57 / 229 / 124 | 384 / 55 / 38 / 248 / 134 |
| 0.25 | 0.5 | 289 | 182 / 37 / 71 / 435 / 306 | 207 / 74 / 46 / 398 / 307 | 195 / 86 / 24 / 420 / 317 |
| 0.25 | 0.75 | 198 | 158 / 38 / 74 / 455 / 221 | 186 / 76 / 46 / 417 / 215 | 171 / 91 / 25 / 438 / 237 |
| 0.5 | 0.5 | 240 | 25 / 22 / 30 / 648 / 258 | 44 / 56 / 11 / 614 / 257 | 37 / 63 / 10 / 615 / 268 |
| 0.5 | 0.75 | 180 | 24 / 17 / 30 / 654 / 200 | 43 / 52 / 11 / 619 / 192 | 32 / 63 / 9 / 621 / 210 |

- 在conf=0.25、IoU=0.5，289/725=39.9%的van属于三组共同错类；此时不再只是相近总量，而是明确的同GT交集。仍不能直接认定标签错误、类不平衡或架构无能力。较低阈值0.05的共同错类为153，跨阈值会重新匹配，不能未对齐GT身份便声称153一定是289的子集。
- 同一工作点CMFC→Full +86/-24，净+62；Add→Full +74/-46，净+28。改善与退步两侧都应看图，不能把净增数当全部改善目标，也不能把最终预测差异称为AFSF即时“救回”。0.05/0.5时CMFC→Full +32/-33，净-1；0.5/0.5时+63/-10，净+53，显示工作点相关差异，但不足以证明仅为校准问题。
- 下一步只读取已有产物，打包完整summary.json/txt、paired_targets.csv、transitions.csv、selection.csv、README.txt和四组各3个PNG：all_three_to_confusing_class、full_tp_cmfc_miss、cmfc_tp_full_miss、add_tp_cmfc_miss。同一PNG去重，最多12张，不重新运行诊断或模型。这些是现有固定seed样本的一部分，用来目检，不代表总体错误比例。
- 打包命令由用户在服务器执行；预计输出 `runs/diagnostics/van_review.zip`，使用ZIP独占创建避免覆盖，并检查所有引用文件位于本次报告目录内。ZIP只含本次明确列出的报告及抽样PNG，不含模型权重、完整数据集或其他项目文件。助手尚未创建该服务器ZIP或收到实际图片。
- 本轮本地只更新CLAUDE.md/WORKLOG.md并记录上述用户结果，未改工具、模型、训练/评估代码或配置；未重复运行已通过的测试。文档随本轮记录提交同步到实验分支；查看和打包已有产物不依赖更新后的文档提交，服务器无需为这一步再pull。


## van_review.zip 全量分析与12张真实图检查（2026-11-02）

**输入与实际执行。** 用户提供C:/Users/bzp02/Downloads/van_review.zip（5,244,507字节）。本地检查ZIP成员路径后独占解压到聊天工作区artifacts/ccfuse_paired/review_received_2026-11-02，读取summary.json/txt、paired_targets.csv、transitions.csv、selection.csv、README及12张PNG。完整CSV有4350行、725个van、6场景；12张是四组各3例的定向子样本，不是全部72张，不用于估计总体比例。原ZIP/提取输入只读，未上传数据到Git、未SSH、未调用模型或GPU。

**全量复核。** 各场景TP、共同错类及转移覆盖与summary一致。工作区analyze_received.mjs保存可重做分析，quant_analysis.json保存定义、分布和SHA256；源文件hash前后不变。根代理用独立csv读取再核验4350/725/289/86、跨阈值joint TP、Full候选中位数和面积分组。paired_targets.csv SHA256为66e15fa408603d36fbf7a296d0b797374b72a199a6e7d27169f05475add9364e。这只核对收到的产物，不补足原预测的历史字节哈希或权重身份。

以下主工作点为conf=0.25、匹配IoU=0.5：

- CMFC→Full新增86个TP，在conf=0.05时两组全部为TP；86个CMFC的几何合格van候选分数均在[0.05,0.25)，Full均≥0.25。最高分van候选的分数中位0.1708→0.3167，IoU中位0.8539→0.8708。原来已有符合该IoU的候选，不应解释成新发现86个未定位目标；仍不证明只是校准或模块即时机制。
- 反向丢失24个TP全部也是Full van候选低于0.25、CMFC≥0.25；23个在0.05时两组都为TP，剩余1个Full van分数低于0.05。
- 三组共同van→car为289/725=39.9%；Add/CMFC/Full有合格van候选289/288/289个，凡同时存在car和van候选，car分数均更高。van分数中位0.0264/0.0318/0.0198，car中位0.7949/0.7568/0.7803；Full的289个van中207个<0.05、82个在[0.05,0.25)。这些是各类最高分NMS后候选，未必为实际匹配框，也不能拆成内部分类概率。
- Full的281个TP中，41个（14.6%）car候选分数高于van；新增86个中18个（20.9%）。TP-first诊断的TP不意味着van分数已赢过所有car候选。
- 0.25时共同289个与0.05时共同153个的精确交集为148；分别独有141和5个。至此用实际GT身份补足此前的集合边界。
- 按全部725个van的归一化GT面积四分位，三组共同van→car为125/182、96/181、58/181、10/181，即68.7%、53.0%、32.0%、5.5%。共同289个面积中位占图0.446%，其他436个0.821%。这是尺度相关性，不证明类别错误或扩大输入必有改善。

**真实图目检。** 本地主代理与图像审阅代理均检查全部12张PNG。00974#6、00827#38、01167#19均为暗RGB停车场小车，IR轮廓可见，三组car框已覆盖目标而van分数很低。00683#9、00762#5、00783#1为长封闭货厢/驾驶室车辆，GT仍为van，模型在van/truck/freight car间分歧，需按数据集定义追溯。00689#1为边界截断目标，形态判断受限。00942#19的Add van候选0.240、CMFC0.203、Full0.370；01170#4的CMFC0.252/Full0.241、01300#1的Full0.242体现阈值邻近变化。00807#21 Full TP仍有更高分car候选。图中未见贯穿样本的统一100像素裁边错误或明显RGB/IR平移，但不能据12图排除细微配准问题，不能自行改GT。

**标签和实现审计。** tools/prepare_dronevehicle.py与data YAML均固定car/truck/freight car/bus/van；NAME_MAP保持car/van，未发现索引交换。默认读取IR XML，同一文本写入两模态，裁白边100、OBB转HBB、裁剪与最小尺寸过滤会改变行号对应；需按转换后坐标匹配原对象，不能把YOLO #1当XML第二个对象。原DroneVehicle论文只列五类及示例，没有找到可操作的van/truck细分定义；作者将van与car外观相似列为其van增益较小的可能原因（https://arxiv.org/pdf/2003.02437，第11页讨论）。该外部解释不证明本项目错误原因；官方repo第三方issue疑问也不能当作作者承认标错。

另发现的独立因素，均未修改：ComputeLoss不读取model.class_weights，cls_pw是标量；当前cls=0.5在5类、3层下会缩放为0.03125，不能仅凭绝对数值认定有错。--image-weights路径引用dataset.n/indices，与双流dataset属性及非mosaic索引链路不兼容，不直接建议启用。单卡factory未指定shuffle；工作区probe_sequential_sampler.py从当前源码AST提取factory/InfiniteDataLoader/_RepeatSampler，使用torch2.5.1+cu124真实DataLoader与17个合成配对ID，在rank=-1、image_weights=False、rect=False、batch12、workers0下验证SequentialSampler、两epoch均0..16、batch[12,5]、RGB/IR对应一致、CUDA未初始化。证据probe_sequential_sampler_result.json；不是服务器torch/workers8/真实增强的运行复现，不能归因当前错误。focal开关同时影响cls与obj，不是纯分类对照。

**下一步与交付边界。** 优先核对上述6个代表GT的源XML及convert_stats.json。用户表示不确定是否保留原始XML，下一条交付是在~/bzp/datasets内查找00683.xml、已有压缩包和转换记录的只读命令；尚未在服务器执行。若源标签一致，可讨论仅改全局cls的受控实验（影响所有类别）或独立shuffle对照；两者都只是候选，不同轮混改，也尚未授权/启动。若源XML缺失，记录限制，不根据模型预测改val标签。完整版v2继续作参考。

本轮仓库只更新CLAUDE.md/WORKLOG.md；实测与分析证据保留聊天工作区，原始数据/PNG/CSV不纳入Git。未改诊断工具、模型、训练/评估代码、配置或已有指标。起点HEAD为2876fdeb186823903225053f31535888c11a6d0d、实验分支工作区干净；文档随本轮记录提交并同步到该分支，实际结果以工具核验及最终回复为准。此检查不要求服务器为文档更新再次pull。


## 转换记录回传与分类损失单变量实验准备（2026-11-02）

**来源与文件查找边界。** 用户在服务器~/bzp/CCFUSE执行此前交付的只读命令并贴回输出。两次find均无输出：第一次仅在~/bzp/datasets中查找00683.xml或00827.xml；第二次在同一目录查找zip、tar、tar.gz、tgz、7z、rar扩展名的文件。只能记为该范围内未找到所查样本XML及所列格式压缩包，不能推定整个服务器没有原始XML、其他样本不存在、或源标注无误。此阶段保留源对象标签未复核限制，不再要求大范围查找或重下整套数据，不根据模型预测改val标签。

**用户提供的convert_stats.json。** label_source=ir、min_size=2.0、quality=95；classes依次为car、truck、freight car、bus、van，与代码及诊断一致。下表的目标总数为五类计数求和，其他计数原样记录；没有重新转换数据，也不把统计匹配当成逐对象源标注验证。

| 统计 | train | val | test |
|---|---:|---:|---:|
| 图像对pairs | 17990 | 1469 | 8980 |
| car | 270348 | 20588 | 137145 |
| truck | 15833 | 1470 | 8657 |
| freight car | 11193 | 918 | 5062 |
| bus | 11334 | 789 | 4467 |
| van | 7701 | 725 | 4282 |
| 五类目标总数 | 316409 | 24490 | 159613 |
| clipped | 44522 | 3370 | 22166 |
| empty_label_files | 33 | 2 | 18 |
| src_bndbox | 22917 | 1455 | 9641 |
| src_polygon | 293494 | 23035 | 149975 |

其他报告字段：train的drop_outside_or_tiny=2、drop_point=22、drop_unknown_name:*=1；test的drop_outside_or_tiny=3、drop_point=16；val未列这些字段。训练car/van=270348/7701约35.1:1，van占316409个目标的约2.43%。类频率不平衡得到确认，但不是当前van错误的因果证明；此前几何合格候选、低van分数和共同van→car的证据仍只是选实验的依据。val的1469对/24490目标/725 van与已完成诊断吻合，不能由此排除单个源标签或类别边界问题。

**当前唯一推荐。** 选择Full v2的全局分类损失增益cls由0.5到1.0的受控实验，检验分类项相对box/obj是否偏弱。该轮作为额外诊断/调参实验单独命名归档，不覆盖旧Full v2、不混入原同协议模块消融表。此项影响所有类别的正负分类项，不是van专用类别权重或频率平衡；不保证提升，也不把当前cls绝对数值小当成实现错误。不同时改变shuffle、focal、image-weights、结构、数据或学习率。保留现有Full v2作为参考，训练和评估参数应以旧实验实际保存的配置为准；最终比较同口径独立val的van AP、总体AP及其他类别，不能仅凭conf=0.25的TP增多判成功，小差异仍需重复seed验证。

**本地只读核对。** train.py第970行声明--hyp-set，1056-1059行解析并在调用train_rgb_ir前应用覆盖；cls已存在于v2 hyp，支持cls=1.0。train_rgb_ir在启动时保存覆盖后的hyp.yaml与opt.yaml；第709行对5类、3检测层缩放cls，0.5/1.0分别得到有效增益0.03125/0.0625。没有改源码或配置，不需要重复模型/前后向测试。全局分类增益与其他已记录采样、focal因素分开，后者仍为独立候选，本轮不处理。

**仅交付检查，不启动训练。** 下一步由用户在服务器先看git status，再只读列出runs/train/*/opt.yaml、hyp.yaml，定位旧Full v2实际训练目录并核实初始化、模型、超参和其他运行选项。历史实际运行设置尚未完整回传，不能提前写成新旧完全同条件；核对后再交付从相同COCO初始化、仅cls变化的完整训练命令。当前没有向用户交付开跑命令，没有训练、评估、SSH或修改服务器。无需为继续追XML停住此参数核验。

本轮起点为codex/cmfc-only-v2-ablation的9bb274288ebf892bf9e2f13320d335aa3f4f4225，文档更新前git status确认干净。仅更新CLAUDE.md当前进度/下一步并在WORKLOG.md追加此记录，保留历史内容；文档提交和远端核验由主代理随后处理，不在此预报完成。

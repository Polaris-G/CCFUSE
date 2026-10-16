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

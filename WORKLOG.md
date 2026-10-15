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

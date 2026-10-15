# CCSFuse 复现方案（交给 VS Code 中 Claude 实现用）

> 论文：Zhang et al., *CCSFuse: Collaborative Compensation and Selective Fusion for UAV-Based RGB-IR Object Detection*, IEEE TGRS, vol. 64, 2026, Art. no. 5400314. DOI: 10.1109/TGRS.2025.3647293
> 出版商：IEEE｜下载平台：IEEE Xplore
> 官方代码：github.com/ZhangT-xxl/CCSFuse —— **截至 2026-10 仍未公开代码（仓库只有 README）**，以下全部按论文自行实现。
> 本地硬件：RTX 3070 Ti Laptop（8 GB 显存）

标注约定：
- **【论文】** = 论文中明确写出的内容（附出处：公式/图/表/章节）
- **【推断】** = 论文没写，按常规做法或图示推断，**实现时可改，需在代码里写注释**

---

## 0. 一句话概括

YOLOv5 双流（RGB / IR 各一个 CSPDarknet 主干）检测器，在主干的 stage3/4/5 输出处：
1. 用 **CMFC**（跨模态交叉注意力，Softmax 分支 + ReLU² 分支自适应加权）互相补偿特征，补偿后的特征**回注**到各自主干继续往下走；
2. 用 **AFSF**（跨模态通道注意力 + concat-1×1conv）把两路补偿特征融合成单路 f³/f⁴/f⁵；
3. f³/f⁴/f⁵ 送入标准 YOLOv5 的 FPN/PAN neck + Detect head。

---

## 1. 整体架构（对应论文 Fig. 2、Sec. III-B）

```
RGB ─ stem/stage1/stage2 ─ stage3 ─┐            ┌─ stage4 ─┐            ┌─ stage5 ─┐
                                   ├─ CMFC_3 ───┤          ├─ CMFC_4 ───┤          ├─ CMFC_5
IR  ─ stem/stage1/stage2 ─ stage3 ─┘            └─ stage4 ─┘            └─ stage5 ─┘
                                   │                       │                       │
                              AFSF_3 → f³             AFSF_4 → f⁴             AFSF_5 → f⁵
                                                                   ↓
                                             YOLOv5 neck (FPN+PAN) + Detect head
```

- **【论文】** 基线 YOLOv5，主干 CSPDarkNet53，两路独立主干（Fig. 2）。
- **【论文】** CMFC 在 stage3、4、5 各一个；输出 f^{i,CMFC}_rgb / f^{i,CMFC}_ir 回注各自分支（"closed-loop"），并送入同一 stage 的 AFSF。
- **【论文】** AFSF 输出 f³、f⁴、f⁵ 送入 FPN 和检测头。
- **【推断】回注方式**：Fig. 2 画的是 `stage_i 输出 ⊕ f^{i,CMFC} → stage_{i+1}`，但 CMFC 内部（式 6）已经有残差 `+f`。两种实现：
  - A（推荐，默认）：`stage_{i+1}` 的输入 = `f^{i,CMFC}`（= 原特征 + 补偿量，已含残差）。
  - B：`stage_{i+1}` 的输入 = `stage_i 输出 + f^{i,CMFC}`（严格照图，等于原特征×2 + 补偿量）。
  做成配置开关，先用 A。
- **【推断】YOLOv5 规模**：论文没写 s/m/l。论文 Sec. V 给出参数量 **33.50 M、174.12 GFLOPs、47.5 FPS**。双流 YOLOv5m 的参数量与 33.5M 最接近，**推断为 YOLOv5m**。调试阶段先用 YOLOv5s，跑通后再换 m。
- **【推断】stage 对应 YOLOv5 层**（以 ultralytics/yolov5 v7.0 的 yaml 为准）：stage3 = 第 4 层 C3 输出（P3/8），stage4 = 第 6 层 C3 输出（P4/16），stage5 = 第 9 层 SPPF 输出（P5/32）。YOLOv5m 对应通道为 192 / 384 / 768。

### 工程基础建议【推断】
不要从零搭双流 YOLOv5。建议以现有的**双流 YOLOv5 多光谱检测代码库**为底座（例如 CFT 的开源实现 *multispectral-object-detection*，它本身就是 YOLOv5 双流、带 DroneVehicle/LLVIP 数据加载和 Add 融合基线）。论文 Table I 的 baseline 正好是"YOLOv5 双分支 + 加法融合"，可直接对应。在它的 yaml/common.py 里把融合模块替换为 CMFC + AFSF 即可。

---

## 2. CMFC 模块（论文 Sec. III-C，式 (1)–(6)，Fig. 3）

### 2.1 计算流程（以 RGB 侧为例，IR 侧对称）

输入：f_rgb, f_ir ∈ ℝ^{B×C×H×W}

1. **Q/K/V 线性映射 + 空间缩减（SRA）**——式 (1)【论文】
   - Q_rgb = W^q_rgb · f_rgb（全分辨率 token，N = H·W）
   - K_rgb = W^k_rgb · SR(f_rgb)，V_rgb = W^v_rgb · SR(f_rgb)（缩减后 token，M = (H/r)·(W/r)）
   - IR 侧同理，**RGB 与 IR 各自一套权重**【论文：W^q_rgb ≠ W^q_ir】
   - SR 采用 PVT 的 spatial-reduction attention【论文：引用 [49]】
   - **【推断】** SR = `Conv2d(C, C, kernel=r, stride=r)` + `LayerNorm`（PVT 标准写法）；r 论文未给，建议 stage3/4/5 分别取 **8 / 4 / 2**（8GB 显存下的关键参数）。
   - **【推断】** 多头：论文未写头数，建议 heads = C/64 或固定 8；d = 每头维度。

2. **SIC 分支（Softmax）**——式 (2)【论文】
   `SIC_{rgb←ir} = Softmax(Q_rgb · K_irᵀ / √d)`（RGB 的 query 去查 IR 的 key）

3. **RIS 分支（ReLU²）**——式 (3)【论文】
   `RIS_{rgb←ir} = ReLU(Q_rgb · K_irᵀ / √d)²`

4. **自适应加权 + 乘 V**——式 (4)(5)【论文】
   ```
   w_SIC, w_RIS = softmax([λ1, λ2])        # λ1、λ2 为可学习标量，初始化均为 1
   f_{rgb←ir} = (w_SIC * SIC_{rgb←ir} + w_RIS * RIS_{rgb←ir}) @ V_ir
   ```
   - **【推断】** λ1、λ2 每个 CMFC 模块一组，RGB→IR 与 IR→RGB 两个方向是否共享论文未说明；默认**共享一组**，可做开关。
   - **【推断】** 乘完 V 后通常还有一个输出投影（Linear/1×1 conv），论文公式里没有，但建议加上（PVT/Transformer 惯例），并做成开关。

5. **残差**——式 (6)【论文】
   `f^{CMFC}_rgb = f_{rgb←ir} + f_rgb`，`f^{CMFC}_ir = f_{ir←rgb} + f_ir`

### 2.2 数值稳定性提示【推断】
- ReLU² 分支没有归一化，值可能很大，AMP（fp16）下有溢出风险。建议：注意力分数计算在 **fp32** 中完成（`with torch.autocast(enabled=False)` 包住 QKᵀ 和两个分支），或对 RIS 再除以 M。先按原式实现，出现 NaN 再加。
- 这一"Softmax + ReLU² 双分支 + softmax 归一化可学习权重"的写法，与 CVPR 2024 的 AST（Adaptive Sparse Transformer，ASSA 模块）的自注意力形式一致，可参考其开源实现；CMFC 相当于它的**跨模态版本**（Q 来自本模态、K/V 来自另一模态）。

### 2.3 接口
```python
class CMFC(nn.Module):
    def __init__(self, c, num_heads=8, sr_ratio=4, out_proj=True, share_lambda=True): ...
    def forward(self, f_rgb, f_ir):   # B,C,H,W
        return f_rgb_cmfc, f_ir_cmfc   # B,C,H,W（形状不变）
```

---

## 3. AFSF 模块（论文 Sec. III-D，式 (7)–(10)，Fig. 4）

输入：f^{CMFC}_rgb, f^{CMFC}_ir ∈ ℝ^{B×C×H×W}；输出：f ∈ ℝ^{B×C×H×W}

1. **各模态通道描述**——式 (7)【论文】
   `w_rgb = Sigmoid(Conv1×1(GAP(f^{CMFC}_rgb)))`，IR 同理 → 各 B×C×1×1
   - **【推断】** RGB/IR 各用一个独立的 1×1 conv（C→C）。
2. **跨模态 MLP**——式 (8)【论文】
   拼接 `w = concat(w_rgb, w_ir)` ∈ B×2C×1×1 → `FC2(GELU(FC1(w)))`
   - **【推断】** 式 (8) 两行写法相同，最合理的理解是：MLP 输出 2C，再**按通道切分**为 w̃_rgb（前 C）、w̃_ir（后 C）。
   - **【推断】** FC1 隐藏维度论文未给，建议 2C/4（reduction=4）。
   - **【推断】** 论文 MLP 后没写激活；直接相乘可能出现负权重。默认**不加**（忠实原文），提供 `sigmoid_after_mlp` 开关。
3. **通道校准**——式 (9)【论文】
   `f^c_rgb = f^{CMFC}_rgb ⊗ w̃_rgb`，`f^c_ir = f^{CMFC}_ir ⊗ w̃_ir`
4. **concat-conv 融合**——式 (10)【论文】
   `f = Conv1×1(concat(f^c_rgb, f^c_ir))`，2C → C
   - **【推断】** 用 YOLOv5 的 `Conv`（Conv+BN+SiLU）。

```python
class AFSF(nn.Module):
    def __init__(self, c, reduction=4, sigmoid_after_mlp=False): ...
    def forward(self, f_rgb, f_ir):
        return f_fused   # B,C,H,W
```

---

## 4. 数据集与预处理（论文 Sec. IV-A）

### DroneVehicle（主实验）
- **【论文】** 28,439 对 RGB-IR，5 类：car, truck, freight car, bus, van；去除白边，缩放到 640×512；训练输入 resize 到 640×640。
- **【推断】** 原图 840×712，四周各约 100 px 白边，裁掉后正好 640×512。
- **【推断】标注**：DroneVehicle 原始是旋转框（OBB，xml 多边形），YOLOv5 用水平框（HBB）→ 取多边形外接矩形。
- **【推断】用哪个模态的标签**：论文 Fig. 5 强调 RGB/IR 标注不一致，但没说训练用哪份。大多数基于 YOLOv5 的工作（CFT、CALNet 等）**使用 IR 标注**，默认采用 IR 标注。
- **【推断】** 划分使用官方 train / val / test（17,990 / 1,469 / 8,980 对），在 test 上报告结果。

### LLVIP（次实验）
- **【论文】** 15,488 对严格对齐图像（12,025 训练 / 3,463 测试），单类 person，RGB/IR 共享标注。
- **【推断】** 原图 1280×1024，训练时 resize 到 640。

### 数据增强
- **【论文】** 水平翻转、随机裁剪、颜色变换。
- **【推断】** 关键：**几何增强必须对 RGB 和 IR 同步**（同一随机参数），颜色变换只作用于 RGB（或分别独立）。Mosaic 论文未提，默认**关闭**；如用现成 YOLOv5 双流代码库，需检查其 mosaic 是否成对实现。

---

## 5. 训练配置

| 项目 | 论文设置【论文】 | 本机 8GB 建议【推断】 |
|---|---|---|
| 硬件 | 2 × RTX 4090D | 1 × 3070 Ti Laptop 8GB |
| 框架 | Python 3.8, PyTorch 2.0.1 | PyTorch 2.x + CUDA 11.8/12.x 均可 |
| 输入尺寸 | 640×640 | 640×640（显存不够时退到 512） |
| 预训练 | 主干用 COCO 预训练 | 两路主干都加载 yolov5m.pt 的 backbone 权重 |
| Epoch | 45 | 45 |
| Batch | 6/GPU × 2 = 12 | **4**（+ 梯度累积 3 步，等效 12） |
| 优化器 (DroneVehicle) | SGD, momentum 0.9, lr 0.001 | 同 |
| 优化器 (LLVIP) | Adam, lr 0.0001 | 同 |
| LR 调度 | One-Cycle | 同（YOLOv5 自带 `--cos-lr` 或 OneCycleLR） |
| 混合精度 | 未提 | **开启 AMP**，注意力分数用 fp32 |
| SRA ratio | 未提 | stage3/4/5 = 8/4/2 |

显存估算思路：stage3 在 640 输入下为 80×80 = 6400 个 query；r=8 时 K/V 为 100 个 token，单头注意力矩阵 6400×100，两个方向 × 两个分支，开销可控。**r 不能太小**（r=1 时为 6400×6400，会直接爆显存）。

训练时间预估：DroneVehicle 约 1.8 万对训练图，笔记本上每 epoch 大约 20–40 分钟，45 epoch 约 1–1.5 天。笔记本需保证散热。

---

## 6. 复现目标（用于判断是否成功）

### Table I：模块消融（DroneVehicle，mAP50，%）【论文】
| CMFC | AFSF | mAP | car | truck | Fer. | bus | van |
|---|---|---|---|---|---|---|---|
| | | 72.8 | – | – | – | – | – |
| ✓ | | 76.8 | 96.5 | 68.8 | 64.8 | 93.2 | 60.6 |
| | ✓ | 78.7 | 96.9 | 72.6 | 68.3 | 94.1 | 61.6 |
| ✓ | ✓ | **79.4** | 97.1 | 72.1 | 67.8 | 94.3 | 64.1 |

### Table II：CMFC 双分支消融（DroneVehicle，mAP50）【论文】
| 设置 | w_SIC : w_RIS | mAP |
|---|---|---|
| Baseline | – | 72.8 |
| 仅 Softmax | 1 : 0 | 76.4 |
| 仅 ReLU² | 0 : 1 | 75.6 |
| CMFC 固定权重 | 0.5 : 0.5 | 76.4 |
| CMFC 可学习 | learnable | 76.8 |

### LLVIP（Table VI）【论文】
CCSFuse：mAP50 = **96.8**，mAP50:95 = **68.3**

### 效率（Sec. V）【论文】
33.50 M 参数，174.12 GFLOPs，47.5 FPS（2×4090D）

**判定标准【推断】**：先比较**相对提升**而不是绝对值。如果自己的 baseline 与论文 72.8 有差距，只要 CMFC/AFSF 带来的提升方向和幅度（+4、+5.9、+6.6）接近，就算复现成功。batch 变小，绝对 mAP 差 1–3 点属正常。

---

## 7. 实现顺序与每步验证

1. **搭建双流 YOLOv5 baseline（加法融合）**
   - 验证：单 batch 前向通过；打印 P3/P4/P5 形状；用 DroneVehicle 的 1% 数据过拟合几个 epoch，loss 明显下降。
2. **数据管线**
   - 验证：可视化 10 对样本（RGB、IR、框叠加），确认白边已去除、框对齐、成对增强同步。
3. **完整训练 baseline** → 对标 Table I 第 1 行（72.8）。
4. **实现 CMFC**（单元测试）
   - 输入 `torch.randn(2,192,80,80)` ×2，输出形状不变；
   - 把 λ 设为使 w_SIC=1，验证输出等同于普通交叉注意力；
   - 检查 fp16 下无 NaN/Inf。
5. **实现 AFSF**（单元测试）：形状、梯度能回传到两路输入。
6. **接入网络**，统计参数量/FLOPs（thop），与 33.5M / 174 GFLOPs 对比，检验 YOLOv5 规模推断是否合理。
7. **按 Table I 顺序跑消融**：+CMFC → +AFSF → +CMFC+AFSF。
8. **跑 Table II 的权重策略消融**（固定 1:0、0:1、0.5:0.5、可学习）。
9. **LLVIP 实验**（Adam，lr 1e-4）。
10. 训练中记录 λ1、λ2 的变化曲线，可作为分析材料。

---

## 8. 所有【推断】汇总（实现时全部做成配置项）

| # | 项 | 默认值 | 备选 |
|---|---|---|---|
| 1 | YOLOv5 规模 | m（调试用 s） | s / l |
| 2 | CMFC 回注方式 | A：next = f^CMFC | B：next = f + f^CMFC |
| 3 | SRA ratio | 8 / 4 / 2 | 4 / 2 / 1 |
| 4 | 注意力头数 | 8 | C/64 |
| 5 | CMFC 输出投影 | 有 | 无 |
| 6 | λ 两方向共享 | 共享 | 独立 |
| 7 | RIS 归一化 | 无 | 除以 M |
| 8 | AFSF MLP 隐藏维度 | 2C/4 | 2C/2、2C |
| 9 | AFSF MLP 后 sigmoid | 无 | 有 |
| 10 | DroneVehicle 标签 | IR 标注，HBB | RGB 标注 |
| 11 | Mosaic | 关 | 开（需成对） |
| 12 | 批量 | 4 + 梯度累积 3 | 6（若显存允许） |

---

## 9. 可直接粘贴给 VS Code 中 Claude 的提示词

```
请阅读本仓库根目录的 CCSFuse_复现方案.md，按第 7 节的顺序实现 CCSFuse（YOLOv5 双流 RGB-IR 检测）。
要求：
1. 以现有双流 YOLOv5 多光谱代码库为底座（如没有，先帮我选择并克隆一个），不要从零写 YOLOv5。
2. 第 8 节列出的所有【推断】项都做成 yaml/命令行可配置，并在代码注释中标明"论文未给出，推断"。
3. CMFC、AFSF 各写单元测试（第 7 节第 4、5 步）。
4. 默认配置针对 RTX 3070 Ti Laptop 8GB：batch 4、AMP、注意力分数 fp32、SRA ratio 8/4/2。
5. 每完成一步先停下来，告诉我如何验证，再继续下一步。
```

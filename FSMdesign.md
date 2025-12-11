
# M-Stream V9: Titan-Stream
## —— 基于动量惊奇与全注意力机制的在线时序预测模型

### 1. 概述 (Overview)

**M-Stream V9 (Titan-Stream)** 是一款专为非平稳时间序列设计的在线预测模型。它彻底摒弃了传统的“静态骨干+局部修补”范式，转而采用 **“动态神经记忆 + 上下文引导 (Memory As Context, MAC)”** 的全新架构。

**核心设计哲学：**
1.  **泛在注意力 (Ubiquitous Attention)**：所有模块（记忆、先验、推理）均基于 Attention 变体构建，确保梯度流的通畅与算子的高效复用。
2.  **惊奇度驱动 (Surprise-Driven)**：利用预测误差的梯度动量（Gradient Momentum）来驱动神经记忆的实时演化，捕捉趋势惯性。
3.  **门控即遗忘 (Gating as Forgetting)**：通过输入依赖的门控机制动态调节记忆的衰减，防止显存溢出与灾难性遗忘。

---

### 2. 模块详细设计 (Module Detail Design)

模型整体采用 **MAC (Memory As Context)** 的三流并行架构。

#### 2.1 模块一：神经长时记忆 (Neural Long-term Memory)
* **角色**：**快系统 (Fast System)** —— 负责捕捉概念漂移、非平稳趋势和近期流式历史。
* **基础机制**：**Linear Attention (Recurrent Mode)**。
* **数学定义**：
    我们将记忆视为一个动态变化的权重矩阵 $M_t \in \mathbb{R}^{D \times D}$。这本质上是 Linear Attention 的循环状态形式。
    $$\text{Linear Attn Output} = Q \cdot (K^T V) \rightarrow M = \sum K^T V$$

* **输入/输出流**：
    * **Input**: 当前滑窗数据的 Embedding $X_t \in \mathbb{R}^{L \times D}$。
    * **Projections**:
        * $Q_{mem} = X_t W_Q$
        * $K_{mem} = X_t W_K$
        * $V_{mem} = X_t W_V$
    * **Output**: 动态上下文特征 $H_{mem} = M_{t-1} \cdot Q_{mem}$ (利用上一时刻的记忆预测当前)。

    **[实现说明 - 修复 2.3]**:
    - 这里的 $H_{mem} = M \cdot Q$ 表示**记忆作为线性变换矩阵**，将 Query 映射到上下文空间。
    - 这与标准 Linear Attention 的 $H = Q \cdot M$ 不同，后者是 Query 左乘记忆。
    - 我们的设计选择使得记忆 $M$ 扮演"特征变换器"的角色，而非"键值缓存"。
    - 这种设计允许记忆直接调制输入特征，更适合捕捉非平稳的动态变换。

* **内部组件设计**：
    1.  **记忆状态 (Memory State)**: 一个可学习的矩阵 $M$, 初始化为零或随机小值。
    2.  **门控网络 (Forgetting Gate)**: 一个轻量级 MLP，用于计算衰减系数 $\alpha_t$。
        * $\alpha_t = \text{Sigmoid}(\text{MLP}(X_t)) \in (0, 1)$。
    3.  **动量缓冲 (Momentum Buffer)**: 存储梯度动量的矩阵 $S$, 形状同 $M$。

#### 2.2 模块二：持久记忆 (Persistent Memory)
* **角色**：**慢系统 (Slow System)** —— 负责存储固定的先验知识（如周期性、物理约束），作为“锚点”防止模型跑偏。
* **基础机制**：**Cross Attention (Static KV)**。
* **结构设计**：
    * 不涉及复杂的更新逻辑，由一组**可学习参数**组成。
    * **Keys ($K_{pers}$)**: $\mathbb{R}^{N_p \times D}$，代表 $N_p$ 个典型的时序模式原型。
    * **Values ($V_{pers}$)**: $\mathbb{R}^{N_p \times D}$，代表对应的特征响应。
* **交互逻辑**：
    * 使用 Core Forecaster 的 Query 去查询这些静态 Key。
    * $H_{pers} = \text{Softmax}(\frac{Q_{core} K_{pers}^T}{\sqrt{D}}) V_{pers}$。
    * *注：在线测试阶段，这部分参数通常冻结或仅接受极低学习率微调。*

#### 2.3 模块三：核心预测器 (Core Forecaster)
* **角色**：**融合引擎 (Fusion Engine)** —— 负责统筹长期规律（持久记忆）、近期趋势（神经记忆）和当前状态，生成最终预测。
* **基础机制**：**Full Self-Attention (Transformer Decoder)**。
* **输入构造 (Prompting)**：
    我们将三种信息拼接，构建一个增强的输入序列：
    $$\text{Input}_{core} = [\underbrace{H_{pers}}_{\text{长期先验}}, \underbrace{H_{mem}}_{\text{近期趋势}}, \underbrace{X_{t}}_{\text{当前观测}}]$$
* **处理逻辑**：
    * 通过多层 Self-Attention 层进行特征交互。
    * Attention 机制会自动根据数据的平稳性动态分配权重（例如：数据平稳时关注 $H_{pers}$，突变时关注 $H_{mem}$）。
* **输出头**：
    * $Y_{pred} = \text{Linear}(\text{Output}_{core})$。

---

### 3. 在线学习机制：惊奇度与动量 (Surprise & Momentum)

这是 V9 版本的灵魂，定义了神经记忆 $M_t$ 如何在测试时（Test-Time）进行更新。

#### 3.1 惊奇度 (Surprise) 定义
惊奇度被定义为**神经记忆对当前数据的“重建误差”的梯度**。如果记忆能很好地解释当前数据，惊奇度低；反之则高。
* **代理任务 (Proxy Task)**: 重建 Value。
    $$\hat{V}_{mem} = M_{t-1} \cdot K_{mem}$$
* **代理损失 (Proxy Loss)**:
    $$\mathcal{L}_{proxy} = || \hat{V}_{mem} - V_{mem} ||^2$$
* **惊奇度 (Instant Surprise)**: Loss 对记忆状态 $M$ 的梯度。
    $$G_t = \nabla_{M} \mathcal{L}_{proxy} \approx (\hat{V}_{mem} - V_{mem}) \cdot K_{mem}^T$$

#### 3.2 动量更新 (Momentum Update)
为了避免噪声干扰并捕捉趋势，我们不直接使用 $G_t$ 更新，而是更新动量 $S_t$。
$$S_t = \beta \cdot S_{t-1} + (1 - \beta) \cdot G_t$$
* $\beta$: 动量系数（如 0.9），决定了记忆的“惯性”。

#### 3.3 门控更新 (Gated Update Rule)
结合遗忘门和动量进行最终更新：
$$M_t = (1 - \alpha_t) \cdot M_{t-1} - \eta \cdot S_t$$
* $\alpha_t$: 遗忘率，由门控网络根据当前输入 $X_t$ 动态生成。
* $\eta$: 学习率 (Step Size)。

---

### 4. 训练与测试流程 (Training & Inference Flow)

#### 4.1 离线训练阶段 (Offline Training)
**目标**：训练 Core Forecaster 的预测能力，初始化 Persistent Memory，并教会 Neural Memory 如何通过梯度更新自己（Meta-Learning）。

1.  **数据准备**：标准的 Batch 数据 $(X, Y)$。
2.  **前向传播 (Forward)**：
    * 初始化 $M_0 = 0, S_0 = 0$。
    * **Chunk 循环**：将长序列切分为多个 Chunk（模拟在线流）。
        * 对每个 Chunk $t$：
            * 生成 $H_{pers}$ (查表) 和 $H_{mem}$ ($M_{t-1}$ 生成)。
            * Core Forecaster 预测 $\hat{Y}$。
            * **模拟在线更新**：计算 $\mathcal{L}_{proxy}$，计算梯度 $G_t$，更新 $S_t$ 和 $M_t$。*注意：这一步需要保留梯度图 (Gradient through Gradient)，类似于元学习或 RNN BPTT。*
3.  **损失计算**：
    * 主预测损失：$\mathcal{L}_{pred} = \text{MSE}(\hat{Y}, Y)$。
    * 辅助损失（可选）：$\mathcal{L}_{proxy}$ 确保 Memory 学会重建特征。
4.  **反向传播**：更新所有参数（包括 $W_Q, W_K, W_V$, MLP, $K_{pers}, V_{pers}$, Core Parameters）。

#### 4.2 在线推理/测试阶段 (Online Inference)
**目标**：在真实流式环境中，利用延迟反馈实时修正 $M_t$。

**初始化**：
* 加载离线训练好的模型参数。
* 初始化 $M = M_{final\_train}$ (或零), $S = 0$。
* Persistent Memory 参数 **冻结**。

**时间步循环 ($t = 1, 2, \dots$)**：

1.  **Step 1: 记忆检索与预测 (Predict)**
    * 接收当前窗口 $X_t$。
    * **神经记忆**：$H_{mem} = M_{t-1} K_{mem}$。
    * **持久记忆**：$H_{pers} = \text{CrossAttn}(X_t, K_{pers}, V_{pers})$。
    * **核心预测**：$\hat{Y}_t = \text{Core}([H_{pers}, H_{mem}, X_t])$。
    * 输出预测结果。

2.  **Step 2: 延迟反馈处理 (Feedback)**
    * 等待真实标签 $Y_{true}$ (通常延迟 $H$ 步到达)。
    * 如果收到标签（或部分标签）：
        * 计算预测误差或代理误差 $\mathcal{L}$。
        * **计算惊奇度**：$G = \nabla_{M} \mathcal{L}$。
        * **更新动量**：$S \leftarrow \beta S + (1-\beta) G$。
        * **计算门控**：$\alpha = \text{GateNet}(X_t)$。
        * **更新记忆**：$M \leftarrow (1-\alpha) M - \eta S$。

---

### 5. 关键超参数 (Key Hyperparameters)

* `d_model`: 隐层维度 (e.g., 128)。
* `n_persistent`: 持久记忆 Token 数量 (e.g., 32)。
* `beta_momentum`: 动量系数 (e.g., 0.9)。
* `learning_rate_memory`: 在线记忆更新率 $\eta$ (e.g., 0.01)。
* `chunk_size`: 离线训练时的 BPTT 长度。

这个文档涵盖了从宏观架构到微观公式的所有细节，您可以直接基于此文档构建 `Model.py` (定义类结构) 和 `Exp.py` (定义训练/测试循环)。
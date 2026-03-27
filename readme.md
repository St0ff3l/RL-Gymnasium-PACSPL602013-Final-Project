Markdown
# RL-Gymnasium-PACSPL602013 Final Project

## 🛠️ 1. 开发者原始配置 (Developer's Reference)
* **操作系统**: Windows 11
* **显卡 (GPU)**: NVIDIA GeForce **RTX 3070** (8GB VRAM)
* **处理器 (CPU)**: Intel Core **i7-12700H**
* **Python 版本**: **3.12.x** (注：建议避开 3.13 以免 CUDA 库不兼容)
* **虚拟环境名**: `final_project_env`

---

## 🚀 2. 环境重建步骤 (Step-by-Step Setup)

### 第一步：创建并激活环境
> **建议安装 Python 3.12 后执行**

```powershell
python -m venv final_project_env
.\final_project_env\Scripts\activate
```

### 第二步：安装 GPU 加速版 PyTorch
> 这是确保 RTX 3070 发挥性能的关键。如果你没有 NVIDIA 显卡，可以跳过此步直接执行第三步（将使用 CPU 运行）。

```PowerShell
# 针对 CUDA 12.1 的安装指令
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### 第三步：一键安装项目依赖
```PowerShell
pip install -r requirements.txt
```

### 第四步：本地 VLM (Ollama) 配置
> 本项目创新部分依赖本地大模型推理，请确保安装了 Ollama 并准备好模型：

```PowerShell
ollama pull llava:7b
```

## ⚠️ 注意事项 (Important Notes)
Git 忽略: 请勿将 final_project_env/ 文件夹上传至 Git 仓库，该目录已包含在 .gitignore 中。

模型路径: 训练产生的模型和日志将存放在 logs_and_results/ 文件夹下。

Kernel 选择: 在 VS Code 中打开 .ipynb 文件时，请务必在右上角选择 final_project_env (Python 3.12.x) 作为内核。

---

## 📁 3. 训练目录与命令行管理

训练结果现在按算法和 run 分目录保存，结构如下：

```text
logs_and_results/
├── baseline/
│   └── run_N/
│       ├── models/
│       ├── videos/
│       ├── data/
│       └── tensorboard/
└── vlm/
	└── run_N/
		├── models/
		├── videos/
		├── data/
		└── tensorboard/
```

新增命令行工具 `train_manager.py` 用于管理 run：

```powershell
# 列出 baseline 的所有 run
python train_manager.py --algo baseline list

# 创建一个新 run（自动编号）
python train_manager.py --algo baseline create

# 创建指定 run（例如 run_10）
python train_manager.py --algo baseline create --run-id 10

# 汇总统计：指定列表
python train_manager.py --algo baseline summary --ids 1,2,3

# 汇总统计：指定范围
python train_manager.py --algo baseline summary --range 1-5

# 清理 run：指定列表
python train_manager.py --algo baseline clean --ids 2,3

# 清理 run：指定范围
python train_manager.py --algo baseline clean --range 6-8
```

将 `--algo baseline` 改成 `--algo vlm` 即可管理 VLM 目录。

---

## 📊 4. Final Report 交互使用说明

`algorithms/final_report.ipynb` 已支持交互输入：

1. Cell 1（视频对比）
输入 baseline 和 vlm 的单个 run id（回车默认最新）。

2. Cell 2（阈值对比 + 平均柱状图）
先输入一次共同 run 选择（如 `1-3` 或 `1,2,4`），用于展示对比。
再输入一次共同平均范围（如 `2-3`），用于计算平均值。

3. Cell 3（训练曲线 + 平均曲线）
与 Cell 2 相同：先选展示 run，再选平均 run。

### 为什么要选两次？

因为“展示集合”和“平均集合”是两个不同用途：

- 第一次选择：决定图里显示哪些 run。
- 第二次选择：决定哪些 run 参与平均计算。

示例：你可以显示 `1-6` 的所有曲线，但只用 `2-6` 做平均，排除 `run_1` 的异常波动。

如果你不需要分开控制，第二次直接回车即可，程序会默认使用第一次选择的同一组 run。
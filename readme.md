# RL-Gymnasium-PACSPL602013 Final Project

This project compares Baseline PPO and VLM-PPO training performance on MountainCar-v0.

## Quick Start

1. Create and activate the environment (Python 3.12)

```powershell
python -m venv final_project_env
.\final_project_env\Scripts\activate
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. If you use VLM, prepare the local model

```powershell
ollama pull llava:7b
```

## Run

Option A: run the full pipeline

```powershell
python main.py
```

Option B: run notebooks manually in VS Code (in order)

1. algorithms/baseline_mountain_car.ipynb
2. algorithms/ours_vlm_ppo.ipynb
3. algorithms/final_report.ipynb

## Output

Training logs, models, and results are saved in:

- logs_and_results/

## Notes

- Python 3.12 is recommended.
- When opening notebooks, select final_project_env as the kernel.
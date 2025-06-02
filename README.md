# ♟️ Chess AI Trainer with PPO and GPU Support

This project implements a reinforcement learning agent that learns to play chess using the **Proximal Policy Optimization (PPO)** algorithm from **Stable-Baselines3**, accelerated using **GPU** via Google Colab.

It features:
- Custom Chess Gym Environment (`cchess_env.py`)
- GPU-based training using PyTorch and PPO
- Colab Notebook for easy training
- Sample code to test the trained model
- Video demo

---

## Files Included

| File | Description |
|------|-------------|
| `chess_env.py` | Custom OpenAI Gym-compatible environment for chess |
| `aichesscolab.ipynb` | Google Colab notebook for training PPO on chess with GPU |
| `train_chess_gpu.py` | Python script for training locally or on a remote machine |
| `requirements.txt` | Required dependencies for running the project |

---

## Colab Notebook

To run the project with GPU acceleration, open the notebook below:

[Open in Google Colab](https://colab.research.google.com/drive/1tnD59rR-oXFkLoVbnLn1068mkcEu-fth?usp=sharing)

*Make sure to select `Runtime > Change runtime type > GPU` before running.*

---

## Demo Video

Check out a demo of the AI learning and playing chess:

[Watch Demo](https://drive.google.com/drive/folders/1HY-yz5OggWu5cY6GkqQW_2y4JTIDnUhb?usp=drive_link)

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/PJT-2011/chess-ai-rl.git
cd chess-ai-ppo-gpu
pip install -r requirements.txt

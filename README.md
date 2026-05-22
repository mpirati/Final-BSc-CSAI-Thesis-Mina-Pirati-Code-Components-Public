# Final-BSc-CSAI-Thesis-Mina-Pirati-Code-Components
Deep reinforcement learning for autonomous navigation in USAR environments. Training, evaluation, and hyperparameter search in a 2D environment, matching a Gazebo environment.


# Deep Reinforcement Learning for USAR Navigation

This repository contains the code, results, and figures accompanying the BSc thesis:

> *"Autonomous navigation in Urban Search and Rescue environments: A comparison of Deep Reinforcement Learning policies"*  
> Mina Pirati, Tilburg University, 2026

---

## Overview

The project trains and evaluates five model-free deep reinforcement learning algorithms, namely DDPG, TD3, SAC, PPO, and TRPO, alongside a random baseline, on a custom 2D navigation environment. The environment is a surrogate of a 3D Gazebo simulation of the OpenRobotics "collapsed house" model, using the Unitree Go2 as the reference platform.

The agent must navigate from a randomly spawned start position to a goal (representing a victim's location) while avoiding walls and 10 randomly placed cylindrical obstacles, using only 64-beam LiDAR and goal information as observations.

---

## Repository Structure
```
├── training_evaluation.py       # Main training and evaluation script (parallel, multiprocessing)
├── hyperparameter_search.py     # Random hyperparameter search (run on university GPU cluster)
├── graphs_edit.ipynb            # Regenerated plots with updated colour palette
├── raw results/        # Per-agent .pkl result files from full training
│   ├── results_Random.pkl
│   ├── results_DDPG.pkl
│   ├── results_TD3.pkl
│   ├── results_SAC.pkl
│   ├── results_PPO.pkl
│   └── results_TRPO.pkl
├── model_weights/               # Trained agent weights from full training
│   ├── DDPG_model.zip
│   ├── TD3_model.zip
│   ├── SAC_model.zip
│   ├── PPO_model.zip
│   └── TRPO_model.zip
└── figures/                     # All figures used in the thesis
    ├── fig1_learning_curves.pdf/png
    ├── fig2_final_return_boxplot.pdf/png
    ├── fig3_steps_violin.pdf/png
    ├── fig4_success_collision_rates.pdf/png
    ├── fig_episode_breakdown.pdf/png
    └── fig_collision_intensity.pdf/png
```
---

## Environment

The `Continuous2DNavEnv` class is a custom Gymnasium environment modelling a 2D projection of the Gazebo "collapsed house" world.

| Property | Value |
|---|---|
| World size | 16.2 × 13.0 m |
| Observation space | 68-dim (64 LiDAR beams + goal distance, bearing, sin/cos heading) |
| Action space | 2-dim continuous (forward speed, angular velocity) |
| Episode horizon | 1500 steps |
| Success criterion | Robot centre within 0.3 m of goal |
| Obstacles | 10 randomly placed cylindrical obstacles (r = 0.35 m) |

---

## Algorithms

All algorithms are implemented via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib). Hyperparameters were selected via random search (Bergstra & Bengio, 2012) over 10 trials per algorithm, trained for 200,000 steps each, and scored by success rate on a fixed evaluation seed.

| Algorithm | Type | Reference |
|---|---|---|
| DDPG | Off-policy, actor-critic | Lillicrap et al. (2015) |
| TD3 | Off-policy, actor-critic | Fujimoto et al. (2018) |
| SAC | Off-policy, maximum entropy | Haarnoja et al. (2018) |
| PPO | On-policy, clipped surrogate | Schulman et al. (2017) |
| TRPO | On-policy, trust region | Schulman et al. (2015) |
| Random | Baseline | — |

---

## Requirements

```bash
pip install stable-baselines3 sb3-contrib gymnasium matplotlib seaborn pandas torch
```

Python 3.9+ is recommended. Full training was run on CPU using Python multiprocessing. Hyperparameter search was run on a university GPU cluster.

---

## Reproducing Results

**Hyperparameter search:**
```bash
python hyperparameter_search.py
```

**Full training and evaluation:**
```bash
python training_evaluation.py
```
Results are saved as `.pkl` files in `results_corrected_hp/`. Figures are saved as `.pdf` and `.png`.

**Regenerate figures from existing results:**
Navigate to the repository directory first:
```bash
cd path/to/Final-BSc-CSAI-Thesis-Mina-Pirati-Code-Components
```
Open and run `graphs_edit.ipynb` in Jupyter to regenerate figures from existing results. Note that the file paths inside the notebook point to the original directory 
and will need to be updated to match your local path.

---


## Pretrained Weights

Trained model weights are provided in `model_weights/` for all five RL agents.
These can be loaded directly using Stable-Baselines3:

```python
from stable_baselines3 import DDPG, TD3, SAC, PPO
from sb3_contrib import TRPO

model = SAC.load("model_weights/SAC_model")
```

To evaluate a pretrained model without retraining, load the weights and pass
the agent to `run_episode()` in `training_evaluation.py`.
Note: Random has no weights as it does not learn.

---

## Results Summary

| Metric | Random | DDPG | TD3 | SAC | PPO | TRPO |
|---|---|---|---|---|---|---|
| Mean Reward | -1185.90 | -187.60 | -101.12 | -359.64 | -530.11 | -63.27 |
| Reward Std | 346.14 | 589.22 | 599.81 | 749.59 | 796.31 | 295.05 |
| Median Reward | -1300.73 | 32.37 | 173.63 | 186.87 | 158.40 | -7.36 |
| Max Reward | -158.33 | 265.83 | 387.96 | 387.49 | 387.37 | 54.67 |
| Min Reward | -1984.84 | -1507.50 | -1507.50 | -1507.50 | -1509.38 | -1507.50 |
| Mean Reward (success) | --- | 217.84 | 233.23 | 241.71 | 231.72 | --- |
| Mean Reward (fail) | -1185.90 | -361.35 | -526.66 | -1124.99 | -1355.43 | -63.27 |
| Mean Steps | 1500.00 | 1071.26 | 722.48 | 723.76 | 778.36 | 1500.00 |
| Steps Std | 0.00 | 654.98 | 701.41 | 690.80 | 693.88 | 0.00 |
| Median Steps | 1500.00 | 1500.00 | 129.50 | 149.00 | 189.00 | 1500.00 |
| Mean Steps (success) | --- | 70.87 | 111.57 | 113.86 | 112.23 | --- |
| Success Rate | 0.00 | 0.30 | 0.56 | 0.56 | 0.52 | 0.00 |
| Collision Rate | 1.00 | 0.32 | 0.28 | 0.50 | 0.52 | 0.04 |
| Timeout Rate | 1.00 | 0.70 | 0.44 | 0.44 | 0.48 | 1.00 |
| Mean Collision Count | 1166.02 | 288.26 | 260.20 | 518.38 | 686.34 | 60.00 |
| Collision Std | 321.24 | 575.91 | 554.84 | 662.63 | 714.52 | 293.94 |
| Mean Coll (success) | --- | 0.80 | 0.32 | 2.07 | 0.38 | --- |
| Mean Coll (fail) | 1166.02 | 411.46 | 590.95 | 1175.50 | 1429.46 | 60.00 |
| Collision Intensity | 0.7773 | 0.2691 | 0.3601 | 0.7162 | 0.8818 | 0.0400 |
| Catastrophic (>1000) | 38 | 10 | 9 | 17 | 24 | 2 |
| Wandering (=0, fail) | 0 | 21 | 10 | 2 | 0 | 48 |
| Risk-Adjusted Score | -0.76 | 0.10 | 0.38 | 0.22 | 0.04 | -0.04 |

---

## References

- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., . . .Wierstra, D. (2015). Continuous control with deep reinforcement learning. doi: 10.48550/arXiv.1509.02971
- Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing function approximation error in actor-critic methods. doi: 10.48550/arXiv.1802.09477
- Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. doi: 10.48550/arXiv.1801.01290
- Schulman, J., Levine, S., Moritz, P., Jordan, M. I., & Abbeel, P. (2017). Trust region policy optimization. doi: 10.48550/arXiv.1502.05477
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. doi: 10.48550/arXiv.1707.06347
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13(10), 281–305. Retrieved from http://jmlr.org/papers/v13/bergstra12a.html
- OpenRobotics. (2023). Collapsed house. Retrieved from https://fuel.gazebosim.org/1.0/OpenRobotics/models/Collapsed%20House
- Gabr, K. (2024). Unitree go2 ros2: A ros 2 jazzy integration for the unitree go2 quadrupedal robot. GitHub. Retrieved from https://github.com/khaledgabr77/unitree_go2_ros2
---

## License

This repository is submitted as part of a BSc thesis at Tilburg University. Code is provided for academic reproducibility. Please cite accordingly if reused.

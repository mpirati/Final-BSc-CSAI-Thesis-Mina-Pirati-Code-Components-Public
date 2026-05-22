# -*- coding: utf-8 -*-

"""
This script was used to find the best hyperparameter configurations. It was run on the university's GPU cluster. 
It runs the hyperparameter search in the same 2D environment as the training and evaluation script. 
It runs a random search, where 10 configurations are sampled per algorithm from the search space. 
The search space justification is given in Section 3.4 and Appendix 6. 
The best configuration is selected based on success rate on a fixed evaluation seed. These are then used in the full training. 
"""


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import heapq
import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import stable_baselines3 as sb3
from sb3_contrib import TRPO
import random


EVAL_EPISODES = 10 # episodes per config evaluation 
MAX_EP_STEPS = 1500 # matches full training 
N_TRIALS     = 10 # random search budget per algorithm 
TRAIN_STEPS = 200000 # time steps per trial 


def simulate_lidar_torch(robot_pos, robot_theta, obstacles_lidar, walls,
                         num_beams=64, fov=2*np.pi, max_range=13.0):
  
  """
  Simulates LiDAR sensor on Unitree Go2. 
  It makes use of ray-casting LiDAR, using the slab method and analytic ray-circle intersection for round obstacles. 

  Its inputs include: 
  - robot_pos: (2, ) array 
  - robot_theta: robot heading 
  - obstacle_lidar: list of [cx, cy, radius]
  - walls: list of (x_min, x_max, y_min, y_max) Axis-Aligned Bounding Boxes (AABBs)
  - num_beams: number of LiDAR beams (64) 
  - fov: total field of view [rad]
  - max_range: sensor maximum range [m]

  It returns a range of hit distances of the LiDAR beams, clipped to max_ranges
  """

  rx, ry = robot_pos[0], robot_pos[1]

  # generate beam angles
  angles = np.linspace(-fov/2, fov/2, num_beams)
  beam_angles = robot_theta + angles

  # direction vectors for each beam
  dx = np.cos(beam_angles)
  dy = np.sin(beam_angles)

  ranges = np.full(num_beams, max_range)

  # check walls using slab method
  for x_min, x_max, y_min, y_max in walls:
    for i in range (num_beams):
      if abs(dx[i]) > 1e-9:
        t_x_enter = (x_min - rx) / dx[i]
        t_x_exit  = (x_max - rx) / dx[i]
      else:
        t_x_enter = -np.inf
        t_x_exit  = np.inf
      if abs(dy[i]) > 1e-9:
        t_y_enter = (y_min - ry) / dy[i]
        t_y_exit  = (y_max - ry) / dy[i]
      else:
        t_y_enter = -np.inf
        t_y_exit = np.inf
      t_enter = max(min(t_x_enter, t_x_exit), min(t_y_enter, t_y_exit))
      t_exit  = min(max(t_x_enter, t_x_exit), max(t_y_enter, t_y_exit))
      if t_enter < t_exit and t_enter > 0:
          ranges[i] = min(ranges[i], t_enter)

  # check obstacles using slab method (circular)
  for obs in obstacles_lidar:
    cx, cy, radius = obs[0], obs[1], obs[2]
    for i in range (num_beams):
      ocx = rx - cx
      ocy = ry - cy
      b = 2 * (dx[i]*ocx + dy[i]*ocy)
      c = ocx**2 + ocy**2 - radius**2
      disc = b**2 - 4*c
      if disc >= 0:
        t = (-b - np.sqrt(disc)) / 2
        if t > 0:
            ranges[i] = min(ranges[i], t)

  return np.clip(ranges, 0, max_range)



class Continuous2DNavEnv(gym.Env):
    """
    Custom 2D environment, inherited from Gymnasium. 
    It models a 2D projection of the Gazebo "collapsed house" model. 

    The observation space has 68 dimensions, including 64 LiDAR beams, goal distance, goal bearing, and sin and cos of robot heading. 

    The action space is continuous and consists of 2 dimensions. It has a linear velocity, which is unidirectional (forward-only), and an angular velocity 
    """
    MAX_RANGE = 13.0   # LiDAR maximum range

    def __init__(self, render_mode=None, world_size=(16.2, 13.0), num_obstacles=10,
                 obstacle_size=0.35, max_step_size=0.2, random_map=True,):
        super().__init__()
        self.world_size   = np.array(world_size, dtype=float)
        self.num_obstacles = num_obstacles
        self.obstacle_size = obstacle_size
        self.render_mode   = render_mode
        self.fig, self.ax  = None, None
        self.max_step_size = max_step_size # Later in the main guard overridden as 0.044. 
        self.random_map    = random_map
        self.done_threshold = 0.3 # range which classifies success
        self.robot_length = 0.60
        self.robot_width  = 0.32

        # Action: [linear_velocity, angular_velocity]
        # linear velocity = maps to forward speed ∈ [0, max_step_size] (unidirectional)
        # angular velocity ∈ [-0.25, 0.25] rad/step
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(2,), dtype=np.float32)

        # Observation: 68 observation dimensions = LiDAR(64) + [goal_dist, goal_angle, sin θ, cos θ]
        self.num_beams = 64
        obs_dim = self.num_beams + 4
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(obs_dim,), dtype=np.float32)

        # diagonal of world (normalisation constant)
        self.max_dist = float(np.linalg.norm(self.world_size))

    def in_wall(self, pos):
      """
      Prevents robot and obstacle from spawning inside the walls.
      Return True if pos falls inside any wall AABB.
      """
      return any(x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max
                 for x_min, x_max, y_min, y_max in self.walls)

    def dist_to_nearest_wall(self, pos):
        """
        Ensures that the robot is not too close to the wall. 
        Function was added after robot got stuck and kept colliding because it spawned too close to the wall.
        Return the minimum Euclidean distance from pos to any wall surface.
        """
        min_dist = np.inf
        for x_min, x_max, y_min, y_max in self.walls:
            dx = max(x_min - pos[0], 0, pos[0] - x_max)
            dy = max(y_min - pos[1], 0, pos[1] - y_max)
            min_dist = min(min_dist, np.sqrt(dx**2 + dy**2))
        return min_dist

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.obstacles   = []


        # Wall layout manually extracted from Gazebo world SDF.
        # Coordinates are (x_min, x_max, y_min, y_max) in metres.
        self.walls =  [
          (0.0,  3.1,  0.0,  7.0),   # left outer wall
          (14.1, 16.2, 0.0,  7.0),   # right outer wall
          (3.1,  5.1,  6.9,  7.1),   # horizontal wall segment 1
          (6.6,  9.6,  6.9,  7.1),   # horizontal wall segment 2
          (11.1, 14.1, 6.9,  7.1),   # horizontal wall segment 3
          (8.0,  8.2,  7.0,  13.0),  # vertical interior wall
          (3.1,  4.1,  0.0,  1.5),   # dent
        ]
        # areas where navigation is possible
        restrictions = [
            (3.1, 14.1, 7.0, 13.0),
            (3.1, 14.1, 0.0, 7.0)
        ]

        # Rejection sampling: resample until position is free of walls,
        # sufficiently far from robot start, and not inside any obstacle.

        if self.random_map:
            self.robot_theta = self.np_random.uniform(-np.pi, np.pi)
            # robot position spawning
            for _ in range(500):
              region = self.np_random.integers(0, len(restrictions))
              x_min, x_max, y_min, y_max = restrictions[region]
              self.robot_pos = np.array([
                  self.np_random.uniform(x_min, x_max),
                  self.np_random.uniform(y_min, y_max)
              ])
              if not self.in_wall(self.robot_pos):
                  break
            # goal_position spawning
            for _ in range(500):
              region = self.np_random.integers(0, len(restrictions))
              x_min, x_max, y_min, y_max = restrictions[region]
              self.goal_pos = np.array([
                  self.np_random.uniform(x_min, x_max),
                  self.np_random.uniform(y_min, y_max)
              ])
              if not self.in_wall(self.goal_pos) and np.linalg.norm(self.goal_pos - self.robot_pos) > 2.0:
                  break

            # obstacles spawning

            for _ in range(self.num_obstacles):
                region = self.np_random.integers(0, len(restrictions))
                for _ in range(200):   # rejection sampling
                  x_min, x_max, y_min, y_max = restrictions[region]
                  pos = np.array([
                  self.np_random.uniform(x_min + self.obstacle_size, x_max - self.obstacle_size),
                  self.np_random.uniform(y_min + self.obstacle_size, y_max - self.obstacle_size)
              ])
                  if (np.linalg.norm(pos - self.robot_pos) > 1.0 and
                          np.linalg.norm(pos - self.goal_pos)  > 1.0) and not self.in_wall(pos) and self.dist_to_nearest_wall(pos) > self.obstacle_size:
                      self.obstacles.append(pos)
                      break
        else:
            self.robot_pos   = np.array([4.0, 2.0])
            self.goal_pos    = np.array([12.0, 2.0])
            self.robot_theta = 0.0
            self.walls =  [
              (0.0,  3.1,  0.0,  7.0),   # left outer wall
              (14.1, 16.2, 0.0,  7.0),   # right outer wall
              (3.1,  5.1,  6.9,  7.1),   # horizontal wall segment 1
              (6.6,  9.6,  6.9,  7.1),   # horizontal wall segment 2
              (11.1, 14.1, 6.9,  7.1),   # horizontal wall segment 3
              (8.0,  8.2,  7.0,  13.0),  # vertical interior wall
              (3.1,  4.1,  0.0,  1.5),   # dent
            ]
            self.obstacles = [
                np.array([5.0, 2.5]),
                np.array([7.0, 4.0]),
                np.array([10.0, 2.0]),
                np.array([12.0, 5.0]),
                np.array([5.0, 9.0]),
                np.array([10.0, 10.0]),
                np.array([6.0, 11.0]),
                np.array([12.0, 9.0]),
            ]

        self.prev_dist = np.linalg.norm(self.robot_pos - self.goal_pos)
        return self._get_obs(), {}
    

    
    def _robot_corners(self, pos=None, theta=None):
        """
        Return the four corners of the robot rectangle in world frame.
        """
        if pos is None: pos = self.robot_pos
        if theta is None: theta = self.robot_theta
        half_l, half_w = self.robot_length / 2, self.robot_width / 2
        c, s = np.cos(theta), np.sin(theta)
        offsets = np.array([[ half_l, half_w], [ half_l,-half_w],
                            [-half_l,-half_w], [-half_l, half_w]])
        return pos + offsets @ np.array([[c,-s],[s,c]]).T

    def _rect_vs_circle(self, obs_pos, obs_radius):
        """
        Return True if the robot rectangle overlaps a circular obstacle.
        """
        dx = obs_pos[0] - self.robot_pos[0]
        dy = obs_pos[1] - self.robot_pos[1]
        c, s = np.cos(self.robot_theta), np.sin(self.robot_theta)
        lx =  c*dx + s*dy
        ly = -s*dx + c*dy
        cx = np.clip(lx, -self.robot_length/2, self.robot_length/2)
        cy = np.clip(ly, -self.robot_width/2,  self.robot_width/2)
        return np.sqrt((lx-cx)**2 + (ly-cy)**2) < obs_radius

    def _rect_vs_wall(self, pos=None, theta=None):
        """
        Return True if any robot corner is inside an axis-aligned wall rectangle.
        """
        corners = self._robot_corners(pos, theta)
        for x_min, x_max, y_min, y_max in self.walls:
            if any(x_min <= cx <= x_max and y_min <= cy <= y_max
                  for cx, cy in corners):
                return True
        return False

    def step(self, action):
        prev_theta = self.robot_theta
        prev_pos = self.robot_pos.copy()
        action = np.clip(action, -1.0, 1.0)
        truncated = False

        reward = 0.0


      # consistent action interpretation
        linear  = (action[0] + 1) / 2 * self.max_step_size # can only move forwards
        angular = action[1] * 0.25
      # update orientation
        self.robot_theta += angular
        self.robot_theta  = (self.robot_theta + np.pi) % (2 * np.pi) - np.pi
      # update position based on orientation
        dx = np.cos(self.robot_theta) * linear
        dy = np.sin(self.robot_theta) * linear
        self.robot_pos += np.array([dx, dy])
        self.robot_pos  = np.clip(self.robot_pos, [0, 0], self.world_size)

        # Reward structure 

        # Collision
        collision = False
        for obs_p in self.obstacles:
            if self._rect_vs_circle(obs_p, self.obstacle_size):
                reward -= 1.0
                collision = True
                away_dir = self.robot_pos - obs_p
                away_norm = np.linalg.norm(away_dir)
                if away_norm > 1e-6:
                    away_dir /= away_norm
                half_diag = np.sqrt((self.robot_length/2)**2 + (self.robot_width/2)**2)
                self.robot_pos = obs_p + away_dir * (self.obstacle_size + half_diag + 0.02)
                self.robot_pos = np.clip(self.robot_pos, [0, 0], self.world_size)
                break

        if self._rect_vs_wall(self.robot_pos, self.robot_theta):
          reward -= 1.0
          collision = True
          self.robot_pos = prev_pos.copy()
          self.robot_theta = prev_theta

        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)

        # progress: normalised displacement toward goal; in [-1, 1] per step
        progress     = (self.prev_dist - dist_to_goal)/self.max_step_size  # scaled down from 10
        reward       += progress* 2 - 0.005
        # dense proximity bonus within 2 m to guide agent into the goal region
        if dist_to_goal < 2.0:
            reward += 0.05 * (2.0 - dist_to_goal)

        # large terminal reward dominates shaping noise on success
        terminated = bool(dist_to_goal < self.done_threshold)
        if terminated:
            reward += 100.0

        self.prev_dist = np.linalg.norm(self.robot_pos - self.goal_pos)
        return self._get_obs(), reward, terminated, truncated, {"collision": collision, "success": terminated}

    def _get_obs(self):
      # prepare obstacles for LiDAR
        obstacles_lidar = [[p[0], p[1], self.obstacle_size]
                           for p in self.obstacles]
      # simulate LiDAR
        lidar = simulate_lidar_torch(
            self.robot_pos, self.robot_theta, obstacles_lidar, self.walls,
            num_beams=self.num_beams, max_range=self.MAX_RANGE)
      # add realistic noise
        noise  = np.random.normal(0, 0.01 + 0.01 * lidar)
        lidar  = np.clip(lidar + noise, 0, self.MAX_RANGE)
    # goal information in robot's frame
        goal_vec   = self.goal_pos - self.robot_pos
        goal_dist  = np.linalg.norm(goal_vec)
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0]) - self.robot_theta
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi # normalized to [-pi, pi]


    # combine observations
        obs = np.concatenate([
            lidar / self.MAX_RANGE,
            [goal_dist / self.max_dist, goal_angle / np.pi,
            np.sin(self.robot_theta), np.cos(self.robot_theta)]

        ])
        return obs.astype(np.float32)

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6.5))
            plt.ion()
        self.ax.clear()
        # walls
        for (x0, x1, y0, y1) in self.walls:
            self.ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0,
                              facecolor="#4A4845", zorder=3))
        # obstacles
        for obs in self.obstacles:
            self.ax.add_patch(plt.Circle(obs, self.obstacle_size,
                              facecolor="#8D6E63", zorder=4))
        # robot rectangle
        corners = self._robot_corners()
        self.ax.add_patch(plt.Polygon(corners, closed=True,
                          facecolor="#1565C0", edgecolor="#0D47A1", zorder=5))
        # heading arrow
        self.ax.annotate("", xy=(self.robot_pos[0] + 0.5*np.cos(self.robot_theta),
                                  self.robot_pos[1] + 0.5*np.sin(self.robot_theta)),
                        xytext=self.robot_pos,
                        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
                        zorder=6)
        # goal
        self.ax.plot(*self.goal_pos, "*", color="#E53935", markersize=14, zorder=6)
        self.ax.set_xlim(0, self.world_size[0])
        self.ax.set_ylim(0, self.world_size[1])
        self.ax.set_aspect("equal")
        self.fig.canvas.draw()
        plt.pause(0.01)

def run_episode(agent, env, max_steps=2000, render=False):
    """
    Roll out one episode for any agent implementing either
    select_action(state) (RandomWalker) or predict(state) (SB3).

    Returns: total_reward, steps, collision (bool), collision_count,
            success (bool), timeout (bool).
    """
    if hasattr(agent, "reset"):
        agent.reset()

    obs = env.reset()
    state = obs[0] if isinstance(obs, tuple) else obs
    if isinstance(state, np.ndarray) and state.ndim > 1:
        state = state[0]

    total_reward = 0.0
    steps        = 0
    collision_count = 0
    collision    = False
    success      = False
    timeout     = False

    for _ in range(max_steps):
        if hasattr(agent, "select_action"):
            action = agent.select_action(state)
        else:
            action, _ = agent.predict(state, deterministic=True)

        result = env.step(action)
        if len(result) == 4:
            next_obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            next_obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        if isinstance(next_obs, np.ndarray) and next_obs.ndim > 1:
            state  = next_obs[0]
            reward = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            done   = done[0]   if isinstance(done,   (list, np.ndarray)) else done
            info_  = info[0]   if isinstance(info,   list)               else info
        else:
            state  = next_obs
            info_  = info

        total_reward += float(reward)
        steps        += 1

        if isinstance(info_, dict):
            if info_.get("collision", False):
                collision = True
                collision_count += 1

            if info_.get("success", False):
                success = True

        if truncated:
          timeout = True

        if done:
          break

        if render:
          env.render()
          time.sleep(0.05)

    return total_reward, steps, collision, collision_count, success, timeout

# Random Search 

SEARCH_SPACE = {
    "DDPG": {
        "learning_rate":   [1e-4, 1e-3, 3e-4, 3e-3],
        "batch_size":      [128, 256, 512],
        "tau":             [0.001, 0.005, 0.01],
        "learning_starts": [1000, 5000, 10000],
        "net_arch":        [[64,64], [128,128], [256,256], [256,256,256]]
    },
    "TD3": {
        "learning_rate":   [1e-4, 1e-3, 3e-4, 3e-3],
        "batch_size":      [128, 256, 512],
        "tau":             [0.001, 0.005, 0.01],
        "learning_starts": [1000, 5000, 10000],
        "net_arch":        [[64,64], [128,128], [256,256], [256,256,256]]
    },
    "SAC": {
        "learning_rate":   [1e-4, 1e-3, 3e-4, 3e-3],
        "batch_size":      [128, 256, 512],
        "tau":             [0.001, 0.005, 0.01],
        "learning_starts": [500, 1000, 2000],
        "net_arch":        [[64,64], [128,128], [256,256], [256,256,256]],
        "ent_coef":        ["auto", 0.1, 0.01]
    },
    "PPO": {
        "learning_rate":   [1e-4, 1e-3, 3e-4, 3e-3],
        "n_steps":         [1024, 2048, 4096],
        "n_epochs":        [5, 10, 20],
        "clip_range":      [0.1, 0.2, 0.3],
        "net_arch":        [[64,64], [128,128], [256,256], [256,256,256]]
    },
    "TRPO": {
        "learning_rate":   [1e-4, 1e-3, 3e-4, 3e-3],
        "n_steps":         [1024, 2048, 4096],
        "net_arch":        [[64,64], [128,128], [256,256], [256,256,256]],
    },
}

def sample_config(agent_name):
    """
    Uniformly sample one value per hyperparameter from SEARCH_SPACE.
    """
    config = {}
    for k, v in SEARCH_SPACE[agent_name].items():
        config[k] = random.choice(v) 
    return config

def build_agent(agent_name, config, env):
  """
  Instantiate an SB3 agent from a sampled config dict.
  All agents share gamma=0.99 and buffer_size=200_000 (off-policy only).
  """
  net = {"net_arch": config["net_arch"]}
  n_act = env.action_space.shape[-1]

  if agent_name == "DDPG":
        return sb3.DDPG(
            "MlpPolicy", env, device=device,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            tau=config["tau"],
            learning_starts=config["learning_starts"],
            buffer_size=200_000,
            gamma=0.99,
            action_noise=NormalActionNoise(np.zeros(n_act), 0.1*np.ones(n_act)),
            policy_kwargs=net,
            verbose=0)

  elif agent_name == "TD3":
        return sb3.TD3(
            "MlpPolicy", env, device=device,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            tau=config["tau"],
            learning_starts=config["learning_starts"],
            buffer_size=200_000,
            gamma=0.99,
            action_noise=NormalActionNoise(np.zeros(n_act), 0.1*np.ones(n_act)),
            policy_kwargs=net,
            verbose=0)

  elif agent_name == "SAC":
        return sb3.SAC(
            "MlpPolicy", env,device=device,
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            tau=config["tau"],
            learning_starts=config["learning_starts"],
            buffer_size=200_000,
            gamma=0.99,
            policy_kwargs=net,
            verbose=0,
            ent_coef = config["ent_coef"])

  elif agent_name == "PPO":
        return sb3.PPO(
            "MlpPolicy", env, device=device,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            n_epochs=config["n_epochs"],
            clip_range=config["clip_range"],
            gamma=0.99,
            policy_kwargs=net,
            verbose=0)

  elif agent_name == "TRPO":
        return TRPO(
            "MlpPolicy",
            env,
            device=device,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            gamma=0.99,
            policy_kwargs=net,
            verbose=0)

def make_env(seed=None):
    """
    Wrap the 2D environment (Continuous2DNavEnv) in Monitor + TimeLimit. 
    max_step_size=0.044 matches the physical Gazebo robot speed.
    """
    env = Monitor(
        TimeLimit(
            Continuous2DNavEnv(
                world_size= (16.2, 13.0), max_step_size=0.044, random_map=True, 
                num_obstacles=10),
            max_episode_steps=MAX_EP_STEPS))
    if seed is not None:
        env.reset(seed=seed)
    return env


# Use GPU if available, but falls back automatically to CPU if not. 
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_random_search():
    """
    For each algorithm, sample N_TRIALS (10) configs, train for TRAIN_STEPS (200,000),
    evaluate on EVAL_EPISODES (10) episodes, and return the best config per agent.
    Evaluation uses a fixed seed (999) for reproducibility across trials.
    Failed trials (e.g. NaN divergence) are caught and skipped. 
    """
    best_configs = {}

    for agent_name in SEARCH_SPACE.keys():
        print(f"\n{'='*50}\nRandom search for {agent_name}\n{'='*50}")
        best_score  = -np.inf
        best_config = None

        for trial in range(N_TRIALS):
            config = sample_config(agent_name)
            print(f"  Trial {trial+1}/{N_TRIALS} | config: {config}")
            try:
              # create a fresh training env
              train_env = make_env()

              # build the agent with sampled config
              agent = build_agent(agent_name, config, train_env)

              # train for TRAIN_STEPS
              agent.learn(total_timesteps = TRAIN_STEPS)

              # evaluate: run EVAL_EPISODES episodes, compute success rate
              successes = []
              eval_env  = make_env(seed=999)
              for _ in range(EVAL_EPISODES):
                  _, _, _, _, success, _ = run_episode(agent, eval_env, max_steps = MAX_EP_STEPS)
                  successes.append(float(success))
              eval_env.close()
              train_env.close()

              score = float(np.mean(successes))
              print(f"    success_rate={score:.2f}")

              # keep best config
              if score > best_score:
                  best_score  = score
                  best_config = config
            except Exception as e:
              print(f"    Trial failed: {e}")
              continue

        best_configs[agent_name] = best_config
        print(f"\n  Best config for {agent_name}: {best_config} (score={best_score:.2f})")

    return best_configs


if __name__ == "__main__":
  best_configs = run_random_search()
  print("\n" + "="*50)
  print("FINAL BEST CONFIGURATIONS:")
  print("="*50)
  for agent, cfg in best_configs.items():
      print(f"{agent}: {cfg}")
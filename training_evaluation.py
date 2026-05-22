"""
This script runs the full training and evaluation of the DRL agents in a custom 2D navigation environment. 
It trains five DRL algorithms (DDPG, TD3, SAC, PPO, and TRPO), and evaluates them, and a baseline (Random walker), on the environment. 
The custom 2D environment inherits from gymnasium, and simulates a cluttered USAR setting. 
It is a 2D replicate of a 3D Gazebo model, in which 10 obstacles are found and the agent must navigate to the goal (i.e. the victim's position) by avoiding these obstacles and walls. 
The Gazebo model is the "collapsed house" model by OpenRobotics (OpenRobotics, 2023). 
The reference navigation agent is the Unitree Go2 (Gabr, 2024). 





Environment summary:
    World size:       16.2 × 13.0 m (matched to Gazebo SDF)
    Observation:      68-dim — 64-beam LiDAR + goal distance, goal bearing,
                      sin/cos of heading
    Action:           2-dim continuous — forward speed, angular velocity
    Episode horizon:  1500 steps
    Success criterion: robot centre within 0.3 m of goal

    
Training is parallelised across agents using Python multiprocessing,
with each agent writing results to a separate .pkl file that are merged
for final plotting.
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
import pickle
import os 
import multiprocessing as mp 
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
import stable_baselines3 as sb3
from sb3_contrib import TRPO

EVAL_FREQ = 25000 # check points every 25,000 steps 
EVAL_EPISODES = 5 # number of evaluation episodes
MAX_EP_STEPS = 1500 # maximum step count per episode 
FULL_STEPS = 750000 # maximum step budget for the whole training 
FULL_TRAIN = True # kept for reference 




def simulate_lidar_torch(robot_pos, robot_theta, obstacles_lidar, walls,
                         num_beams=64, fov=2*np.pi, max_range=13.0):
  
  """
  Simulates LiDAR sensor on Unitree Go2. 
  It makes use of ray-casting LiDAR, using the slab method and analytic ray-circle intersection for round obstacles. 

  Its inputs include: 
  - robot_pos: (2, ) array 
  - robot_theta: robot heading 
  - obstacle_lidar: list of [cx, cy, radius]
  - walls: list of (x_min, x_max, y_min, y_max) AABBs
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
                 obstacle_size=0.35, max_step_size=0.2, random_map=True,): # Except for max_step_size, the defaults already correspond to the correct environment configurations 
        super().__init__()
        self.world_size   = np.array(world_size, dtype=float)
        self.num_obstacles = num_obstacles
        self.obstacle_size = obstacle_size
        self.render_mode   = render_mode
        self.fig, self.ax  = None, None
        self.max_step_size = max_step_size # Later in the main guard overridden as 0.088. 
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

    def reset(self, seed=None, options=None): # what makes environment outline match to the Gazebo collapsed house model outline 
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
        progress     = (self.prev_dist - dist_to_goal)/self.max_step_size  
        reward       += progress* 2 - 0.005 # previously 10 but scaled down to 2 because it was too high 
        
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

class BaseAlgorithm:
    """
    Abstract base class for all navigation agents.
    """
    def select_action(self, state): 
        raise NotImplementedError
    def reset(self): 
        pass


class RandomWalker(BaseAlgorithm): 
    """
    Baseline agent that samples uniformly random actions each step.
    """
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def select_action(self, state): # function of random walker, chooses a random step 
        return self.rng.uniform(-1.0, 1.0, size=self.action_space.shape)

def compute_metrics(rewards, steps,collisions=None, collision_counts=None, successes=None,timeouts=None): # all the metrics, in the thesis displayed in Table 6 in Appendix. 
    """
    Returns a dictionary of 
    mean / standard deviation / minimum / maximum reward, 
    mean / standard deviation steps, 
    success rate, 
    collision rate, 
    mean collision count, 
    timeout rate. 
    The results are reported in Table 6 (Appendix 6).
    """
    
    rewards = np.array(rewards, dtype=float)
    steps   = np.array(steps,   dtype=float)
    m = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward":  float(np.std(rewards)),
        "mean_steps":  float(np.mean(steps)),
        "std_steps":   float(np.std(steps)),
        "max_reward":  float(np.max(rewards)),
        "min_reward":  float(np.min(rewards)),
    }
    if successes is not None:
        m["success_rate"] = float(np.mean(successes))
    if collisions is not None:
        m["collision_rate"] = float(np.mean(collisions))
    if collision_counts is not None:
        m["mean_collision_count"] = float(np.mean(collision_counts))

    if timeouts is not None:
        m["timeout_rate"] = float(np.mean(timeouts))
    return m


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



def run_experiment(agents, train_env, eval_envs,
                   episodes_per_eval_env=5,
                   train_steps=200000,
                   eval_freq=50000):

    all_rewards     = {}
    all_steps       = {}
    all_collisions  = {}
    all_collision_counts = {}
    all_successes   = {}
    all_timeouts    = {}
    training_curves = {}


    rl_agents = {"DDPG", "SAC", "PPO", "TD3", "TRPO"}

    for name, agent in agents.items(): # training
        patience = 20 # early stopping  (halts if smoothed return and does not improve by > 0.5 over 20 consecutive checkpoints)
        min_delta = 0.5
        best_reward = -np.inf
        patience_count = 0
        total_steps = 0
        max_steps = train_steps
        recent_rewards = []
        print("\n{'='*60}\nRunning agent: {}\n{'='*60}".format(name))

        if name in rl_agents:
            training_rewards = []
            training_steps_list = []

            while True:
                agent.learn(total_timesteps=eval_freq,
                            reset_num_timesteps=False)
                total_steps += eval_freq
                eval_r = []
                for ee in eval_envs:
                    for _ in range(episodes_per_eval_env):
                        r, _, _, _, _, _ = run_episode(agent, ee, max_steps=MAX_EP_STEPS)
                        eval_r.append(r)
                avg = float(np.mean(eval_r))

                # inside the while loop, after computing avg:
                recent_rewards.append(avg)
                smoothed = float(np.mean(recent_rewards[-3:]))

                if smoothed > best_reward + min_delta:
                    best_reward = smoothed
                    patience_count = 0
                else:
                    patience_count += 1
                training_rewards.append(avg)
                training_steps_list.append(total_steps)
                print("  [{}] step {}  avg_reward = {:.2f} patience = {} / {}".format(name, total_steps, avg, patience_count, patience))
                if patience_count >= patience or total_steps >= max_steps:
                  print("{} converged at step {}".format(name, total_steps))
                  agent.save("{}_model".format(name))
                  break

            training_curves[name] = {
                "steps":   training_steps_list,
                "rewards": training_rewards,
            }

        # Final evaluation
        print("\nFinal evaluation: {}".format(name))
        fr, fs, fc, fcc, fsucc, ftimeout = [], [], [], [], [], []
        for ee in eval_envs:
            for _ in range(episodes_per_eval_env):
                if hasattr(agent, "reset"):
                    agent.reset()
                r, s, col, col_count, suc, tout = run_episode(agent, ee, max_steps=MAX_EP_STEPS)
                fr.append(r)
                fs.append(s)

                fc.append(float(col))
                fcc.append(col_count)

                fsucc.append(float(suc))
                ftimeout.append(float(tout))

        all_rewards[name]    = fr
        all_steps[name]      = fs
        all_collisions[name] = fc
        all_successes[name]  = fsucc
        all_collision_counts[name] = fcc
        all_timeouts[name] = ftimeout

        m = compute_metrics(
                fr,
                fs,
                collisions=fc,
                collision_counts=fcc,
                successes=fsucc,
                timeouts=ftimeout
            )
        print("\n{} metrics:".format(name))
        for k, v in m.items():
            print(f"  {k}: {v:.3f}")

    if hasattr(train_env, "close"):
        train_env.close()
    for ee in eval_envs:
        if hasattr(ee, "close"):
            ee.close()

    plot_results(training_curves, all_rewards, all_steps,
                 all_collisions, all_successes)

    return all_rewards, all_steps, training_curves


# NOTE: An important note is that a new file was written to re-create the plots, using different colours (See graphs_edit.ipynb). 
# The new file only differs in the colours used for the agents, and all other aspects of the function remain the same. 


PALETTE = {
    "Random":    "#9E9E9E",
    "LocalAStar":"#4CAF50",
    "DDPG":      "#2196F3",
    "TD3":       "#03A9F4",
    "SAC":       "#FF5722",
    "PPO":       "#FF9800",
    "TRPO":      "#9C27B0",
}


def _agent_color(name):
    return PALETTE.get(name, "#607D8B")


def plot_results(training_curves, all_rewards, all_steps,
                 all_collisions, all_successes):
    """
    Generate and save five diagnostic figures:
    fig1 –  learning curves (return vs environment steps)
    fig2 –  final return distributions (box + strip plot)
    fig3 –  steps-per-episode distributions (violin plot)
    fig4 –  success and collision rates (grouped bar chart)
    fig5 –  early-training comparison of off-policy methods (first 2×10⁴ steps);
            omitted from the final thesis as it was not informative after
            substantial environment changes.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        150,
    })

    rl_names     = [n for n in all_rewards if n in training_curves]
    agent_names  = list(all_rewards.keys())

    #
    if rl_names:
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        for name in rl_names:
            curve = training_curves[name]
            xs    = np.array(curve["steps"])
            ys    = np.array(curve["rewards"])
            ax1.plot(xs, ys, label=name,
                     color=_agent_color(name), linewidth=2, marker="o",
                     markersize=4)
            # shaded ±σ would need multiple seeds; mark single run with dashed
            # confidence band placeholder 
            ax1.fill_between(xs, ys * 0.95, ys * 1.05,
                             color=_agent_color(name), alpha=0.15)

        ax1.set_xlabel("Environment steps")
        ax1.set_ylabel("Mean evaluation return")
        ax1.set_title("Training learning curves")
        ax1.legend(frameon=False)
        ax1.ticklabel_format(axis="x", style="sci", scilimits=(4, 4))
        fig1.tight_layout()
        plt.savefig("fig1_learning_curves.pdf", bbox_inches="tight")
        pass

    
    reward_df = pd.DataFrame({
        n: pd.Series(v) for n, v in all_rewards.items()
    }).melt(var_name="Agent", value_name="Return")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=reward_df, x="Agent", y="Return",
                palette={n: _agent_color(n) for n in agent_names},
                order=agent_names,
                width=0.5, linewidth=1.2,
                flierprops=dict(marker="x", markersize=4, linewidth=0.8),
                ax=ax2)
    sns.stripplot(data=reward_df, x="Agent", y="Return",
                  palette={n: _agent_color(n) for n in agent_names},
                  order=agent_names,
                  dodge=False, jitter=True, size=3, alpha=0.5, ax=ax2)
    ax2.set_title("Final evaluation return distribution")
    ax2.set_xlabel("")
    ax2.set_ylabel("Cumulative return")
    ax2.tick_params(axis="x", rotation=15)
    fig2.tight_layout()
    plt.savefig("fig2_final_return_boxplot.pdf", bbox_inches="tight")
    pass

    #
    steps_df = pd.DataFrame({
        n: pd.Series(v) for n, v in all_steps.items()
    }).melt(var_name="Agent", value_name="Steps")

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.violinplot(data=steps_df, x="Agent", y="Steps",
                   palette={n: _agent_color(n) for n in agent_names},
                   order=agent_names,
                   inner="quartile", linewidth=1.0, cut=0, ax=ax3)
    ax3.set_title("Steps per episode")
    ax3.set_xlabel("")
    ax3.set_ylabel("Steps to termination")
    ax3.tick_params(axis="x", rotation=15)
    fig3.tight_layout()
    plt.savefig("fig3_steps_violin.pdf", bbox_inches="tight")
    pass

    #
    rate_data = {
        "Agent":         agent_names,
        "Success rate":  [float(np.mean(all_successes[n]))   for n in agent_names],
        "Collision rate":[float(np.mean(all_collisions[n]))  for n in agent_names],
    }
    rate_df = pd.DataFrame(rate_data).melt(
        id_vars="Agent", var_name="Metric", value_name="Rate")

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=rate_df, x="Agent", y="Rate", hue="Metric",
                order=agent_names,
                palette={"Success rate": "#43A047", "Collision rate": "#E53935"},
                linewidth=0.8, edgecolor="white", ax=ax4)
    ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("")
    ax4.set_ylabel("Rate")
    ax4.set_title("Success and collision rates")
    ax4.legend(frameon=False, loc="upper right")
    ax4.tick_params(axis="x", rotation=15)
    fig4.tight_layout()
    plt.savefig("fig4_success_collision_rates.pdf", bbox_inches="tight")
    pass


    MAX_STEPS   = 20000
    detail_agents = [n for n in ["DDPG", "TD3", "SAC"] if n in training_curves]

    if detail_agents:
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        for name in detail_agents:
            xs = np.array(training_curves[name]["steps"])
            ys = np.array(training_curves[name]["rewards"])
            mask = xs <= MAX_STEPS
            xs, ys = xs[mask], ys[mask]
            color = _agent_color(name)
            ax5.plot(xs, ys, label=name, color=color,
                     linewidth=2, marker="o", markersize=4)
            ax5.fill_between(xs, ys * 0.95, ys * 1.05,
                             color=color, alpha=0.15)

        ax5.set_xlabel("Environment steps")
        ax5.set_ylabel("Mean evaluation return")
        ax5.set_title("DDPG vs TD3 vs SAC — first 2×10⁴ steps")
        ax5.set_xlim(0, MAX_STEPS)
        ax5.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x > 0 else "0"))
        ax5.legend(frameon=False)
        fig5.tight_layout()
        plt.savefig("fig5_ddpg_td3_sac_2e4.pdf", bbox_inches="tight")
        pass

# for parallel training purposes, the following function was added
def train_single_agent(agent_name, result_dir, train_steps, eval_freq, eval_episodes): 
    """
    Trains and evaluates a single agent in an isolated process. 
    Results are stored in a pkl file for later merging. 
    This function allows parallel multi-agent runs, reducing the computational time significantly compared to sequential running. 
    """
    print("{} starting".format(agent_name))
    world_size=(16.2, 13.0)

    def make_env(seed=None):
        """
        Wrap the 2D environment (Continuous2DNavEnv) in Monitor + TimeLimit. 
        max_step_size=0.088 is double the Gazebo robot speed (0.044 m/step). 
        """
        env = Monitor(
            TimeLimit(
                Continuous2DNavEnv(
                    world_size=world_size, max_step_size=0.088, random_map=True, # doubled speed compared to Gazebo (in Gazebo: max_step_size = 0.044)
                    num_obstacles=10),
                max_episode_steps=MAX_EP_STEPS))
        if seed is not None:
            env.reset(seed=seed)
        return env
    
    train_env = make_env() 
    eval_envs = [make_env(seed = s) for s in range(100, 110)]
    n_act = train_env.action_space.shape[-1]

    if agent_name == "Random": 
        agent = RandomWalker(train_env.action_space) 
    elif agent_name == "DDPG": 
        agent = sb3.DDPG(
            "MlpPolicy", train_env,
            learning_rate=1e-3, batch_size=512,
            tau=0.001, learning_starts=10000,
            buffer_size=200000, gamma=0.99,
            action_noise=NormalActionNoise(np.zeros(n_act),
                                           0.1 * np.ones(n_act)),
            policy_kwargs={"net_arch": [64, 64]},
            verbose=0, device = "cpu")
    elif agent_name == "TD3": 
        agent =  sb3.TD3(
            "MlpPolicy", train_env,
            learning_rate=1e-3, batch_size=128,
            tau=0.001, learning_starts=10000,
            buffer_size=200000, gamma=0.99,
            action_noise=NormalActionNoise(np.zeros(n_act), 0.1 * np.ones(n_act)),
            policy_kwargs={"net_arch": [256, 256]},
            verbose=0,device = "cpu")
    elif agent_name == "SAC": 
        agent = sb3.SAC(
            "MlpPolicy", train_env,
            learning_rate=3e-4, batch_size=256,
            ent_coef="auto",learning_starts=2000,
            tau = 0.005,
            gamma = 0.99,
            buffer_size = 200000,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=0,device = "cpu" )
    elif agent_name == "PPO": 
        agent = sb3.PPO(
            "MlpPolicy", train_env,
            learning_rate=1e-4, n_steps=2048,
            n_epochs=5,
            clip_range=0.1,
            policy_kwargs=dict(net_arch=[64, 64],
                               activation_fn=torch.nn.Tanh),
            verbose=0,device = "cpu")
    elif agent_name == "TRPO": 
        agent = TRPO(
            "MlpPolicy", train_env,
            learning_rate=0.003,
            n_steps= 1024,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=0, device = "cpu")
    rl_agents = {"DDPG", "TD3", "SAC", "PPO", "TRPO"}
    patience = 20 
    best_reward = -np.inf 
    patience_count = 0 
    total_steps = 0 
    recent_rewards = []
    training_curves = {}

    fr, fs, fc, fcc, fsucc, ftimeout = [], [], [], [], [], []

    if agent_name in rl_agents: 
        training_rewards = []
        training_steps_list = [] 

        while True: 
            agent.learn(total_timesteps= eval_freq, reset_num_timesteps= False)
            total_steps += eval_freq 
            eval_r = [] 

            for e in eval_envs: 
                for _ in range(eval_episodes): 
                    r, _, _, _, _, _ = run_episode(agent, e, max_steps = MAX_EP_STEPS)
                    eval_r.append(r)
            avg = float(np.mean(eval_r))
            recent_rewards.append(avg)
            smoothed = float(np.mean(recent_rewards[-3:]))
            if smoothed > best_reward + 0.5: 
                best_reward = smoothed 
                patience_count = 0 
            else: 
                patience_count += 1 
            training_rewards.append(avg)
            training_steps_list.append(total_steps)
            print (" [{}] step {} avg_reward = {:.2f} patience = {} / {}".format(agent_name, total_steps, avg, patience_count, patience))

            if patience_count >= patience: 
                print ("{} converged at step {}".format(agent_name, total_steps))
                agent.save("{} model".format(agent_name))
                break 
            elif total_steps >= train_steps: 
                print("Max steps per agent achieved : {}".format(total_steps))
                agent.save("{} model".format(agent_name)) 
                break 
        training_curves[agent_name] = { 
            "steps" : training_steps_list, 
            "rewards" : training_rewards, 
        }
    print("Running final evaluation on {}".format(agent_name))
    for e in eval_envs: 
        for _ in range(eval_episodes): 
            r, s, col, col_count, suc, tout = run_episode(agent, e, max_steps = MAX_EP_STEPS) 
            fr.append(r)
            fs.append(s)
            fc.append(float(col))
            fcc.append(col_count)
            fsucc.append(float(suc))
            ftimeout.append(float(tout))
    train_env.close() 

    for e in eval_envs: 
        e.close() 

    results = {
        "training_curves":      training_curves,
        "all_rewards":          {agent_name: fr},
        "all_steps":            {agent_name: fs},
        "all_collisions":       {agent_name: fc},
        "all_collision_counts": {agent_name: fcc},
        "all_successes":        {agent_name: fsucc},
        "all_timeouts":         {agent_name: ftimeout},
    }
        
    out_path = ("{}/results_{}.pkl".format(result_dir, agent_name))
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print("[{}] Saved results to {}".format(agent_name, out_path))


# full training
if __name__ == "__main__":
    mp.freeze_support()
    result_dir = "results_corrected_hp"
    os.makedirs(result_dir, exist_ok = True)

    agent_names = ["Random", "DDPG", "TD3", "SAC", "PPO", "TRPO"]
    train_steps = 750000
    eval_freq = 25000
    eval_episodes = 5 

    # Each agent runs in its own process (parallel training) 
    processes = [] 
    for name in agent_names: 
        p = mp.Process(
            target = train_single_agent, 
            args = (name, result_dir, train_steps, eval_freq, eval_episodes)
        )
        p.start() 
        processes.append(p)


    for p in processes: 
        p.join() 

    print ("All agent done. Merging results now")

    agent_order = ["Random", "DDPG", "TD3", "SAC", "PPO", "TRPO"]
    merged_training_curves = {}
    merged_rewards = {}
    merged_steps = {}
    merged_collisions = {}
    merged_col_counts = {}
    merged_successes = {}
    merged_timeouts = {}

    for name in agent_order: 
        path = "{}/results_{}.pkl".format(result_dir, name)

        if not os.path.exists(path): 
            print("missing results for {}, skipping".format(name))
            continue 
        with open(path, "rb") as f: 
            res = pickle.load(f)
        merged_training_curves.update(res["training_curves"])
        merged_rewards.update(res["all_rewards"])
        merged_steps.update(res["all_steps"])
        merged_collisions.update(res["all_collisions"])
        merged_col_counts.update(res["all_collision_counts"])
        merged_successes.update(res["all_successes"])
        merged_timeouts.update(res["all_timeouts"])
    plot_results(
        merged_training_curves,
        merged_rewards,
        merged_steps,
        merged_collisions,
        merged_successes,
    )
    print("Plots saved.")
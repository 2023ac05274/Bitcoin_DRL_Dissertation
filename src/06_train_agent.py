import pandas as pd
import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from trading_env import BitcoinTradingEnv

# --- Configuration ---
SYMBOL = "BTCUSD"
INPUT_FILE = f"train.csv"
INPUT_PATH = os.path.join("data", "processed", INPUT_FILE)
MODELS_DIR = "models"
LOGS_DIR = "logs"
LOG_FILE = os.path.join("logs", "agent_training.log")

# Hyperparameters
TIMESTEPS = 200000  # Short quick training
LEARNING_RATE = 0.0002  # Learning rate for PPO
N_STEPS = 2048       # Steps per update
BATCH_SIZE = 64     # Minibatch size
N_EPOCHS = 10     # Number of epochs per update
GAMMA = 0.99     # Discount factor 
GAE_LAMBDA = 0.95   # GAE lambda
CLIP_RANGE = 0.2    # Clipping range
ENT_COEF = 0.04     # Entropy coefficient

#log for hyperparameters
logging.info("Hyperparameters:")
logging.info(f"  TIMESTEPS: {TIMESTEPS}")
logging.info(f"  LEARNING_RATE: {LEARNING_RATE}")
logging.info(f"  N_STEPS: {N_STEPS}")
logging.info(f"  BATCH_SIZE: {BATCH_SIZE}")
logging.info(f"  N_EPOCHS: {N_EPOCHS}")
logging.info(f"  GAMMA: {GAMMA}")
logging.info(f"  GAE_LAMBDA: {GAE_LAMBDA}")
logging.info(f"  CLIP_RANGE: {CLIP_RANGE}")
logging.info(f"  ENT_COEF: {ENT_COEF}")

# --- Setup Logging  ---
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'), # Append mode
        logging.StreamHandler()
    ]
)

def plot_training_results(log_dir):
    """Plots the Mean Reward curve from evaluations."""
    eval_file = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(eval_file):
        logging.warning("No evaluations.npz found to plot.")
        return

    data = np.load(eval_file)
    timesteps = data['timesteps']
    results = data['results']
    mean_rewards = np.mean(results, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label="Mean Reward", color='green')
    plt.title(f"Agent Training Progress ({SYMBOL})")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path = os.path.join(MODELS_DIR, f"agent_learning_curve_{int(time.time())}.png")
    plt.savefig(save_path)
    logging.info(f"Learning curve saved to: {save_path}")
    plt.close()

def train():
    logging.info("--- Starting Quick Training Session ---")

    #  Setup Directories
    os.makedirs(MODELS_DIR, exist_ok=True)

    #  Load Data
    if not os.path.exists(INPUT_PATH):
        logging.error("Error: Feature file not found.")
        return
    
    df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
    logging.info(f"Loaded {len(df)} rows of data.")

    #  Initialize Environment (With Monitor for stats)
    #  wrap the Env in 'Monitor' to track episode rewards easily
    env = DummyVecEnv([lambda: Monitor(BitcoinTradingEnv(df, execution_mode='taker'))])

    #  Setup Callback for Evaluation/Plotting
    # Evaluating on the training env itself just for the sake of the curve in this simple script
    eval_callback = EvalCallback(
        env,
        best_model_save_path=MODELS_DIR,
        log_path=LOGS_DIR,
        eval_freq=20000,
        deterministic=False,
        render=False
    )

    #  Initialize PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOGS_DIR,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,

        # Policy Network Architecture
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]
        )
    )

    #  Start Training
    logging.info(f"Training for {TIMESTEPS} timesteps...")
    start_time = time.time()
    
    model.learn(total_timesteps=TIMESTEPS, callback=eval_callback)
    
    end_time = time.time()
    logging.info(f"--- Training Complete in {(end_time - start_time)/60:.2f} minutes ---")

    #  Save
    save_path = os.path.join(MODELS_DIR, f"PPO_{SYMBOL}_QUICK_{int(time.time())}")
    model.save(save_path)
    logging.info(f"Model saved to: {save_path}.zip")

    #  Plot
    plot_training_results(LOGS_DIR)

if __name__ == "__main__":
    train()
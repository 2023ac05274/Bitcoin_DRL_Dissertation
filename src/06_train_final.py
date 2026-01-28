import pandas as pd
import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback
from trading_env import BitcoinTradingEnv

# --- CONFIGURATION ---
SYMBOL = "BTCUSD"
MODELS_DIR = "models"
LOGS_DIR = "logs"
LOG_FILE = os.path.join("logs", "training_final.log")

# Data Paths 
TRAIN_FILE = os.path.join("data", "processed", "train.csv")
VAL_FILE = os.path.join("data", "processed", "val.csv")

# Transfer Learning
PRETRAINED_MODEL_NAME = "PPO_BTCUSD_QUICK_1768301861"  # Load from quick training 
PRETRAINED_PATH = os.path.join(MODELS_DIR, f"{PRETRAINED_MODEL_NAME}.zip") if PRETRAINED_MODEL_NAME else None

# Hyperparameters
TIMESTEPS = 1000000   # Long training run
EVAL_FREQ = 20000     # Evaluate every 20k steps
PATIENCE = 10         # Stop if no improvement after 10 evals (10 * 20k = 200k steps)

# --- Setup Logging (Append Mode) ---
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'), # Append
        logging.StreamHandler()
    ]
)

def plot_final_curve(log_folder):
    """Plots the official learning curve from the validation set."""
    eval_path = os.path.join(log_folder, "evaluations.npz")
    if not os.path.exists(eval_path):
        return

    data = np.load(eval_path)
    timesteps = data['timesteps']
    mean_rewards = np.mean(data['results'], axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label='Validation Reward', color='blue', linewidth=2)
    plt.title(f"Final PPO Training Performance: {SYMBOL}")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path = os.path.join(MODELS_DIR, f"final_learning_curve_{int(time.time())}.png")
    plt.savefig(save_path)
    logging.info(f"Final Learning Curve saved to: {save_path}")
    plt.close()

def train_final():
    logging.info(f"--- Starting Final Training Pipeline for {SYMBOL} ---")
    
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. LOAD DATA
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(VAL_FILE):
        logging.error("CRITICAL ERROR: Train/Val files not found. Run 'src/09_split_data.py' first.")
        return

    df_train = pd.read_csv(TRAIN_FILE, index_col=0, parse_dates=True)
    df_val = pd.read_csv(VAL_FILE, index_col=0, parse_dates=True)
    logging.info(f"Train Rows: {len(df_train)} | Val Rows: {len(df_val)}")
    
    # 2. CREATE ENVIRONMENTS
    train_env = DummyVecEnv([lambda: BitcoinTradingEnv(df_train, execution_mode='taker', leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04)])
    eval_env = DummyVecEnv([lambda: BitcoinTradingEnv(df_val, execution_mode='taker', leverage=5, stop_loss_pct=0.02, take_profit_pct=0.04)])

    # 3. IMPROVED CALLBACKS
    
    # A. Early Stopping Mechanism
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=PATIENCE, 
        min_evals=5, 
        verbose=1
    )
    
    # B. Main Evaluation Callback (triggers Early Stopping)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODELS_DIR,
        log_path=LOGS_DIR,
        eval_freq=EVAL_FREQ,
        callback_after_eval=stop_train_callback, # Link early stopping here
        deterministic=False,
        verbose=1
    )
    
    # C. Checkpoint Callback (Backup every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=os.path.join(MODELS_DIR, "checkpoints"),
        name_prefix="ppo_btc"
    )

    # Combine callbacks list
    callback_list = [eval_callback, checkpoint_callback]

    # 4. INITIALIZE MODEL
    if PRETRAINED_PATH and os.path.exists(PRETRAINED_PATH):
        logging.info(f"--- TRANSFER LEARNING: Loading {PRETRAINED_PATH} ---")
        model = PPO.load(PRETRAINED_PATH, env=train_env)
        reset_timesteps = False 
    else:
        logging.info("--- FRESH TRAINING ---")
        model = PPO(
            "MlpPolicy", 
            train_env, 
            verbose=1, 
            tensorboard_log=LOGS_DIR,
            learning_rate=0.0003,
            ent_coef=0.05, 
            gamma=0.99
        )
        reset_timesteps = True

    # 5. START TRAINING
    logging.info(f"Training for {TIMESTEPS} steps...")
    start_time = time.time()
    
    model.learn(total_timesteps=TIMESTEPS, callback=callback_list, reset_num_timesteps=reset_timesteps)
    
    end_time = time.time()
    duration = (end_time - start_time) / 60
    logging.info(f"--- Training Complete in {duration:.2f} minutes ---")

    # 6. SAVE & PLOT
    final_path = os.path.join(MODELS_DIR, f"PPO_{SYMBOL}_FINAL_{int(time.time())}")
    model.save(final_path)
    logging.info(f"Final Model saved to: {final_path}.zip")
    
    plot_final_curve(LOGS_DIR)

if __name__ == "__main__":
    train_final()
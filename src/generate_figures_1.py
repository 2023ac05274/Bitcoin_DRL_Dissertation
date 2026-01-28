import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set common style
plt.style.use('default')

# ==========================================
# FIGURE 1: MODULES IN TRADING SYSTEM
# ==========================================
def draw_figure_1():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, "AUTOMATED TRADING SYSTEM", fontsize=14, weight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightgray", ec="black"))

    # Define boxes
    modules = [
        ("Data Processing\nPipeline (DPP)", 1.5, 4, "lightblue"),
        ("Custom Trading\nEnvironment (Gym)", 4.5, 4, "lightgreen"),
        ("DRL Agent\n(PPO Model)", 7.5, 4, "salmon"),
        ("Live Execution\nEngine", 10.5, 4, "wheat")
    ]

    # Draw Boxes
    for name, x, y, color in modules:
        rect = patches.FancyBboxPatch((x-1.2, y-1), 2.4, 2, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x, y, name, fontsize=11, weight='bold', ha='center', va='center')

    # Draw connecting arrows (Implicit flow)
    ax.arrow(2.8, 4, 0.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
    ax.arrow(5.8, 4, 0.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
    ax.arrow(8.8, 4, 0.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')

    plt.title("Figure 1: MODULES IN TRADING SYSTEM", y=-0.05)
    plt.tight_layout()
    plt.savefig("Figure_1_Modules.png", dpi=300)
    print("Generated Figure_1_Modules.png")
    plt.show()

# ==========================================
# FIGURE 2: FUNCTIONAL BLOCK DIAGRAM
# ==========================================
def draw_figure_2():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Coordinates for blocks (Circular Loop Layout)
    # 1. Exchange (Left)
    # 2. Data (Top Left)
    # 3. State (Top Right)
    # 4. Agent (Right)
    # 5. Action (Bottom Right)
    # 6. Risk (Bottom Left)
    
    blocks = {
        'Exchange': (1, 3, "Delta Exchange\n(API)"),
        'Data': (3.5, 5, "Data\nPreprocessor"),
        'State': (6, 5, "State\nVector (Features)"),
        'Agent': (8.5, 5, "PPO Agent\n(Brain)"),
        'Action': (11, 3, "Action\n(Buy/Sell)"),
        'Risk': (6, 1, "Risk Manager\n(Dynamic Lev/TP/SL)")
    }

    # Draw Blocks
    for key, (x, y, label) in blocks.items():
        color = 'white'
        if key == 'Agent': color = '#ff9999' # Red
        elif key == 'Exchange': color = '#99ff99' # Green
        else: color = '#e0e0e0' # Grey
        
        box = patches.FancyBboxPatch((x-1, y-0.5), 2, 1, boxstyle="round,pad=0.2", 
                                     ec="black", fc=color, lw=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, weight='bold')

    # Draw Arrows (The Loop)
    # Exchange -> Data
    ax.annotate("", xy=(2.5, 5), xytext=(1, 4), arrowprops=dict(arrowstyle="->", lw=2))
    # Data -> State
    ax.annotate("", xy=(4.8, 5), xytext=(4.5, 5), arrowprops=dict(arrowstyle="->", lw=2))
    # State -> Agent
    ax.annotate("", xy=(7.3, 5), xytext=(7, 5), arrowprops=dict(arrowstyle="->", lw=2))
    # Agent -> Action
    ax.annotate("", xy=(11, 4), xytext=(9.7, 5), arrowprops=dict(arrowstyle="->", lw=2))
    # Action -> Risk
    ax.annotate("", xy=(8, 1), xytext=(11, 2.3), arrowprops=dict(arrowstyle="->", lw=2))
    # Risk -> Exchange
    ax.annotate("", xy=(1, 2.3), xytext=(4.8, 1), arrowprops=dict(arrowstyle="->", lw=2))

    plt.title("Figure 2: FUNCTIONAL BLOCK DIAGRAM (Closed Loop Control)", y=0.02)
    plt.tight_layout()
    plt.savefig("Figure_2_Flowchart.png", dpi=300)
    print("Generated Figure_2_Flowchart.png")
    plt.show()

# ==========================================
# FIGURE 3: DATA AUGMENTATION (MIRROR WORLD)
# ==========================================
def draw_figure_3():
    # Generate Synthetic Data
    x = np.linspace(0, 100, 200)
    
    # "Real" Bull Market Trend (Random Walk with upward drift)
    np.random.seed(42)
    trend = np.linspace(30000, 60000, 200)
    noise = np.random.normal(0, 2000, 200).cumsum()
    noise = noise - noise.min() # Shift to positive
    price_real = trend + noise * 0.5
    
    # "Augmented" Bear Market (Inverted)
    # Logic: New = 1 / Old (Scaled)
    price_inv = 1 / price_real
    # Rescale for visual comparison on same plot
    scale_factor = np.mean(price_real) / np.mean(price_inv)
    price_aug = price_inv * scale_factor

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Real Data
    ax1.plot(x, price_real, color='blue', label='Original Data (Bull Market)')
    ax1.set_title("Training Sample A: Real Market Regime", fontweight='bold')
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Augmented Data
    ax2.plot(x, price_aug, color='red', label='Augmented Data (Inverted/Bear Market)')
    ax2.set_title("Training Sample B: Mirror World (Synthetic Bear Market)", fontweight='bold')
    ax2.set_ylabel("Inverted Price (Scaled)")
    ax2.set_xlabel("Time (Minutes)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Figure 3: DATA AUGMENTATION VISUALIZATION", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig("Figure_3_Augmentation.png", dpi=300)
    print("Generated Figure_3_Augmentation.png")
    plt.show()

# Run all
if __name__ == "__main__":
    draw_figure_1()
    draw_figure_2()
    draw_figure_3()
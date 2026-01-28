import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Define boxes
boxes1 = [
    ("Delta Exchange\n(Raw Data)", 1.5, 6, "lightblue"),
    ("Data Cleaning\n(Fill Gaps)", 4.5, 6, "lightgreen"),
    ("Feature\nEngineering\n(Indicators)", 7.5, 6, "salmon"),
    ("Data\nAugmentation\n(Mirror World)", 10.5, 6, "wheat"),
    ("Processed\nData", 6, 2, "lightgray")
]

# Draw boxes
for name, x, y, color in boxes1:
    rect = patches.FancyBboxPatch((x-1.2, y-1), 2.4, 2, boxstyle="round,pad=0.1",
                                  linewidth=2, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x, y, name, fontsize=10, weight='bold', ha='center', va='center')

# Draw connecting arrows
ax.arrow(2.8, 6, 0.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
ax.arrow(5.8, 6, 0.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
ax.arrow(8.8, 6, 0.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')

# Arrow to Processed Data
ax.arrow(4.5, 4.9, 0.8, -2, head_width=0.2, head_length=0.2, fc='k', ec='k')
ax.arrow(7.5, 4.9, -0.8, -2, head_width=0.2, head_length=0.2, fc='k', ec='k')
ax.arrow(10.5, 4.9, -2.5, -2, head_width=0.2, head_length=0.2, fc='k', ec='k')


plt.title("Image 1: Data Processing Pipeline (DPP) Module", y=-0.05)
plt.tight_layout()
plt.savefig("Image_1_DPP.png", dpi=300)
print("Generated Image_1_DPP.png")
plt.show()

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Define boxes
boxes2 = [
    ("Agent", 3, 4, "salmon"),
    ("Custom\nGym Environment", 9, 4, "lightgreen"),
    ("State\n(Market Info)", 6, 6.5, "lightblue"),
    ("Action\n(Buy/Sell/Hold)", 6, 1.5, "wheat"),
    ("Reward\n(PnL)", 6, 4, "lightgray")
]

# Draw boxes
for name, x, y, color in boxes2:
    rect = patches.FancyBboxPatch((x-1.2, y-1), 2.4, 2, boxstyle="round,pad=0.1",
                                  linewidth=2, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x, y, name, fontsize=10, weight='bold', ha='center', va='center')

# Draw connecting arrows
# Agent -> Action
ax.arrow(4.3, 3, 0.8, -1, head_width=0.2, head_length=0.2, fc='k', ec='k')
# Action -> Environment
ax.arrow(7.3, 2, 0.8, 1, head_width=0.2, head_length=0.2, fc='k', ec='k')
# Environment -> State
ax.arrow(7.7, 5, -0.8, 1, head_width=0.2, head_length=0.2, fc='k', ec='k')
# State -> Agent
ax.arrow(4.7, 6, -0.8, -1, head_width=0.2, head_length=0.2, fc='k', ec='k')
# Environment -> Reward
ax.arrow(7.7, 4, -0.8, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
# Reward -> Agent
ax.arrow(4.7, 4, -0.8, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')


plt.title("Image 2: Custom Trading Environment Module", y=-0.05)
plt.tight_layout()
plt.savefig("Image_2_Gym_Env.png", dpi=300)
print("Generated Image_2_Gym_Env.png")
plt.show()

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Define boxes
boxes3 = [
    ("State Input", 2, 4, "lightblue"),
    ("PPO Agent\n(Actor-Critic)", 6, 4, "salmon"),
    ("Actor Network\n(Policy)", 6, 6.5, "wheat"),
    ("Critic Network\n(Value Function)", 6, 1.5, "lightgray"),
    ("Action Probabilities", 10, 6.5, "lightgreen"),
    ("State Value Estimate", 10, 1.5, "lightgreen")
]

# Draw boxes
for name, x, y, color in boxes3:
    rect = patches.FancyBboxPatch((x-1.2, y-1), 2.4, 2, boxstyle="round,pad=0.1",
                                  linewidth=2, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x, y, name, fontsize=10, weight='bold', ha='center', va='center')

# Draw connecting arrows
# State Input -> Agent
ax.arrow(3.3, 4, 1.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
# Agent -> Actor
ax.arrow(6, 5.1, 0, 0.8, head_width=0.2, head_length=0.2, fc='k', ec='k')
# Agent -> Critic
ax.arrow(6, 2.9, 0, -0.8, head_width=0.2, head_length=0.2, fc='k', ec='k')
# Actor -> Action Probabilities
ax.arrow(7.3, 6.5, 1.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
# Critic -> State Value Estimate
ax.arrow(7.3, 1.5, 1.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')


plt.title("Image 3: DRL Agent (PPO Model) Module", y=-0.05)
plt.tight_layout()
plt.savefig("Image_3_DRL_Agent.png", dpi=300)
print("Generated Image_3_DRL_Agent.png")
plt.show()

# Create a figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')

# Define boxes
boxes = [
    ("Agent Action", 2, 4, "salmon"),
    ("Risk Manager\n(Leverage, TP/SL)", 6, 4, "wheat"),
    ("API Gateway\n(Auth & Order)", 10, 4, "lightblue"),
    ("Delta Exchange", 10, 1, "lightgreen")
]

# Draw boxes
for name, x, y, color in boxes:
    rect = patches.FancyBboxPatch((x-1.2, y-1), 2.4, 2, boxstyle="round,pad=0.1",
                                  linewidth=2, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x, y, name, fontsize=10, weight='bold', ha='center', va='center')

# Draw connecting arrows
# Agent Action -> Risk Manager
ax.arrow(3.3, 4, 1.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
# Risk Manager -> API Gateway
ax.arrow(7.3, 4, 1.4, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
# API Gateway -> Delta Exchange
ax.arrow(10, 2.9, 0, -1.2, head_width=0.2, head_length=0.2, fc='k', ec='k')


plt.title("Image 4: Live Execution Engine Module", y=-0.05)
plt.tight_layout()
plt.savefig("Image_4_Execution_Engine.png", dpi=300)
print("Generated Image_4_Execution_Engine.png")
plt.show()
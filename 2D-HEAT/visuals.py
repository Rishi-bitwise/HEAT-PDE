import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration ---
SIDE = 256
TOTAL_FRAMES = 64
FRAMES_TO_LOAD = TOTAL_FRAMES // 4  # Load only the first quarter (16 frames)

FILE_NAME = '2d_heat.csv'
OUTPUT_GIF = 'heat_simulation.gif'

# Reduced frame rate
FPS = 4
INTERVAL_MS = 1000 // FPS

print(f"Loading only the first {FRAMES_TO_LOAD} frames from {FILE_NAME}...")

try:
    # Read only the first half of the CSV rows
    raw_data = np.loadtxt(
        FILE_NAME,
        delimiter=',',
        usecols=range(SIDE * SIDE),
        max_rows=FRAMES_TO_LOAD
    )
except FileNotFoundError:
    print(f"Error: {FILE_NAME} not found. Run your CUDA code first.")
    exit()

# Reshape into (32, 256, 256)
heat_frames = raw_data.reshape((FRAMES_TO_LOAD, SIDE, SIDE))
print("Data loaded and reshaped successfully.")

# --- Plot Setup ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("2D Heat Equation Simulation", fontsize=14)
ax.axis('off')

img = ax.imshow(
    heat_frames[0],
    cmap='inferno',
    origin='upper',
    vmin=-100,
    vmax=100
)
fig.colorbar(img, ax=ax, label='Temperature (°C)',
             fraction=0.046, pad=0.04)

# --- Animation Function ---
def update(frame_index):
    img.set_array(heat_frames[frame_index])
    return [img]

print("Generating animation...")
ani = animation.FuncAnimation(
    fig,
    update,
    frames=FRAMES_TO_LOAD,
    interval=INTERVAL_MS,   # 100 ms = 10 FPS
    blit=True,
    repeat=True
)

# --- Save to GIF ---
print(f"Saving to {OUTPUT_GIF}...")
writer = animation.PillowWriter(
    fps=FPS,
    metadata=dict(artist='CUDA Simulation'),
    bitrate=1800
)
ani.save(OUTPUT_GIF, writer=writer)

print("GIF saved!")
plt.show()
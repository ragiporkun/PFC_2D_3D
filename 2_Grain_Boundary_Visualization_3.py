import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.patches as patches
from matplotlib.colors import BoundaryNorm  # added

# --- NEW: imports for clustering ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------
# Grid / domain params (match the psi_3750 grid: 384 x 128)
dx, dy = 1600 / 2**11, 1600 / 2**11
nx, ny = 384, 128

# Load the data
df = pd.read_csv("local_maxima_with_avg_angles_floored_1000_1.csv")

x = np.arange(nx) * dx
y = np.arange(ny) * dy

# Extract coordinates and angles
x_coords = df['x'].values
y_coords = df['y'].values
avg_angles = df['avg_angle'].values
psi_vals = df['psi'].values
phi_vals = df['phi6'].values

# Remove any NaN values
mask = ~np.isnan(avg_angles)
x_coords = x_coords[mask]
y_coords = y_coords[mask]
avg_angles = avg_angles[mask]
psi_vals = psi_vals[mask]
phi_vals = phi_vals[mask]

#left = np.arange(-0.5, -0.1, 0.1)
#center = np.array([-0.1, 0.1])
#right = np.arange(0.1, 0.6, 0.1)

#bounds = np.concatenate([left, center, right])
bounds = np.arange(-0.50, 0.51, 0.1)
cmap_discrete = plt.get_cmap('hsv', 24)
norm_discrete = BoundaryNorm(bounds, cmap_discrete.N, clip=True)


fig = plt.figure(figsize=(20, 12))

# 1. Scatter plot with color-coded angles (DISCRETE, 0..pi/3)
plt.subplot(1, 1, 1)
scatter = plt.scatter(
    x_coords, y_coords,
    c=phi_vals,
    cmap=cmap_discrete,
    norm=norm_discrete,
    s=450, alpha=0.7
)
cbar1 = plt.colorbar(scatter, label='Average Angle (radians)', ticks=bounds, spacing='proportional')
cbar1.set_ticklabels([f'{tick:.2f}' for tick in bounds])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Average Angles')
plt.axis('equal')

plt.show()

import os
import sys
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.getcwd()))
import alphashape


points_2d = [(0., 0.), (0., 1.), (1., 1.), (1., 0.),
          (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)]

alpha_shape = alphashape.alphashape(points_2d, 0.)

fig, ax = plt.subplots()
ax.scatter(*zip(*points_2d))
# Extract exterior coordinates from the shapely polygon
coords = np.array(alpha_shape.exterior.coords)
ax.add_patch(Polygon(coords, alpha=0.2))
plt.show()

import numpy as np
import matplotlib.pyplot as plt


MASTU_WALL_OUTLINE = np.array([
    (0.84, -1.551),
    (0.864, -1.573),
    (0.893, -1.587),
    (0.925, -1.59),
    (1.69, -1.552),
    (1.73, -1.68),
    (1.35, -2.06),
    (1.09, -2.06),
    (0.9, -1.87),
    (0.63, -1.6),
    (0.84, -1.551),
])

plt.ion()
plt.figure()
plt.plot(MASTU_WALL_OUTLINE[:, 0], MASTU_WALL_OUTLINE[:, 1], 'k')
plt.axis('equal')


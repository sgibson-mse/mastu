import numpy as np
import matplotlib.pyplot as plt

from cherab.mastu.mse import load_mse_sightlines
from cherab.mastu.nbi import SS_DEBUG_GEOMETRY

beam_pos = SS_DEBUG_GEOMETRY['position']
beam_direction = SS_DEBUG_GEOMETRY['direction']

los, los_vector = load_mse_sightlines()

samples = np.linspace(0,10,200)
los_x, los_y, los_z = [], [], []

xb, yb, zb = [], [], []

for sample in samples:
    los_sample = los + los_vector * sample
    beam_sample = beam_pos + beam_direction * sample

    los_x.append(los_sample[0])
    los_y.append(los_sample[1])
    los_z.append(los_sample[2])

    xb.append(beam_sample[0])
    yb.append(beam_sample[1])
    zb.append(beam_sample[2])


# ### Plot the Geometry and LOS #######

theta = np.arange(0,360,1)*np.pi/180.
r_in = 0.2
r_out = 1.35

plt.figure()

plt.plot(los_x, los_y, label='LOS Vector')
plt.plot(xb, yb, label='Beam vector')
plt.plot(r_in * np.cos(theta), r_in * np.sin(theta), color='black')
plt.plot(r_out * np.cos(theta), r_out * np.sin(theta), color='black')
plt.plot(beam_pos.x, beam_pos.y, 'o', label='pini')
plt.plot(los.x, los.y, 'o', label='MSE port')
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.legend()
plt.show()
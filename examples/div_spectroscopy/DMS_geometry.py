

# External imports
import pyuda
client = pyuda.Client()
import os
import numpy as np
import matplotlib.pyplot as plt

# Core and external imports
from raysect.core import Vector3D, Point3D
from raysect.core.ray import Ray as CoreRay
from raysect.primitive.mesh import Mesh
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.observer import FibreOptic, SpectralRadiancePipeline0D, PowerPipeline0D
from raysect.optical import World, translate, rotate, rotate_basis, Spectrum

# Cherab and raysect imports
from cherab.core.atomic import Line
from cherab.core.atomic.elements import deuterium, carbon
from cherab.openadas import OpenADAS
from cherab.solps import load_solps_from_mdsplus
from cherab.core.model import ExcitationLine, RecombinationLine, Bremsstrahlung
from cherab.core.model.lineshape import StarkBroadenedLine
from cherab.mastu.machine import *
from cherab.mastu.div_spectrometer import fibres, spectrometer


plt.ion()
world = World()
z_vec = Vector3D(0, 0, 1)

MESH_PARTS = CENTRE_COLUMN + LOWER_DIVERTOR_ARMOUR


for cad_file in MESH_PARTS:
    directory, filename = os.path.split(cad_file[0])
    name, ext = filename.split('.')
    print("importing {} ...".format(filename))
    Mesh.from_file(cad_file[0], parent=world, material=AbsorbingSurface(), name=name)


# Load plasma from SOLPS model
mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
ref_number = 121853
sim        = load_solps_from_mdsplus(mds_server, ref_number)
plasma     = sim.create_plasma(parent=world)
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
mesh       = sim.mesh
vessel     = mesh.vessel

fibgeom = fibres()
group = 1

# plot density from SOLPS 
te_samples = np.zeros((500, 500))
ne_samples = np.zeros((500, 500))

xl, xu = (0.5, 2.0)
yl, yu = (-2.5, -1.0)

xrange = np.linspace(xl, xu, 500)
yrange = np.linspace(yl, yu, 500)

def clamp(n, minn, maxn):
    if n < minn:
        return minn
    elif n > maxn:
        return maxn
    else:
        return n

for i, x in enumerate(xrange):
    for j, y in enumerate(yrange):
        val = plasma.electron_distribution.effective_temperature(x, 0.0, y)
        te_samples[j, i] = clamp(val,0,50.0)
        val = plasma.electron_distribution.density(x, 0.0, y)
        ne_samples[j, i] = val#clamp(val,0,50.0)


f, ax_2d = plt.subplots()
ax_2d.set_ylim([yl, yu])

plt.imshow(ne_samples, extent=[xl, xu, yl, yu], origin='lower', cmap=plt.get_cmap('binary'))
plt.xlim(xl, xu)
plt.ylim(yl, yu)
plt.colorbar()


# Retrieve machine data
print("Retrieving machine components")
print("limiter")
limiter = client.geometry("/limiter/efit", 50000)
print("pfcoils")
pfcoils = client.geometry("/magnetics/pfcoil", 50000)
print("passive")
passive = client.geometry("/passive/efit", 50000)
limiter.plot(ax_2d=ax_2d, show=False, color="grey")
passive.plot(ax_2d=ax_2d, show=False, color="black")
pfcoils.plot(ax_2d=ax_2d, show=False, color="black")

# plot out sightlines
fibgeom.plot_bundles(ax_2d)

plt.show()





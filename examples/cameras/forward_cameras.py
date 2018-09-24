
# Core and external imports
import os
import matplotlib.pyplot as plt

from raysect.optical import World
from raysect.optical.observer import VectorCamera
from raysect.primitive.mesh import Mesh
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.spectralfunction import ConstantSF
from raysect.optical.observer import RGBPipeline2D, SpectralPowerPipeline2D, PowerPipeline2D

# Cherab and raysect imports
from cherab.core.atomic import Line
from cherab.core.atomic.elements import deuterium, carbon
from cherab.core.model import ExcitationLine, RecombinationLine
from cherab.tools.observers import load_calcam_calibration
from cherab.solps import load_solps_from_mdsplus
from cherab.openadas import OpenADAS

from cherab.mastu.machine import CENTRE_COLUMN, LOWER_DIVERTOR_ARMOUR

world = World()

MESH_PARTS = CENTRE_COLUMN + LOWER_DIVERTOR_ARMOUR


for cad_file in MESH_PARTS:
    directory, filename = os.path.split(cad_file)
    name, ext = filename.split('.')
    print("importing {} ...".format(filename))
    Mesh.from_file(cad_file, parent=world, material=AbsorbingSurface(), name=name)  # material=Lambert(ConstantSF(0.25))


# Load plasma from SOLPS model
mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
ref_number = 69636  #69637
sim = load_solps_from_mdsplus(mds_server, ref_number)
plasma = sim.create_plasma(parent=world)
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
mesh = sim.mesh
vessel = mesh.vessel


# Pick emission models
# d_alpha = Line(deuterium, 0, (3, 2))
# plasma.models = [ExcitationLine(d_alpha), RecombinationLine(d_alpha)]

# d_gamma = Line(deuterium, 0, (5, 2))
# plasma.models = [ExcitationLine(d_gamma), RecombinationLine(d_gamma)]

ciii_465 = Line(carbon, 2, ('2s1 3p1 3P4.0', '2s1 3s1 3S1.0'))
plasma.models = [ExcitationLine(ciii_465)]


# Select from available Cameras
# camera_config = load_calcam_calibration('/home/mcarr/mastu/cameras/mug_bulletb_midplane.nc')
# camera_config = load_calcam_calibration('/home/mcarr/mastu/cameras/mug_divcam_isp.nc')
camera_config = load_calcam_calibration('/home/mcarr/mastu/cameras/mug_divcam_sxd.nc')


# RGB pipeline for visualisation
rgb = RGBPipeline2D(display_unsaturated_fraction=0.96, name="sRGB")

# Get the power and raw spectral data for scientific use.
power_unfiltered = PowerPipeline2D(display_unsaturated_fraction=0.96, name="Unfiltered Power (W)")
power_unfiltered.display_update_time = 15
# spectral = SpectralPowerPipeline2D()


# Setup camera for interactive use...
plt.ion()
pixels_shape, pixel_origins, pixel_directions = camera_config
camera = VectorCamera(pixel_origins, pixel_directions, pipelines=[rgb, power_unfiltered], parent=world)
# camera.render_engine = SerialEngine()
camera.spectral_bins = 15
camera.pixel_samples = 1
camera.observe()
plt.ioff()
plt.show()

# Can get access to the underlying spectral data in the spectral pipeline.
# spectral.wavelengths
# spectral.frame.mean[250, 250, :]

# Or setup camera for batch run on cluster
# pixels_shape, pixel_origins, pixel_directions = camera_config
# camera = VectorCamera(pixel_origins, pixel_directions, pixels=pixels_shape, sensitivity=1E-34, parent=world)
# camera.spectral_samples = 15
# camera.pixel_samples = 50
# camera.display_progress = False
# camera.accumulate = True
#
# # start ray tracing
# for p in range(1, 5000):
#     print("Rendering pass {} ({} samples/pixel)...".format(p, camera.accumulated_samples + camera.pixel_samples * camera.spectral_rays))
#     camera.observe()
#     camera.save("mastu_divcam_sxd_dalpha_{}_samples.png".format(camera.accumulated_samples))
#     print()
#



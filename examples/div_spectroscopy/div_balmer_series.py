

# External imports
import os
import numpy as np
import matplotlib.pyplot as plt

# Core and external imports
from raysect.core import Vector3D, Point3D
from raysect.core.ray import Ray as CoreRay
from raysect.primitive.mesh import Mesh
from raysect.optical.material.absorber import AbsorbingSurface
from raysect.optical.observer import FibreOptic, SpectralRadiancePipeline0D
from raysect.optical import World, translate, rotate, rotate_basis, Spectrum

# Cherab and raysect imports
from cherab.core.atomic import Line
from cherab.core.atomic.elements import deuterium, carbon
from cherab.openadas import OpenADAS
from cherab.solps import load_solps_from_mdsplus
from cherab.core.model import ExcitationLine, RecombinationLine, Bremsstrahlung
from cherab.core.model.lineshape import StarkBroadenedLine
from cherab.mastu.machine import *


plt.ion()
world = World()
z_vec = Vector3D(0, 0, 1)

MESH_PARTS = CENTRE_COLUMN + LOWER_DIVERTOR_ARMOUR


for cad_file in MESH_PARTS:
    directory, filename = os.path.split(cad_file)
    name, ext = filename.split('.')
    print("importing {} ...".format(filename))
    Mesh.from_file(cad_file, parent=world, material=AbsorbingSurface(), name=name)


# Load plasma from SOLPS model
mds_server = 'solps-mdsplus.aug.ipp.mpg.de:8001'
ref_number = 69636
sim = load_solps_from_mdsplus(mds_server, ref_number)
plasma = sim.create_plasma(parent=world)
plasma.atomic_data = OpenADAS(permit_extrapolation=True)
mesh = sim.mesh
vessel = mesh.vessel


# Setup deuterium lines
d_alpha = Line(deuterium, 0, (3, 2))
d_beta = Line(deuterium, 0, (4, 2))
d_gamma = Line(deuterium, 0, (5, 2))
d_delta = Line(deuterium, 0, (6, 2))
d_epsilon = Line(deuterium, 0, (7, 2))

d_alpha_excit = ExcitationLine(d_alpha, lineshape=StarkBroadenedLine)
d_alpha_recom = RecombinationLine(d_alpha, lineshape=StarkBroadenedLine)
d_beta_excit = ExcitationLine(d_beta, lineshape=StarkBroadenedLine)
d_beta_recom = RecombinationLine(d_beta, lineshape=StarkBroadenedLine)
d_gamma_excit = ExcitationLine(d_gamma, lineshape=StarkBroadenedLine)
d_gamma_recom = RecombinationLine(d_gamma, lineshape=StarkBroadenedLine)
d_delta_excit = ExcitationLine(d_delta, lineshape=StarkBroadenedLine)
d_delta_recom = RecombinationLine(d_delta, lineshape=StarkBroadenedLine)
d_epsilon_excit = ExcitationLine(d_epsilon, lineshape=StarkBroadenedLine)
d_epsilon_recom = RecombinationLine(d_epsilon, lineshape=StarkBroadenedLine)
plasma.models = [d_alpha_excit, d_alpha_recom, d_beta_excit, d_beta_recom, d_gamma_excit,
                 d_gamma_recom, d_delta_excit, d_delta_recom, d_epsilon_excit, d_epsilon_recom]


start_point = Point3D(1.669, 0, -1.6502)
forward_vector = Vector3D(1-1.669, 0, -2+1.6502).normalise()
up_vector = Vector3D(0, 0, 1.0)

spectra = SpectralRadiancePipeline0D()
fibre = FibreOptic([spectra], acceptance_angle=1, radius=0.001, spectral_bins=800000, spectral_rays=1,
                   pixel_samples=5, transform=translate(*start_point)*rotate_basis(forward_vector, up_vector), parent=world)

fibre.min_wavelength = 350.0
fibre.max_wavelength = 700.0

fibre.observe()


# Find the next intersection point of the ray with the world
intersection = world.hit(CoreRay(start_point, forward_vector))
if intersection is not None:
    hit_point = intersection.hit_point.transform(intersection.primitive_to_world)
else:
    raise RuntimeError("No intersection with the vessel was found.")

# Traverse the ray with equation for a parametric line,
# i.e. t=0->1 traverses the ray path.
parametric_vector = start_point.vector_to(hit_point)
t_samples = np.arange(0, 1, 0.01)

# Setup some containers for useful parameters along the ray trajectory
ray_r_points = []
ray_z_points = []
distance = []
te = []
ne = []
dalpha = np.zeros(len(t_samples))
dgamma = np.zeros(len(t_samples))
dbeta = np.zeros(len(t_samples))
ddelta = np.zeros(len(t_samples))
depsilon = np.zeros(len(t_samples))

# get the electron distribution
electrons = plasma.electron_distribution

# At each ray position sample the parameters of interest.
for i, t in enumerate(t_samples):
    # Get new sample point location and log distance
    x = start_point.x + parametric_vector.x * t
    y = start_point.y + parametric_vector.y * t
    z = start_point.z + parametric_vector.z * t
    sample_point = Point3D(x, y, z)
    ray_r_points.append(np.sqrt(x**2 + y**2))
    ray_z_points.append(z)
    distance.append(start_point.distance_to(sample_point))

    # Sample plasma conditions
    te.append(electrons.effective_temperature(x, y, z))
    ne.append(electrons.density(x, y, z))

    Spectrum(350, 700, 8000)
    # Log magnitude of emission
    dalpha[i] = d_alpha_excit.emission(sample_point, forward_vector, Spectrum(350, 700, 8000)).total() + \
                d_alpha_recom.emission(sample_point, forward_vector, Spectrum(350, 700, 8000)).total()
    dgamma[i] = d_gamma_excit.emission(sample_point, forward_vector, Spectrum(350, 700, 8000)).total() + \
                d_gamma_recom.emission(sample_point, forward_vector, Spectrum(350, 700, 8000)).total()
    dbeta[i] = d_beta_excit.emission(sample_point, forward_vector, Spectrum(350, 700, 8000)).total() + \
               d_beta_recom.emission(sample_point, forward_vector, Spectrum(350, 700, 8000)).total()
    ddelta[i] = d_delta_excit.emission(sample_point, forward_vector, Spectrum(350, 700, 8000)).total() + \
                d_delta_recom.emission(sample_point, forward_vector, Spectrum(350, 700, 8000)).total()
    depsilon[i] = d_epsilon_excit.emission(sample_point, forward_vector, Spectrum(350, 700, 8000)).total() + \
                  d_epsilon_recom.emission(sample_point, forward_vector, Spectrum(350, 700, 8000)).total()


# Normalise the emission arrays
dalpha /= dalpha.sum()
dgamma /= dgamma.sum()
dbeta /= dbeta.sum()
ddelta /= ddelta.sum()
depsilon /= depsilon.sum()

# Plot the trajectory parameters

sim.plot_pec_emission_lines([d_alpha_excit, d_alpha_recom], title='D_alpha')
plt.plot(ray_r_points, ray_z_points, 'k')
plt.plot(ray_r_points[0], ray_z_points[0], 'b.')
plt.plot(ray_r_points[-1], ray_z_points[-1], 'r.')

sim.plot_pec_emission_lines([d_gamma_excit, d_gamma_recom], title='D_gamma')
plt.plot(ray_r_points, ray_z_points, 'k')
plt.plot(ray_r_points[0], ray_z_points[0], 'b.')
plt.plot(ray_r_points[-1], ray_z_points[-1], 'r.')

sim.plot_pec_emission_lines([d_beta_excit, d_beta_recom], title='D_beta')
plt.plot(ray_r_points, ray_z_points, 'k')
plt.plot(ray_r_points[0], ray_z_points[0], 'b.')
plt.plot(ray_r_points[-1], ray_z_points[-1], 'r.')

plt.figure()
plt.plot(distance, te)
plt.xlabel("Ray distance (m)")
plt.ylabel("Electron temperature (eV)")
plt.title("Electron temperature (eV) along ray path")

plt.figure()
plt.plot(distance, ne)
plt.xlabel("Ray distance (m)")
plt.ylabel("Electron density (m^-3)")
plt.title("Electron density (m^-3) along ray path")

plt.figure()
plt.plot(distance, dalpha, label='Dalpha')
plt.plot(distance, dgamma, label='Dgamma')
plt.plot(distance, dbeta, label='Dbeta')
plt.plot(distance, ddelta, label='Ddelta')
plt.plot(distance, depsilon, label='Depsilon')
plt.xlabel("Ray distance (m)")
plt.ylabel("Normalised emission")
plt.title("Normalised emission along ray path")
plt.legend()

plt.show()



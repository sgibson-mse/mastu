
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from scipy.constants import electron_mass, atomic_mass


from raysect.optical import World, translate, rotate_basis, Vector3D, Point3D, Ray
from raysect.optical import Ray, d65_white, World, Point3D, Vector3D, translate, rotate
from raysect.optical.observer import PinholeCamera
from raysect.primitive import Cylinder
from raysect.core.workflow import SerialEngine

from cherab.core import Beam, Plasma, Maxwellian, Species
from cherab.core.math import sample3d, ConstantVector3D, Interpolate1DCubic, IsoMapper2D, IsoMapper3D, AxisymmetricMapper, Blend2D, Constant2D, Constant3D, VectorAxisymmetricMapper, \
    Interpolate2DCubic, Interpolate1DLinear
from cherab.core.atomic import hydrogen, deuterium, carbon, Line
from cherab.core.atomic import elements

from cherab.core.model import SingleRayAttenuator, BeamEmissionLine, ExcitationLine, RecombinationLine
from cherab.core.model import BeamCXLine
from cherab.core.model.beam.beam_emission import SIGMA_TO_PI, SIGMA1_TO_SIGMA0, PI2_TO_PI3, PI4_TO_PI3

from cherab.openadas import OpenADAS

from cherab.mastu.equilibrium import MASTUEquilibrium
from cherab.tools.equilibrium import plot_equilibrium

from cherab.mastu.nbi.pini import load_debugging_pini
from cherab.mastu.mse import load_mse_sightlines


import pyuda

client = pyuda.Client()

def map_r_to_psin(eq):

    """
    Find the normalise psi values that correspond to the radial values from the equilibrium
    :param eq: Equilibrium class
    :return: 1D Interpolation function between the radial co-ordinates of the equilibrium and normalised psi values
    """

    z = eq.magnetic_axis.y

    r = equilibrium.r.data[0,:]

    samples = len(r)

    psin = np.empty(samples)

    for i, ri in enumerate(r):
        psin[i] = eq.psi_normalised(ri, z)

    return Interpolate1DCubic(r, psin)

def get_2d_HRTS_profiles(time, r_values, values, profile_name, plot):

    """
    Function to return the temperature, or density in 2D (R,Z)

    Get the data from the database, find the data at the appropriate timeslice.
    Find the normalised psi values that correspond to the radial co-ordinates of the data.
    Find the values inside of the plasma ie. where psi < 1
    Apply the mask and create interpolation function of r,psi
    Map in 2D the ion temperature, inside the LCFS

    :param time: (float) Requested timeslice
    :param radius (1D array, floats) - Radial co-ordinates for data
    :param data (1D array, floats) - values
    :return: 2D profile in R,Z plane
    """

    #Find the time index
    tidx = equilibrium._find_nearest(values.dims[0].data, time)

    #Get the timeslice
    data = values.data[tidx,:]
    data[np.isnan(data)]=0.
    radius = r_values.data[tidx,:]

    #Interpolate psi onto spatial extent of the data

    interp_psi = map_r_to_psin(eq)

    data_psi = np.empty(len(radius))

    for i, r_pos in enumerate(radius):
        data_psi[i] = interp_psi(r_pos)

    #Find indices inside the plasma

    mask = np.where(data_psi<=1.0)[0][:]

    #Find psi values inside the plasma

    psi_plasma = data_psi[mask]

    #find the psi minimum, ie. center of the plasma

    psi0 = np.argmin(psi_plasma)

    #define the mask from center to plasma edge

    mask = mask[psi0:]

    #This is now the profile from the center of the plasma to the edge as a function of psi normalised

    values_1d = values.data[tidx,mask]
    psi_n = data_psi[mask]

    # values_1d = np.append(values_1d, [0.0])
    # psi_n = np.append(psi_n, [1.0])

    profile_1d = Interpolate1DCubic(psi_n, values_1d, extrapolate=True)

    profile_2d = AxisymmetricMapper(Blend2D(Constant2D(0.0), IsoMapper2D(eq.psi_normalised, profile_1d), eq.inside_lcfs))

    if plot:
        # Plot the 2D profile

        xl, xu = (0.0, 2.0)
        yl, yu = (-2.0, 2.0)

        profile_rz = np.zeros((500, 500))
        xrange = np.linspace(xl, xu, 500)
        yrange = np.linspace(yl, yu, 500)

        for i, x in enumerate(xrange):
            for j, y in enumerate(yrange):
                profile_rz[j, i] = profile_2d(x, 0.0, y)

        plt.figure()
        plt.imshow(profile_rz)
        plt.title('{} in poloidal plane'.format(profile_name))
        plt.colorbar()
        plt.show()

    return psi_n, profile_1d, profile_2d

def return_profile(profile_func):

    xl, xu = (0.0, 2.0)
    yl, yu = (-2.0, 2.0)

    profile_rz = np.zeros((500, 500))
    xrange = np.linspace(xl, xu, 500)
    yrange = np.linspace(yl, yu, 500)

    for i, x in enumerate(xrange):
        for j, y in enumerate(yrange):
            profile_rz[j, i] = profile_func(x, 0.0, y)

    return profile_rz

def carbon_density(profile_func, psi_n):

    """
    Set the carbon density to 1% of the electron density at the LCFS

    :param ne_profile - Interpolation function of the electron density
    :return:C6 density (array) as a function of R,Z

    """

    density_c6 = []

    # set to 1% electron density
    for i, psi in enumerate(psi_n):
        density_c6.append(0.01*profile_func(psi))

    return Interpolate1DCubic(psi_n, density_c6, extrapolate=True)


pulse = 24409
time = 0.25
plot = False

world = World()

#Create Atomic Data Source
adas = OpenADAS(permit_extrapolation=True, missing_rates_return_null=True)  # create atomic data source

#Set up the plasma
plasma = Plasma(parent=world)

#Read in the equilibrium
equilibrium = MASTUEquilibrium(pulse)

#Get the equilibrium for the timeslice of interest
eq = equilibrium.time(time)

#Plot the equilibrium
plot_equilibrium(eq)

# Assign B field to the plasma from the equilibrium
plasma.b_field = VectorAxisymmetricMapper(eq.b_field)

#Add the atomic data
plasma.atomic_data = adas

# Get the flow velocity - Set to zero.
flow_velocity = lambda x, y, z: Vector3D(0,0,0)

# Get HRTS data from database

ti_values = client.get("AYC_TE", pulse) #Ion temperature
ne_values = client.get("AYC_NE", pulse) #Electron Density
t_radius = client.get("ayc_r", pulse)  # Radial co-ordinates

#get electron density and ion temperature profiles

#add a zero to the end of density and psi profiles to make sure there is a definite end to the plasma to make sure the beam attenuation is right

psin, ti_1d, ti_2d = get_2d_HRTS_profiles(time, t_radius, ti_values, profile_name='Ion Temperature', plot=plot)
psin, ne_1d, ne_2d = get_2d_HRTS_profiles(time, t_radius, ne_values, profile_name='Electron Density', plot=plot)

#Return the values for electron density and ion temperature
ne = return_profile(ne_2d)
ti = return_profile(ti_2d)

#Set the carbon density to be 1% of the electron density
density_c6_psi = carbon_density(ne_1d, psin)

#Get the carbon density as a 2D interpolated function
density_c6_rz = AxisymmetricMapper(Blend2D(Constant2D(0.0), IsoMapper2D(eq.psi_normalised, density_c6_psi), eq.inside_lcfs))

#Set the deuterium density
density_d = lambda x, y, z: ne_2d(x,y,z) - 6 * density_c6_rz(x,y,z)

# Produce Maxwellian objects of the D, C6 and electron distributions

d_distribution = Maxwellian(density_d, ti_2d, flow_velocity, deuterium.atomic_weight * atomic_mass)
c6_distribution = Maxwellian(density_c6_rz, ti_2d, flow_velocity, carbon.atomic_weight * atomic_mass)
e_distribution = Maxwellian(ne_2d, ti_2d, flow_velocity, electron_mass)

#Set up each of the species to make up the plasma

d0 = Species(elements.deuterium, 0, d_distribution) #Deuterium, charge 0
d1 = Species(elements.deuterium, 1, d_distribution) #Deuterium, charge 1
c6 = Species(carbon, 6, c6_distribution) # Carbon, charge 6

# Set the plasma composition
plasma.electron_distribution = e_distribution
plasma.composition = [d0, d1, c6] #plasma made up of D, D+ and C6

# # Define the plasma geometry - cylindrical geometry

PLASMA_WIDTH = 2
PLASMA_HEIGHT = 4
integration_step = 0.02
plasma.integrator.step = integration_step
plasma.integrator.min_samples = 1000
plasma.geometry = Cylinder(PLASMA_WIDTH, PLASMA_HEIGHT)
plasma.geometry_transform = translate(0, 0, -PLASMA_HEIGHT/2)

# Add background emission
d_alpha = Line(deuterium, 0, (3, 2))

#Add the atomic model to the plasma, here we just have excitation and recombination. NB: Should add more to this later

# plasma.models = [ExcitationLine(d_alpha), RecombinationLine(d_alpha)]


# ######## Neutral Beam #########
#
#Position of SS PINI grid and SS Duct in machine co-ordinates


attenuator = (SingleRayAttenuator, {'clamp_to_zero': True}) # Need a single ray attenuator for each beamlet.
# emission_models = [(BeamCXLine, {'line':Line(carbon, 5, (8, 7))})]
# emission_models = [(BeamCXLine, {'line':Line(deuterium, 0, (3, 2))})]
emission_models = [(BeamEmissionLine, {'line':Line(deuterium, 0, (3, 2)), 'sigma_to_pi': SIGMA_TO_PI, 'sigma1_to_sigma0':SIGMA1_TO_SIGMA0,
                                       'pi2_to_pi3':PI2_TO_PI3, 'pi4_to_pi3':PI4_TO_PI3})]
mast_pini = load_debugging_pini(plasma, adas, attenuator, emission_models, world, integration_step=0.02)
full, half, third = mast_pini.components

plt.figure()
x, _, z, beam_density = sample3d(full.density, (-1, 1, 200), (0, 0, 1), (4, 7, 200))
plt.imshow(np.transpose(np.squeeze(beam_density)), extent=[-1, 1, 4, 7], origin='lower')
plt.colorbar()
plt.axis('equal')
plt.xlabel('x axis (beam coords)')
plt.ylabel('z axis (beam coords)')
plt.title("Beam full energy density profile in r-z plane")

z = np.linspace(4, 7, 200)

beam_full_densities = [full.density(0, 0, zz) for zz in z]
beam_half_densities = [half.density(0, 0, zz) for zz in z]
beam_third_densities = [third.density(0, 0, zz) for zz in z]
plt.figure()
plt.plot(z, beam_full_densities, label="full energy")
plt.plot(z, beam_half_densities, label="half energy")
plt.plot(z, beam_third_densities, label="third energy")
plt.xlabel('z axis (beam coords)')
plt.ylabel('beam component density [m^-3]')
plt.title("Beam attenuation by energy component")
plt.legend()
plt.show()

# # ##### Observation #####

los, los_vector = load_mse_sightlines()

ray = Ray(origin=Point3D(los.x, los.y, los.z), direction=los_vector, min_wavelength=600, max_wavelength=680, bins=2000)
s = ray.trace(world)
plt.figure()
plt.plot(s.wavelengths, s.samples)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Radiance (W/m^2/str/nm)')
plt.title('Sampled BES Spectrum')
plt.show()

print()
print(s.samples.min(), s.samples.max())


# camera = PinholeCamera((256, 256), parent=world, transform=translate(0.6, -1, 7)*rotate_basis(Vector3D(0, 0, -1), Vector3D(0, 1, 0)))
# camera.pixel_samples = 50
# camera.spectral_bins = 15
# camera.observe()
#
# camera = PinholeCamera((256, 256), fov=52.9, parent=world, transform=translate(*los)*rotate_basis(Vector3D(*los_vector), Vector3D(0, 0, 1)))
# camera.pixel_samples = 50
# camera.spectral_bins = 15
# camera.observe()


# los = Point3D(-1.742, 1.564, 0.179) #camera position
# los_vector = Vector3D(0.919, -0.389, -0.057).normalise() #optical axis

# transform = translate(los.x, los.y, los.z) * rotate_basis(los_vector, up)
import re
import numpy as np

from raysect.core import Point3D, Vector3D, translate, rotate_basis
# from raysect.core.scenegraph.node import Node
from raysect.core import Node

from cherab.openadas import OpenADAS
from cherab.core.atomic.elements import deuterium
from cherab.core import Beam

import pyuda

EDGE_WIDENING = 0.01
atomic_data = OpenADAS(permit_extrapolation=True)

PINI_LENGTHS = [5.17]

SS_DEBUG_GEOMETRY = {
    'position': Point3D(0.188819939,-6.88824321,0.0),
    'direction': Vector3D(0.07039384131953216, 0.9975192765577418, 0.0),
    'length': 10.0}

SS_DEBUG_ENERGIES = {
    'energy': 65000/2,
    'power': 1e+06}


class MASTPini(Node):

    """ Object representing a PINI in a scenegraph.

    Create a ready-to-observe PINI for charge-exchange spectroscopy.

    :param tuple pini_geometry: a tuple containing:
        * the source (Point3D),
        * the direction (Vector3D),
        * the divergence (Divergence of beam sigma, tuple of two angles in degrees, horizontal then vertical),
        * the initial width (Gaussian beam sigma, in meters)
        * the length (float in meters)
    of the PINI.
    :param tuple pini_parameters: a tuple containing:
        * the first component energy (float in eV/amu),
        * the total beam power in W,
        * the species
    of the PINI.
    :param Plasma plasma:
    :param attenuation_instructions:
    :param emission_instructions:
    :param parent: the scenegraph parent, default is None.
    :param name:
    """

    def __init__(self, pini_geometry, pini_parameters, plasma, atomic_data, attenuation_instructions,
                 emission_instructions, integration_step=0.02, parent=None, name=""):

        source, direction, divergence, initial_width, length = pini_geometry
        energy, power_fractions, element = pini_parameters

        self._components = []
        self._length = length
        self._parent_reminder = parent

        # Rotation between 'direction' and the z unit vector
        # This is important because the beam primitives are defined along the z axis.
        self._origin = source
        self._direction = direction
        direction.normalise()
        rotation = rotate_basis(direction, Vector3D(0., 0., 1.))
        transform_pini = translate(*source) * rotation

        Node.__init__(self,
                      parent=parent,
                      transform=transform_pini,
                      name=name)

        attenuation_model_class, attenuation_model_arg = attenuation_instructions

        power_fractions = [SS_DEBUG_ENERGIES['power']*0.5, SS_DEBUG_ENERGIES['power']*0.25, SS_DEBUG_ENERGIES['power']*0.25]

        # the 3 energy components are different beams
        for comp_nb in [1, 2, 3]:

            # creation of the attenuation model
            # Note that each beamlet needs its own attenuation class instance.
            attenuation_model = attenuation_model_class(**attenuation_model_arg)

            # creation of the emission models
            emission_models = []
            for (emission_model_class, argument_dictionary) in emission_instructions:
                emission_models.append(emission_model_class(**argument_dictionary))

            beam = Beam(parent=self, transform=translate(0., 0., 0.), name="Beam component {}".format(comp_nb))
            beam.plasma = plasma
            beam.atomic_data = atomic_data
            beam.energy = energy / comp_nb
            beam.power = power_fractions[comp_nb - 1]
            beam.element = element
            beam.sigma = initial_width
            beam.divergence_x = divergence[0]
            beam.divergence_y = divergence[1]
            beam.length = length
            beam.attenuator = attenuation_model
            beam.models = emission_models
            beam.integrator.step = integration_step
            beam.integrator.min_samples = 10

            self._components.append(beam)

    @property
    def origin(self):
        return self._origin

    @property
    def direction(self):
        return self._direction

    @property
    def length(self):
        return self._length

    @property
    def components(self):
        return self._components

    @property
    def energy(self):
        return self._components[0].energy  # first component energy

    @energy.setter
    def energy(self, value):
        for i in range(3):
            component = self._components[i]
            component.energy = value / (i + 1)

    @property
    def power_fractions(self):
        return self._components[0].power, self._components[1].power, self._components[2].power

    @power_fractions.setter
    def power_fractions(self, value):
        for i in range(3):
            self._components[i].power = value[i]

    @property
    def power(self):
        total_power = 0.
        for component in self._components:
            total_power += component.power
        return total_power

    @property
    def element(self):
        return self._components[0].element

    @element.setter
    def element(self, value):
        for component in self._components:
            component.element = value

    def emission_function(self, point, direction, spectrum):

        for beam in self._components:
            spectrum = beam.emission_function(point, direction, spectrum)

        return spectrum


# def load_pini_from_uda(shot, beam_id, plasma, atomic_data, attenuation_instructions, emission_instructions,
#                        world, integration_step=0.02):
#     """
#     Create a new JETPini instance for given pini ID from the NBI PPF settings.
#
#     :param int shot: Shot number.
#     :param beam_id: Code for beam to load. Either SS or SW
#     :param Plasma plasma: Plasma this pini will use for attenuation and emission calculations.
#     :param attenuation_instructions:
#     :param emission_instructions:
#     :param world:
#     :return: Loaded MAST pini from uda.
#     """
#
#     if not beam_id in ['SS','SW']:
#         raise ValueError('Not a valid beam id. Must be either SS or SW.')
#
#
#     # TODO - need to load pini geometry from a central location
#     source, direction, divergence, initial_width, length = get_pini_alignment(shot, int(pini_index))
#
#     # 1/e width is converted in standard deviation, assuming a gaussian shape.
#     # TODO - check whether inital width is one side of the Gaussian or full width.
#     # Code below implies this is the full width, not the half.
#     initial_width = initial_width / (2 * np.sqrt(2))
#     # 1/e width divergences are converted in standard deviation divergences, assuming a gaussian shape.
#     divergence = (np.rad2deg(np.arctan(np.tan(np.deg2rad(divergence[0]))/np.sqrt(2))),
#                   np.rad2deg(np.arctan(np.tan(np.deg2rad(divergence[1]))/np.sqrt(2))))
#
#     pini_geometry = source, direction, divergence, initial_width, length
#
#     ########################################################
#     # Load pini parameters from PPF -> assemble output tuple
#
#     # first component energy (float in eV/amu)
#     ppf.ppfuid('JETPPF', rw='R')
#     ppf.ppfgo(pulse=shot, seq=0)
#     _, _, data, _, _, ierr = ppf.ppfget(shot, 'NBI'+octant, 'ENG'+pini_index)
#     if ierr != 0:
#         raise OSError('No available NBI{}.{}'.format(octant, pini_index))
#
#     energy = data[0] / deuterium.atomic_weight
#
#     # tuple of three power fractions corresponding to decreasing energies, in W),
#     _, _, data, _, _, _ = ppf.ppfget(shot, 'NBI'+octant, 'PFR'+pini_index)
#     power_fractions = tuple(data)
#
#     # Make an NBI masking function from NBL* power level time signal.
#     _, _, data, _, t, _ = ppf.ppfget(shot, 'NBI'+octant, 'NBL'+pini_index)
#     mask = np.empty(len(t), dtype=np.bool_)
#     for i in range(len(t)):
#         if data[i] > 250000:
#             mask[i] = True
#         else:
#             mask[i] = False
#     turned_on = TimeSeriesMask(mask, t)
#
#     # Assemble tuple of pini parameters
#     pini_parameters = (energy, power_fractions, turned_on, deuterium)
#
#     # Construct JETPini and return
#     return JETPini(pini_geometry, pini_parameters, plasma, atomic_data, attenuation_instructions,
#                    emission_instructions, integration_step=integration_step, parent=world)


def load_debugging_pini(plasma, atomic_data, attenuator, emission_models, world, integration_step=0.02):
    """
    Load a MAST pini with preconfigured debugging settings.

    :param pini_id: Code for pini to load.
    :param Plasma plasma: Plasma this pini will use for attenuation and emission calculations.
    :param attenuation_instructions:
    :param emission_instructions:
    :param world:
    :return: Loaded MAST pini from PPF.
    """

    origin = SS_DEBUG_GEOMETRY['position']
    direction = SS_DEBUG_GEOMETRY['direction']
    divergence = (0.499995, 0.700488)
    initial_width = 0.001  # Approximate with 1mm as an effective point source.
    pini_length = SS_DEBUG_GEOMETRY['length']
    pini_geometry = (origin, direction, divergence, initial_width, pini_length)

    energy = SS_DEBUG_ENERGIES['energy']

    power_fractions = [SS_DEBUG_ENERGIES['power'] * 0.5, SS_DEBUG_ENERGIES['power'] * 0.25,
                       SS_DEBUG_ENERGIES['power'] * 0.25]

    pini_parameters = (energy, power_fractions, deuterium)

    # Construct MASTPini and return
    return MASTPini(pini_geometry, pini_parameters, plasma,
                   atomic_data, attenuator, emission_models, integration_step=integration_step, parent=world)


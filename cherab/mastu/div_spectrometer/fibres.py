import matplotlib.pyplot as plt
import math
import numpy as np
from cherab.mastu.machine import *

radians = 3.14159265 / 180.0


class fibres:
    """
    Geometry data for fibre bundles
    """

    def __init__(self):
        self.loaded = -1
        self.zangle = 0.0
        self.xangle = 0.0
        self.set_fibre_details()
        self.set_lens()
        self.set_bundle(group=1)
        self.set_fibre(number=1)

    def set_fibre_details(self,number=20):
        self.diam = 400E-6
        self.ferule_spacing=200E-6
        self.numfibres=number

    def set_lens(self,focal_length=50E-3):
        self.focal_length=focal_length
        self.fov = 2.0 * np.rad2deg(np.arctan((self.diam + self.ferule_spacing) * self.numfibres / 2.0 / self.focal_length))
        
    def set_bundle(self, group=None):
        if group == 1:
            self.group = 1
            self.load_HL01_B()
            self.loaded = 1
        if group == 2:
            self.group = 2
            self.load_HL02_B()
            self.loaded = 1
        if group == 3:
            self.group = 3
            self.load_HU01_A()
            self.loaded = 1
        if group == 4:
            self.group = 4
            self.load_HU02_A()
            self.loaded = 1
        if group == 5:
            self.group = 5
            self.load_HE05_1()
            self.loaded = 1
        if group == 6:
            self.group = 6
            self.load_HM10_H()
            self.loaded = 1

        if self.loaded == -1:
            print("Group not recognised - loading failed")
        else:
            self.yangle = self.calc_yangle()

    def set_fibre(self, number=1):
        if number > self.numfibres:
            print("Warning: Fibre does not exist - setting to 1")
            number = 1
        self.fibre = number
        self.yangle = self.calc_yangle()
        if self.group == 6:
            self.load_HM10_H()
            viewang = np.linspace(-3.8,3.8,20.0)
            self.zangle = self.zangle - viewang[self.fibre - 1]
        else:
            viewang = np.linspace(-self.fov/2.0,self.fov/2.0,self.numfibres)
            self.yangle = self.yangle - viewang[self.fibre - 1]

    def load_HL01_B(self):
        self.origin = (0.500, 2.080, -1.320)
        self.target = (0.95, -2.08)
        self.zangle = 0.0

    def load_HL02_B(self):
        self.origin = (1.420, 1.604, -1.500)
        self.target = (0.00, -2.05)
        self.zangle = 0.0

    def load_HU01_A(self):
        self.origin = (0.553, 2.066, 1.320)
        self.target = (0.95, 2.08)
        self.zangle = -1.0

    def load_HU02_A(self):
        self.origin = (1.508, 1.513, 1.560)
        self.target = (0.00, 1.91)
        self.zangle = 0.0

    def load_HE05_1(self):
        self.origin = (1.166, -0.961, 0.650)
        self.target = (0.70, 1.05)
        self.zangle = 10.0

    def load_HM10_H(self):
        self.origin = (-2.066,0.553 , -0.10)
        self.target = (0.00, 0.00)
        self.zangle = -34.0

    def Rval(self):
        return np.sqrt(self.origin[0]**2 + self.origin[1]**2)

    def rotangle(self):
        return math.acos(self.origin[0] / self.Rval()) * np.sign(self.origin[1])

    def calc_yangle(self):
        return 180 - math.atan((self.origin[2] - self.target[1]) / (self.Rval() - self.target[0])) / radians

    def xhat(self):
        xhat = (math.cos(self.zangle * radians) * math.cos(self.yangle * radians))
        yhat = (math.sin(self.zangle * radians) * math.cos(self.xangle * radians) + \
                math.cos(self.zangle * radians) * math.sin(self.yangle * radians) * math.sin(self.xangle * radians))
        rad  = np.sqrt(yhat ** 2 + (self.Rval() - xhat) ** 2)
        ang  = math.acos((self.Rval() - xhat) / rad) * np.sign(yhat) + self.rotangle()
        return self.origin[0] - rad * math.cos(ang)

    def yhat(self):
        xhat = (math.cos(self.zangle * radians) * math.cos(self.yangle * radians))
        yhat = (math.sin(self.zangle * radians) * math.cos(self.xangle * radians) + \
                math.cos(self.zangle * radians) * math.sin(self.yangle * radians) * math.sin(self.xangle * radians))
        rad  = np.sqrt(yhat ** 2 + (self.Rval() - xhat) ** 2)
        ang  = math.acos((self.Rval() - xhat) / rad) * np.sign(yhat) + self.rotangle()
        return self.origin[1] - rad * math.sin(ang)

    def zhat(self):
        return math.sin(self.zangle * radians) * math.sin(self.xangle * radians) - \
               math.cos(self.zangle * radians) * math.sin(self.yangle * radians) * math.cos(self.xangle * radians)

    def load_cad(self, load_full=False):
        from raysect.primitive.mesh import Mesh
        from raysect.optical.material.absorber import AbsorbingSurface
        from raysect.optical import World
        import os
        world = World()

        if load_full:
            MESH_PARTS = MASTU_FULL_MESH + VACUUM_VESSEL + \
                         UPPER_DIVERTOR_NOSE + UPPER_DIVERTOR_ARMOUR + UPPER_DIVERTOR + \
                         LOWER_DIVERTOR_NOSE + LOWER_DIVERTOR_ARMOUR + LOWER_DIVERTOR + \
                         LOWER_ELM_COILS + LOWER_GAS_BAFFLE + \
                         UPPER_ELM_COILS + UPPER_GAS_BAFFLE + \
                         ELM_COILS + PF_COILS + \
                         T5_LOWER + T4_LOWER + T3_LOWER + T2_LOWER + T1_LOWER + \
                         T5_UPPER + T4_UPPER + T3_UPPER + T2_UPPER + T1_UPPER + \
                         C6_TILES + C5_TILES + C4_TILES + C3_TILES + C2_TILES + C1_TILE + \
                         B1_UPPER + B2_UPPER + B3_UPPER + B4_UPPER + \
                         B1_LOWER + B2_LOWER + B3_LOWER + B4_LOWER +  \
                         BEAM_DUMPS + SXD_BOLOMETERS
        else:
            MESH_PARTS = CENTRE_COLUMN + LOWER_DIVERTOR_ARMOUR

        #MESH_PARTS=VACUUM_VESSEL
        for cad_file in MESH_PARTS:
            directory, filename = os.path.split(cad_file[0])
            name, ext = filename.split('.')
            print("importing {} ...".format(filename))
            Mesh.from_file(cad_file[0], parent=world, material=AbsorbingSurface(), name=name)
        return world

    def fibre_distance_world(self, world):

        from cherab.tools.observers.intersections import find_wall_intersection
        from raysect.core import Vector3D, Point3D

        start_point = Point3D(self.origin[0], self.origin[1], self.origin[2])
        forward_vector = Vector3D(self.xhat(), self.yhat(), self.zhat())

        hit_point, primitive = find_wall_intersection(world, start_point, forward_vector, delta=1E-3)

        return abs((self.origin[0] - hit_point[0]) / self.xhat())

    def fibre_distance(self):
        if self.group <= 2:
            pt1 = [0.33, -1.303]
            pt2 = [1.09, -2.060]
            pt1i = [1.09, -2.06]
            pt2i = [1.35, -2.06]
        elif self.group <= 5 and self.group > 2:
            pt1 = [0.33, 1.303]
            pt2 = [1.09, 2.060]
            pt1i = [1.09, 2.06]
            pt2i = [1.35, 2.06]

        dist = 1.4

        xtarg = self.origin[0] + self.xhat() * dist
        ytarg = self.origin[1] + self.yhat() * dist
        ztarg = self.origin[2] + self.zhat() * dist
        rtarg = np.sqrt(xtarg**2+ytarg**2)
        target = (rtarg, ztarg)
        origin = (self.Rval(),self.origin[2])
        iint = line_intersection((origin, target), (pt1, pt2))
        if self.group <= 2:
            if iint[1] < pt2[1]:
                iint = line_intersection((origin, target), (pt1i, pt2i))
        elif self.group <= 4 and self.group > 2:
            if iint[1] > pt2[1]:
                iint = line_intersection((origin, target), (pt1i, pt2i))
        xint = math.cos(self.rotangle()) * iint[0]

        return (xint - self.origin[0]) / self.xhat()

    def plot_bundles(self, ax_2d):
        import copy as copy
        import matplotlib.pyplot as plt
        import numpy as np

        orig_fibre = copy.deepcopy(self.fibre)
        orig_group = copy.deepcopy(self.group)
        lrange = 6
        try:
            self.world
        except:
            self.world = self.load_cad()

        for i in range(0, lrange):
            bundle = i + 1
            self.set_bundle(group=bundle)
            for j in range(1, self.numfibres+1):
                self.set_fibre(number=j)
                try:
                    dist_var = np.arange(0, self.fibre_distance_world(self.world), 0.001, dtype=float)
                except:
                    dist_var = np.arange(0, 3.5, 0.001, dtype=float)
                zarr = []
                rarr = []

                for k in range(0, len(dist_var)):
                    xtarg = self.origin[0] + self.xhat() * dist_var[k]
                    ytarg = self.origin[1] + self.yhat() * dist_var[k]
                    ztarg = self.origin[2] + self.zhat() * dist_var[k]

                    rtarg = np.sqrt(xtarg ** 2 + ytarg ** 2)
                    rarr.append(rtarg)
                    zarr.append(ztarg)

                if i == 1 or i == 3:
                    plt.plot(rarr, zarr, 'b', alpha=0.7)
                else:
                    plt.plot(rarr, zarr, 'r', alpha=0.7)

        plt.savefig('DMS_Geometry.png')
        plt.show()
        self.set_fibre(number=orig_fibre)
        self.set_bundle(group=orig_group)
    def plot_bundles_birdseye(self):
        import copy as copy
        import matplotlib.pyplot as plt
        import numpy as np
        orig_fibre = copy.deepcopy(self.fibre)
        orig_group = copy.deepcopy(self.group)
        lrange = 6
        plt.figure()
        x = []
        y = []
        for i in range(0,360):
            x.append(math.cos(i*radians) * 1.5)
            y.append(math.sin(i*radians) * 1.5)

        plt.plot(x,y,'k')
        x = []
        y = []
        for i in range(0,360):
            x.append(math.cos(i*radians) * 1.2)
            y.append(math.sin(i*radians) * 1.2)

        plt.plot(x,y,'k--')
        x = []
        y = []
        for i in range(0, 360):
            x.append(math.cos(i * radians) * 2.0)
            y.append(math.sin(i * radians) * 2.0)

        plt.plot(x, y,'k')

        for i in range(0,12):
            ang = i * 360.0/12
            plt.plot((0,2.0*math.cos(ang * radians)),(0,2.0 * math.sin(ang * radians)),'k')

        try:
            self.world
        except:
            self.world = self.load_cad()

        for i in range(0, lrange):
            bundle = i + 1
            self.set_bundle(group=bundle)
            for j in range(1, self.numfibres+1):
                self.set_fibre(number=j)
                try:
                    dist_var = np.arange(0, self.fibre_distance_world(self.world), 0.001, dtype=float)
                except:
                    dist_var = np.arange(0, 3.5, 0.001, dtype=float)

                zarr = []
                rarr = []
                for k in range(0, len(dist_var)):
                    xtarg = self.origin[0] + self.xhat() * dist_var[k]
                    ytarg = self.origin[1] + self.yhat() * dist_var[k]
                    ztarg = self.origin[2] + self.zhat() * dist_var[k]
                    rtarg = np.sqrt(xtarg ** 2 + ytarg ** 2)
                    rarr.append(xtarg)
                    zarr.append(ytarg)

                if i == 1 or i == 3:
                    plt.plot(rarr, zarr, 'b', alpha=0.7)
                else:
                    plt.plot(rarr, zarr, 'r', alpha=0.7)

        plt.show()
        self.set_fibre(number=orig_fibre)
        self.set_bundle(group=orig_group)
    def plot_bundle(self, ax_2d,col='b'):
        import copy as copy
        import matplotlib.pyplot as plt
        import numpy as np

        orig_fibre = copy.deepcopy(self.fibre)
        try:
            self.world
        except:
            self.world = self.load_cad()

        for j in range(1, self.numfibres+1):
            self.set_fibre(number=j)
            try:
                dist_var = np.arange(0, self.fibre_distance_world(self.world), 0.001, dtype=float)
            except:
                dist_var = np.arange(0, 3.5, 0.001, dtype=float)
            zarr = []
            rarr = []
            for k in range(0, len(dist_var)):
                xtarg = self.origin[0] + self.xhat() * dist_var[k]
                ytarg = self.origin[1] + self.yhat() * dist_var[k]
                ztarg = self.origin[2] + self.zhat() * dist_var[k]

                rtarg = np.sqrt(xtarg ** 2 + ytarg ** 2)
                rarr.append(rtarg)
                zarr.append(ztarg)
            plt.plot(rarr, zarr, col, alpha=0.7)

        self.set_fibre(number=orig_fibre)
    def plot_bundle_birdseye(self):
        import copy as copy
        import matplotlib.pyplot as plt
        import numpy as np
        orig_fibre = copy.deepcopy(self.fibre)
        plt.figure()
        x = []
        y = []
        for i in range(0,360):
            x.append(math.cos(i*radians) * 1.5)
            y.append(math.sin(i*radians) * 1.5)

        plt.plot(x,y,'k')
        x = []
        y = []
        for i in range(0,360):
            x.append(math.cos(i*radians) * 1.2)
            y.append(math.sin(i*radians) * 1.2)

        plt.plot(x,y,'k--')
        x = []
        y = []
        for i in range(0, 360):
            x.append(math.cos(i * radians) * 2.0)
            y.append(math.sin(i * radians) * 2.0)

        plt.plot(x, y,'k')

        for i in range(0,12):
            ang = i * 360.0/12
            plt.plot((0,2.0*math.cos(ang * radians)),(0,2.0 * math.sin(ang * radians)),'k')

        try:
            self.world
        except:
            self.world = self.load_cad()

        for j in range(1, self.numfibres+1):
            self.set_fibre(number=j)
            try:
                dist_var = np.arange(0, self.fibre_distance_world(self.world), 0.001, dtype=float)
            except:
                dist_var = np.arange(0, 3.5, 0.001, dtype=float)

            zarr = []
            rarr = []
            for k in range(0, len(dist_var)):
                xtarg = self.origin[0] + self.xhat() * dist_var[k]
                ytarg = self.origin[1] + self.yhat() * dist_var[k]
                ztarg = self.origin[2] + self.zhat() * dist_var[k]
                rtarg = np.sqrt(xtarg ** 2 + ytarg ** 2)
                rarr.append(xtarg)
                zarr.append(ytarg)

            plt.plot(rarr, zarr, 'b', alpha=0.7)

        plt.show()
        self.set_fibre(number=orig_fibre)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


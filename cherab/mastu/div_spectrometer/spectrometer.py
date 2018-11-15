import os
import numpy as np
import matplotlib.pyplot as plt
import math

class spectrometer:
    """
    Spectrometer and detector settings
    """
    def __init__(self):
        self.detector="proEM:1024B"
        self.pixels=1024
        self.spectrograph="SP2500i"
        self.grating=300
        self.wcentre=402.0
        self.set_spec()
    def check_inputs(self):
        if self.grating != 300:
            if self.grating  != 600:
                if self.grating != 1200:
                    print("Warning: Only gratings 300/600/1200 available - setting 300")
                    self.grating = 300
    def set_spec(self):
        self.check_inputs()
        self.wrange()
        self.wupper = self.wcentre + self.range/2.0
        self.wlower = self.wcentre - self.range/2.0
        self.wresol = (self.wupper - self.wlower)/self.pixels
    def wrange(self):
        self.range=0.0
        if self.grating == 300:
            self.range = 88.0
        if self.grating == 600:
            self.range = 44.0
        if self.grating == 1200:
            self.range = 20
    def set_grating(self,grating):
        self.grating=grating
        self.set_spec()
    def set_wcentre(self,centre):
        self.wcentre=centre
        self.set_spec()
    def set_range(self,setting=1):
        if setting == 1:
            # Primary setting outputs:
            #   Continuum ratio ~ Te
            #   Stark broadening ~ ne
            #   D[gamma,delta,epsilon] ratio
            #   Impurity/molecular (CD) emission
            self.set_grating(300)
            self.set_wcentre(395.0)
        if setting == 2:
            # Secondary setting outputs:
            #   Doppler broadening (He II/C III/C IV)
            #   CX measurement (C IV)
            self.set_grating(1200)
            self.set_wcentre(465.0)
        if setting == 3:
            # Secondary setting outputs:
            #   He I (668/706/728 nm) line ratios
            self.set_grating(300)
            self.set_wcentre(694.0)

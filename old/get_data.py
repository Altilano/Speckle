import os
import time
import sys
import clr
import numpy as np
import matplotlib.pyplot as plt
from slmcontrol import *
import time 
from pylablib.devices import Thorlabs
from slmcontrol import *
from cameras.Ximea import XimeaCamera
from cameras.ImagingSource import ImagingSourceCamera

# Add References to .NET libraries

clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.KCube.StepperMotorCLI.dll")
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.StepperMotorCLI import *
from System import Decimal  # necessary for real world units

#input beam


camera = ImagingSourceCamera()



#input beam
slm = SLMdisplay(monitor=1)
config_path = 'config.ini'
config = configparser.ConfigParser()
config.read(config_path)
x,y = build_grid(config_path)
input = hg(x,y,0,0,config['incoming'].getfloat('waist'))

stage_z = Thorlabs.kinesis.KinesisMotor('26005619',scale='ZST',  default_channel=2, is_rack_system=False)
print(f" stage z  -- {stage_z.get_position(scale= 'ZST')}")
stages = [stage_z]

for _ in stages:
    print(_.is_homed())
    if _.is_homed() == False:
        _.home()
for _ in stages:
    _.wait_for_home()

    stage_z.move_to(0.010) # in meter
stage_z.wait_move()
resolution = 20
step_size = 100e-9

ls = 1                  #laguerre vai até quanto?


#data = np.empty((resolution,ls,480, 640),dtype="uint8")
data = np.empty((resolution,480, 640),dtype="uint8")

"""for i in range(resolution):
        stage_z.move_by(step_size)
        stage_z.wait_move()
        for n in range(ls):
            desired = hg(x, y, n +1, 0, .5)
            holo = generate_hologram(desired,input,x,y,195,3,4,0,0,method='bessel1')
            slm.updateArray(holo)
            time.sleep(0.15)
            data[i,n] = camera.capture()"""

for i in range(resolution):
        stage_z.move_by(step_size)
        stage_z.wait_move()
        desired = hg(x, y, 1, 0, .5)
        holo = generate_hologram(desired,input,x,y,195,3,4,0,0,method='bessel1')
        slm.updateArray(holo)
        time.sleep(0.15)
        data[i] = camera.capture()

camera.close()


np.savez('hg10 step100e-9', data)
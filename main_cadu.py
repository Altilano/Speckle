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

"""camera.close()
print('finished')
exit()"""

resolution = 10
step_size = 3e-6



hg_modes = [[0,0], [0,1], [1,0], [2,0], [1,1], [0,2], [3,0], [2,1], [1,2], [0,3], [4,0], [3,1], [2,2], [1,3], [0,4], [5,0], [4,1], [3,2], [2,3], [1,4], [0,5]]
lg_modes = [[0,1], [0,-1], [0,2], [1,0], [0,-2], [0,3], [1,1], [1,-1], [0,-3], [0,4], [1, 2], [1,-2], [2, 0], [0,-4], [0,5], [1,3], [1, -3], [2, 1], [2,-1], [0,-5]]

hg_cadu = [[0,0], [0,1], [1,0], [2,0], [1,1], [0,2]]
lg_cadu = [[0,1], [0,-1], [0,2], [1,0], [0,-2]]



for mode in range(len(hg_cadu)):
    stage_z.move_to(0.003) # in meter
    stage_z.wait_move()
    stage_z.move_to(0.004) # in meter
    stage_z.wait_move()

    data = np.empty((resolution,480, 640),dtype="uint8")

    print('measuring hg mode:', hg_cadu[mode])
    n = hg_cadu[mode][0]
    m = hg_cadu[mode][1]

    desired = hg(x, y, n, m, .5)
    holo = generate_hologram(desired,input,x,y,195,3,4,0,0,method='bessel1')
    slm.updateArray(holo)
    time.sleep(0.1)


    for i in range(resolution):
        stage_z.move_by(step_size)
        stage_z.wait_move()
        time.sleep(0.15)
        data[i] = camera.capture()

    np.savez(f'Amostras_Cadu\\amostra120_hg{n}{m}_step3e-6_Resolution{resolution}vertical', data)



for mode in range(len(lg_cadu)):
    stage_z.move_to(0.003) # in meter
    stage_z.wait_move()
    stage_z.move_to(0.004) # in meter
    stage_z.wait_move()

    data = np.empty((resolution,480, 640),dtype="uint8")

    print('measuring lg mode:', lg_cadu[mode])
    n = lg_cadu[mode][0]
    m = lg_cadu[mode][1]

    desired = lg(x, y, n, m, .5)
    holo = generate_hologram(desired,input,x,y,195,3,4,0,0,method='bessel1')
    slm.updateArray(holo)
    time.sleep(0.1)


    for i in range(resolution):
        stage_z.move_by(step_size)
        stage_z.wait_move()
        time.sleep(0.15)
        data[i] = camera.capture()

    np.savez(f'Amostras_Cadu\\amostra120_lg{n}{m}_step3e-6_Resolution{resolution}vertical', data)



camera.close()

print('the end')

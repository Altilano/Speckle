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

resolution = 100
step_size = 100e-9



hg_modes = [[0,0], [0,1], [1,0], [2,0], [1,1], [0,2], [3,0], [2,1], [1,2], [0,3]]#, [4,0], [3,1], [2,2], [1,3], [0,4], [5,0], [4,1], [3,2], [2,3], [1,4], [0,5]]
lg_modes = [[0,1], [0,-1], [0,2], [1,0], [0,-2], [0,3], [1,1], [1,-1], [0,-3]]#, [0,4], [1, 2], [1,-2], [2, 0], [0,-4], [0,5], [1,3], [1, -3], [2, 1], [2,-1], [0,-5]]

hgh = np.linspace(0,10, 11, dtype='int')

"""for size in np.linspace(0.3, 0.6, 7):
    for mode in range(len(lg_modes)):
        stage_z.move_to(0.003) # in meter
        stage_z.wait_move()
        stage_z.move_to(0.004) # in meter
        stage_z.wait_move()

        data = np.empty((resolution,480, 640),dtype="uint8")

        print('measuring mode lg:', lg_modes[mode], 'size:', size)
        n = lg_modes[mode][0]
        m = lg_modes[mode][1]

        desired = lg(x, y, n, m, size)
        holo = generate_hologram(desired,input,x,y,195,3,4,0,0,method='bessel1')
        slm.updateArray(holo)
        time.sleep(0.1)


        for i in range(resolution):
            stage_z.move_by(step_size)
            stage_z.wait_move()
            time.sleep(0.1)
            data[i] = camera.capture()

        np.savez(f'Amostras_10_2025\\amostra36_lg{n}{m}_step100e-9_Resolution{resolution}_modesize{size:.2f}', data)
    
print("lg end")

for size in np.linspace(0.3, 0.6, 7):
    for mode in range(len(hg_modes)):
        stage_z.move_to(0.003) # in meter
        stage_z.wait_move()
        stage_z.move_to(0.004) # in meter
        stage_z.wait_move()

        data = np.empty((resolution,480, 640),dtype="uint8")

        print('measuring mode hg:', hg_modes[mode], 'size:', size)
        n = hg_modes[mode][0]
        m = hg_modes[mode][1]

        desired = hg(x, y, n, m, size)
        holo = generate_hologram(desired,input,x,y,195,3,4,0,0,method='bessel1')
        slm.updateArray(holo)
        time.sleep(0.1)


        for i in range(resolution):
            stage_z.move_by(step_size)
            stage_z.wait_move()
            time.sleep(0.1)
            data[i] = camera.capture()

        np.savez(f'Amostras_10_2025\\amostra36_hg{n}{m}_step100e-9_Resolution{resolution}_modesize{size:.2f}', data)

print('hg end')"""

"""size = 0.3
for mode in range(len(hgh)):
    stage_z.move_to(0.003) # in meter
    stage_z.wait_move()
    stage_z.move_to(0.004) # in meter
    stage_z.wait_move()

    data = np.empty((resolution,480, 640),dtype="uint8")

    print('measuring mode hg:', hgh[mode],'0', 'size:', size)
    n = hgh[mode]
    m = 0

    desired = hg(x, y, n, m, size)
    holo = generate_hologram(desired,input,x,y,195,3,4,0,0,method='bessel1')
    slm.updateArray(holo)
    time.sleep(0.1)


    for i in range(resolution):
        stage_z.move_by(step_size)
        stage_z.wait_move()
        time.sleep(0.1)
        data[i] = camera.capture()

    np.savez(f'Amostras_azul\\amostra600_hg{n}{m}_step100e-9_Resolution{resolution}_modesize{size:.2f}', data)

hgh = hgh[1:]

for mode in range(len(hgh)):
    stage_z.move_to(0.003) # in meter
    stage_z.wait_move()
    stage_z.move_to(0.004) # in meter
    stage_z.wait_move()

    data = np.empty((resolution,480, 640),dtype="uint8")

    print('measuring mode hg:', '0',hgh[mode], 'size:', size)
    n = 0
    m = hgh[mode]

    desired = hg(x, y, n, m, size)
    holo = generate_hologram(desired,input,x,y,195,3,4,0,0,method='bessel1')
    slm.updateArray(holo)
    time.sleep(0.1)


    for i in range(resolution):
        stage_z.move_by(step_size)
        stage_z.wait_move()
        time.sleep(0.1)
        data[i] = camera.capture()

    np.savez(f'Amostras_azul\\amostra600_hg{n}{m}_step100e-9_Resolution{resolution}_modesize{size:.2f}', data)"""

stage_z.move_to(0.004) # in meter
stage_z.wait_move()

size = 0.3
data = np.empty((int(2*len(hgh)-1), 480, 640),dtype="uint8")
for mode in range(len(hgh)):

    n = hgh[mode]
    m = 0

    desired = hg(x, y, n, m, size)
    holo = generate_hologram(desired,input,x,y,195,3,4,0,0,method='bessel1')
    slm.updateArray(holo)
    time.sleep(0.1)


    data[mode] = camera.capture()

hgh = hgh[1:]
for mode in range(len(hgh)):

    n = 0
    m = hgh[mode]

    desired = hg(x, y, n, m, size)
    holo = generate_hologram(desired,input,x,y,195,3,4,0,0,method='bessel1')
    slm.updateArray(holo)
    time.sleep(0.1)

    data[mode + len(hgh)+1] = camera.capture()

np.savez(f'Amostras_azul\\amostra220_5cm_modesize{size:.2f}', data)

camera.close()

print('the end')

from slmcontrol import *
import tgsm
import partially_coherent_sources as pcs
from loqufftools.cameras.Ximea import XimeaCamera
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime

slm = SLMdisplay(monitor=2)

config_path = 'Hamamatsu_LCOS-SLM_X10468-07.ini'
config = configparser.ConfigParser()
config.read(config_path)
resX = config['slm'].getint('resX')
resY = config['slm'].getint('resY')

x, y = build_grid(config_path, nmasks=2)

for tau1 in [-1, 0, 1]:
    for tau2 in [1]:  # [-1, 0, 1]:
        w = .7
        q = .8

        n_im = 300

        # vx , vy = np.meshgrid(x)
        # weights = tgsm.weights_lg(w, q, tau, cutoff=1e-2)
        # kernel = tgsm.kernel_lg(x, y, w, q, tau, cutoff=1e-2)

        weights1 = tgsm.weights_lg(w, q, tau1, cutoff=1e-2)
        kernel1 = tgsm.kernel_lg(x, y, w, q, tau1, cutoff=1e-2)

        fields1 = pcs.generate_masks(
            n_im, weights1, kernel1, method='phase_randomized')

        weights2 = tgsm.weights_lg(w, q, tau2, cutoff=1e-2)
        kernel2 = tgsm.kernel_lg(x, y, w, q, tau2, cutoff=1e-2)

        fields2 = pcs.generate_masks(
            n_im, weights2, kernel2, method='phase_randomized')

        a = w*.2
        d = w*.8

        camera = XimeaCamera(1280, 1024)
        camera.set_exposure(50)
        N = fields1.shape[0]
        images = []  # np.zeros((1024,1280))
        slit = double_slit(x, y, a, d)
        try:
            for (field1, field2) in tqdm(zip(fields1, fields2)):
                holo = generate_hologram(
                    [slit*field1, slit*field2], config_path)
                slm.updateArray(holo)
                images.append(camera.capture())
                # images += camera.capture() / N
        except:
            pass
        finally:
            images = np.array(images)
            images_mean = np.mean(images, axis=0)
            print(images_mean.shape)
            camera.close()
            file_path = r'C:\Users\laboq\OneDrive\Área de Trabalho\tgsm- part two\shg\tgsm-infra1-slit-tau' + \
                str(tau1) + '.npz'  # \
            # + str(tau1) + '_tau2_' + str(tau2) + '.npz'
            print(f'Saving data {file_path} ....')
            np.savez(file_path, image=images, images_mean=images_mean)
            print(f'Saved data {file_path} ....')

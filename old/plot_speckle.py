import numpy as np
import matplotlib.pyplot as plt
import cv2

file_path = r"C:\Python\Speckle\Resultados  parciais - VR_NT\Resultados  parciais - VR_NT\hg11 step100e-9.npz"
data = np.load(file_path)['arr_0']

# Remover pixels mortos da câmera
cut = data < 250
data = data * cut

for i in range(20):
    print(np.shape(data[i]))
#plt.imshow(data[0])
#plt.colorbar()
#plt.show()

cv2.imwrite('hg11_step0.png', data[0])

"""im = cv2.imread('hg11_step0.png')

print(np.amin(im), np.amax(im))"""
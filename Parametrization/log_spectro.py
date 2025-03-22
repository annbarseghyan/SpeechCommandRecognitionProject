


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram



def log_specrtro_grayscale_matrix(data):

    f, t, Sxx = spectrogram(data*32768.0, fs=16000)
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)

    
    fig, ax = plt.subplots()
    plt.axis('off')  
    ax.set_position([0, 0, 1, 1]) 
    cax = ax.pcolormesh(t, np.log10(f+1), 20*np.log10(Sxx+1), shading='gouraud', cmap='gray')
    
    
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    grayscale_array = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    
    return grayscale_array
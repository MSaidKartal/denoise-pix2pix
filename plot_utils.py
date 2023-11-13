import numpy as np
from ipywidgets import interact, IntSlider
from matplotlib.patches import PathPatch, Rectangle
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def calc_psvol(high, low):
    psnr = []
    ssim = []
    for i in range(len(high)):
        range = high[i].max()-high[i].min()
        psnr.append(compare_psnr(high[i], low[i], data_range=rangeD))
        ssim.append(compare_ssim(high[i], low[i], data_range=rangeD))
    return psnr, ssim


def show_volume(vol, z, fig_size=(14, 7)):
    #print(vol[0].shape)
    #print(vol[1].shape)
    psnr, ssim = calc_psvol(vol[1], vol[0])
    
    fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=fig_size)
    v_z, v_y, v_x = vol[0].shape

    axarr[0].imshow(vol[0][z, :, :], cmap="gray")
    axarr[0].set_title(f"Low Resolution\nPSNR:{psnr[z]:.3f} / SSIM:{ssim[z]:.3f}", fontweight='bold', fontsize=20)
    axarr[0].axis('off')

    axarr[1].imshow(vol[1][z, :, :], cmap="gray")
    axarr[1].set_title(f"High Resolution", fontweight='bold', fontsize=20)
    axarr[1].axis('off')

    fig.tight_layout()


def interactive_show(vol):
    volume= vol[0]
    vol_shape = volume.shape
    interact(
        lambda z: plt.show(show_volume(vol, z)),
        z=IntSlider(min=0, max=vol_shape[0], step=1, value=int(vol_shape[0] / 2)),

    )


def show_3volume(vol, z, fig_size=(21, 7)):
    #print(vol[0].shape)
    #print(vol[1].shape)

    fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=fig_size)
    v_z, v_y, v_x, n = vol[0].shape

    axarr[0].imshow(vol[0][z, :, :, 0], cmap="gray")
    axarr[0].set_title('Input (Low Resolution)', fontweight='bold', fontsize=18)
    axarr[0].axis('off')

    axarr[1].imshow(vol[1][z, :, :, 0], cmap="gray")
    axarr[1].set_title('Target (High Resolution)', fontweight='bold', fontsize=20)
    axarr[1].axis('off')

    axarr[2].imshow(vol[2][z, :, :, 0], cmap="gray")
    axarr[2].set_title('Generated Image', fontweight='bold', fontsize=20)
    axarr[2].axis('off')

    fig.tight_layout()


def interactive_inference(vol):
    volume= vol[0]
    vol_shape = volume.shape
    interact(
        lambda z: plt.show(show_3volume(vol, z)),
        z=IntSlider(min=0, max=vol_shape[0], step=1, value=int(vol_shape[0] / 2)),

    )

def mae_calc(y_true, predictions):
    y_true, predictions = np.array(y_true), np.array(predictions)
    return np.mean(np.abs(y_true - predictions))

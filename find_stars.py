import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from astropy.io import fits
from astropy.visualization import ZScaleInterval


def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function
    
    Parameters:
    -----------
    coords : tuple
        (x, y) coordinates where x and y are meshgrid arrays
    amplitude : float
        Height of the gaussian
    x0, y0 : float
        Center position of the gaussian
    sigma_x, sigma_y : float
        Width of the gaussian in x and y directions
    theta : float
        Rotation angle in radians
    offset : float
        Baseline offset
    
    Returns:
    --------
    z : ndarray
        2D Gaussian evaluated at x, y points
    """
    x, y = coords
    
    # Rotation
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    # Gaussian function
    z = offset + amplitude * np.exp(
        -(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2))
    )
    
    return z.ravel()

def fit_gaussian_2d(image):
    """
    Fit a 2D Gaussian to the input image
    
    Parameters:
    -----------
    image : ndarray
        Input image
    
    Returns:
    --------
    popt : ndarray
        Optimal parameters (amplitude, x0, y0, sigma_x, sigma_y, theta, offset)
    pcov : ndarray
        Covariance matrix for the parameters
    """
    # Create x and y indices
    y, x = np.indices(image.shape)
    
    # Initial guess for parameters
    height = np.max(image) - np.min(image)
    offset = np.min(image)
    
    # Find the peak position
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    
    # Initial guess for width
    sigma_x = sigma_y = np.sqrt(np.sum((image - offset) * ((x - x_max)**2 + (y - y_max)**2)) / np.sum(image - offset))
    
    # Initial parameters
    initial_guess = [height, x_max, y_max, sigma_x, sigma_y, 0, offset]
    
    # Bounds for the parameters
    # (amplitude, x0, y0, sigma_x, sigma_y, theta, offset)
    lower_bounds = [0, 0, 0, 0, 0, -np.pi/2, 0]
    upper_bounds = [np.inf, image.shape[1], image.shape[0], image.shape[1], image.shape[0], np.pi/2, np.inf]
    bounds = (lower_bounds, upper_bounds)
    
    # Fit the data
    popt, pcov = curve_fit(
        gaussian_2d, 
        (x, y), 
        image.ravel(), 
        p0=initial_guess,
        bounds=bounds
    )
    
    return popt, pcov



def get_noise_level(img):


    return np.nanstd(img)



def get_star_locs(fname, sigma=10, return_image=False):
    starttime = time.time()

    img = fits.getdata(fname)

    scaler = ZScaleInterval()
    limits = scaler.get_limits(img)

    noise = get_noise_level(img-np.nanmedian(img))

    masked_img = (img-np.nanmedian(img)).copy()
    masked_img[masked_img < sigma*noise] = 0
    masked_img[masked_img > 0] = 1

#    ax[0].imshow(img, cmap='Greys_r', vmin=limits[0], vmax=50000, origin='lower')
    labels_img = cv.threshold(masked_img, 0, 1, cv.THRESH_BINARY)[1]
    num_labels, labels_img = cv.connectedComponents(masked_img.astype(np.int8), connectivity=4)
#    ax[1].imshow(labels_img)


    stars = {}
    padding = 1

    # go through each of the new labels
    for ii in range(1, num_labels):
        xinds, yinds = np.where(labels_img == ii)

        if len(xinds) <= 8:
            continue

        left = np.min(xinds)
        bottom = np.min(yinds)
        width = np.max(xinds)-left+1
        height = np.max(yinds)-bottom+1


        cutout = img[left:left+width, bottom:bottom+height]

        if np.any(cutout >= 2**16-1):
            continue
        
        stars[ii] = (left-padding,
                     bottom-padding,
                     width+2*padding,
                     height+2*padding)


    num_stars = len(stars)

#    fig2, axs = plt.subplots(int(np.sqrt(num_stars)), 
#                            int(num_stars/int(np.sqrt(num_stars)))+1)


    finalxs = []
    finalys = []

    for inum, ii in enumerate(stars.keys()):
        left   = stars[ii][0]
        bottom = stars[ii][1]
        width  = stars[ii][2]
        height = stars[ii][3]


        cutout = img[left:left+width, bottom:bottom+height]

        popt, pcov = fit_gaussian_2d(cutout)
        x = popt[1]
        y = popt[2]
#        axs.flatten()[inum].imshow(cutout)
#        ax[0].scatter(bottom+x, left+y, color='red',marker='+', alpha=0.9)
        finalxs.append(float(bottom+x))
        finalys.append(float(left+y))

    print(f"Found {len(finalxs)} stars in {(time.time()-starttime):.2f}sec for {fname}")

    if return_image:
        return np.array(finalxs), np.array(finalys), img
    else:
        return np.array(finalxs), np.array(finalys), None




if __name__ == "__main__":
    xs, ys = get_star_locs('./test_data/0001.fit')
    xs2, ys2 = get_star_locs('./test_data/0015.fit')

    print(len(xs), len(xs2))
    plt.show()

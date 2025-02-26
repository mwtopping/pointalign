import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from tqdm import tqdm
import itertools
import numpy as np
import cv2 as cv

from find_stars import *

from astropy.visualization import ZScaleInterval




def transform_points(rot, shift, points, recenter=False):

    if recenter:
        mean = np.mean(points, axis=0)
        points -= mean

    ones = np.ones(np.shape(points)[0])
    points = np.column_stack((points, ones))

    rotmat = np.matrix([[np.cos(rot),-1*np.sin(rot), shift[0]],
                        [np.sin(rot),np.cos(rot), shift[1]],
                        [0,0,1]])
    outpoints = (rotmat*points.T).T[:,:2]
    if recenter:
        outpoints += mean
    return outpoints 


# generate some random data
#xs = np.random.uniform(0, 1, size=nstars)
#ys = np.random.uniform(0, 1, size=nstars)


#shuffle_inds = np.random.choice(range(nstars), size=nstars, replace=False)

#xs_shift = xs[shuffle_inds] + 0.05
#ys_shift = ys[shuffle_inds] + 0.08

#temp_points = np.column_stack((xs[shuffle_inds].copy(), ys[shuffle_inds].copy()))
#shifted_points = np.array(transform_points(0.1, [0.00, 0.00], temp_points, recenter=True))
#xs_shift = shifted_points[:,0]
#ys_shift = shifted_points[:,1]

def get_angles(xs, ys):

    a = np.sqrt((xs[0]-xs[1])**2 + (ys[0]-ys[1])**2)
    b = np.sqrt((xs[1]-xs[2])**2 + (ys[1]-ys[2])**2)
    c = np.sqrt((xs[0]-xs[2])**2 + (ys[0]-ys[2])**2)
    
    a1 = np.arccos((a*a + c*c - b*b) / (2*a*c))#opposite point 0
    a2 = np.arccos((a*a + b*b - c*c) / (2*a*b))#opposite point 1
    a3 = np.arccos((b*b + c*c - a*a) / (2*b*c))#opposite point 2

    angles = np.array([a1, a2, a3])
    sorted_inds = np.argsort(angles)
    sorted_angles = angles[sorted_inds]
    sorted_pos = [xs[sorted_inds], ys[sorted_inds]]

    return sorted_angles, sorted_pos


def get_all_tris(ids):
    perms = list(itertools.combinations(ids, 3))
    return perms

def match_angles(angles1, angles2):
    # get first two angles of each list
    #  angles are sorted, so this is smallest 2 angles
    angles1 = np.array(angles1)[:,:2]
    angles2 = np.array(angles2)[:,:2]

    tree_angles1 = KDTree(angles1)
    tree_angles2 = KDTree(angles2)

    dists, inds = tree_angles2.query(angles1, k=1)

    return dists, inds

    
def get_all_angles(xs, ys):
    allperms = set()
    N=4
    for ii in tqdm(range(len(xs))):
        dists = (xs-xs[ii])**2 + (ys-ys[ii])**2
        closest = np.argsort(dists).tolist()[:N+1]

        perms = [tuple(sorted(p)) for p in get_all_tris(closest)]
        for p in perms:
            allperms.add(p)

    allperms = list(allperms)
    angles = []
    pos = []
    for p in allperms:
        sorted_angles, sorted_pos = get_angles(xs[list(p)], ys[list(p)])
        angles.append(sorted_angles)
        pos.append(sorted_pos)

    return angles, pos



def get_frame_transform(ref_fname, target_fname):
    xs, ys, img1 = get_star_locs(target_fname, return_image=True)
    xs_shift, ys_shift, img2 = get_star_locs(ref_fname, return_image=True)


    angles, pos = get_all_angles(xs, ys)
    angles_shift, pos_shift = get_all_angles(xs_shift, ys_shift)

    dists, inds = match_angles(angles, angles_shift)
    weights = np.pow(np.clip(1-dists*10, 0, 1), 4)

    all_refs = []
    all_targs = []


    for ii, ind in enumerate(inds):
        w = weights[ii][0]
        ind = ind[0]
        p = pos[ii]
        ps = pos_shift[ind]

        ref_point  = [p[0][0] , p[1][0]]
        targ_point = [ps[0][0], ps[1][0]]

        if ref_point not in all_refs:
            all_refs.append(ref_point)
            all_targs.append(targ_point)




    H, inpts = cv.estimateAffinePartial2D(np.array(all_refs),
                                          np.array(all_targs),
                                          ransacReprojThreshold=3.0)


    ones = np.ones(np.shape(all_refs)[0])
    points = np.column_stack((all_refs, ones))



    newpoints = np.matrix(points) * H.T
#    [ax.scatter(p[0], p[1], color='black', linewidths=2, facecolor='none') for p in np.array(newpoints)]
    print(H)
    print("Number of inliers: ", len(inpts))
    print(len(newpoints))





    scaler = ZScaleInterval()
    finalimg = img1+img2
    limits = scaler.get_limits(finalimg)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    axs[0].imshow(finalimg, vmin=limits[0], vmax=limits[1], cmap='Greys_r', origin='lower')

    outshape = (np.shape(img2)[1], np.shape(img2)[0])

    transformed1 = cv.warpAffine(img1, H, outshape)


    axs[1].imshow(transformed1+img2, vmin=limits[0], vmax=limits[1], cmap='Greys_r', origin='lower')



if __name__ == "__main__":
    ref_fname = "./test_data/0001.fit"
    target_fname = "./test_data/0015.fit"
    get_frame_transform(ref_fname, target_fname)

    plt.show()
    

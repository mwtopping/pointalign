import numpy as np
from scipy.optimize import minimize, basinhopping, dual_annealing
import matplotlib.pyplot as plt
import cv2 as cv



def get_transformation_matrix(rot, shift):

    mat = [[ 0 + shift[0], 0],
           [ 0, 0 + shift[1]]]

    return np.matrix(mat)

def get_W(inpts, targs):
    return np.matrix(inpts).T * np.matrix(targs)


def demoment(points):
    mean = np.zeros(np.shape(points)[1])
    
    Npoints = np.shape(points)[0]
    Ndim = np.shape(points)[1]
    for p in points:
        mean += np.array(p).squeeze()

    return (mean/Npoints).squeeze()

def generate_random_data(N):
    xs = np.random.uniform(size=N)
    ys = np.random.uniform(size=N)

    points = np.array([(x, y, 1) for x, y in zip(xs, ys)])

    return np.matrix(points)

def get_com(points):
    pass

def transform_points(rot, shift, points):
    rotmat = np.matrix([[np.cos(rot),-1*np.sin(rot), shift[0]],
                        [np.sin(rot),np.cos(rot), shift[1]],
                        [0,0,1]])

#    for p in points:
#
#        print((rotmat * p.T).T)
#        print(np.shape(rotmat * p.T))

    #return np.array([np.array((rotmat * point.T).T + shift) for point in points]).squeeze()
    return (rotmat*points.T).T
#    return np.array([(rotmat * point.T).T + shift for point in points])


def e(points, targets, S):
    E = 0
    for p, t in zip(points, targets):
        E += np.sum(p*p)
        E += np.sum(t*t)

    E += 2*np.sum(S)

    return E


def get_error(points, targets, inds):




    dist = 0

    for p, t in zip(points[inds], targets):
        p = np.array(p)[0]
        dist += np.sum((p-t)**2)
    return dist


def total_error(points, targets, inds, sx, sy, rot):
    shift = [sx, sy]

    test = transform_points(rot, shift, targets.copy())

    return get_error(points, test, inds)




    
if __name__ == "__main__":

    points = generate_random_data(20)
    orig_points = points.copy()
    shift = [.2, .2]
    rot = 0.0
    targets = transform_points(rot, shift, orig_points)

    print(np.shape(points), np.shape(targets))


    fig, ax = plt.subplots()
#    _= [ax.scatter(np.array(p)[0], np.array(p)[1], color='black') for p in np.array(points)]
#    _= [ax.scatter(np.array(p)[0], np.array(p)[1], color='blue') for p in np.array(targets)]
 

    for ii in range(1):

        inds = []

        for t in targets[:,:2]:
            dists = np.sum(np.array(points[:,:2]-t)**2, axis=1)
            closest = np.argmin(dists)
            t = np.array(t)
            p = np.array(points[closest])
#            ax.plot([t[0][0], p[0][0]], 
#                       [t[0][1], p[0][1]])
            inds.append(int(closest))



        
        normed_points = points[inds] - demoment(points[inds]) + demoment(points[inds])

        print(np.array(points[:,:2]))
        print(np.array(targets[:,:2]))

        H, inpts = cv.estimateAffinePartial2D(np.array(targets[:,:2]),
                                              np.array(normed_points[:,:2]))


        print(H)
        newpoints = targets * H.T

#        newpoints = transform_points(res[2], [res[0], res[1]], targets.copy())



   #
#        _= [ax.scatter(np.array(p)[0], np.array(p)[1], color='green', marker='o', s=2) for p in np.array(normed_points)]
    #    E = calc_error([np.array(p-pmean) for p in points],
    #               [np.array(t-tmean) for t in targets], 
    #               S)
    #
        _= [ax.scatter(np.array(p)[0], np.array(p)[1], color='blue', marker='o', s=12) for p in np.array(normed_points)]
        _= [ax.scatter(np.array(p)[0], np.array(p)[1], color='red', marker='s', s=12) for p in np.array(targets)]



        targets = newpoints.copy()
        ones = np.ones(np.shape(targets)[0])
        targets = np.column_stack((targets, ones))

#    _= [ax.scatter(np.array(p)[0], np.array(p)[1], color='orange', marker='*', s=3) for p in np.array(newpoints)]

#        _= [ax.scatter(np.array(p)[0], np.array(p)[1], color='blue', marker='o', s=12) for p in np.array(normed_points)]
#        _= [ax.scatter(np.array(p)[0], np.array(p)[1], color='red', marker='s', s=12) for p in np.array(normed_targets)]
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])

    plt.show()


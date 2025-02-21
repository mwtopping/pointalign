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

    points = np.array([(x, y) for x, y in zip(xs, ys)])

    return np.matrix(points)

def get_com(points):
    pass

def transform_points(rot, shift, points):
    rotmat = np.matrix([[np.cos(rot),-1*np.sin(rot)],
                        [np.sin(rot),np.cos(rot)]])

#    for p in points:
#
#        print((rotmat * p.T).T)
#        print(np.shape(rotmat * p.T))

    #return np.array([np.array((rotmat * point.T).T + shift) for point in points]).squeeze()
    return np.array([(rotmat * point.T).T + shift for point in points])


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

    points = generate_random_data(30)
    shift = [1, 1]
    rot = 0.3
    targets = transform_points(rot, shift, points.copy())

    fig, ax = plt.subplots()
    _= [ax.scatter(np.array(p)[0], np.array(p)[1], color='black') for p in np.array(points)]
    _= [ax.scatter(np.array(p)[:,0], np.array(p)[:,1], color='blue') for p in np.array(targets)]
 

    for ii in range(10):

        inds = []

        for t in targets:
            dists = np.sum(np.array(points-t)**2, axis=1)
            closest = np.argmin(dists)

            inds.append(int(closest))


#        def mine(params):
#            return total_error(points, targets, inds, params[0], params[1], params[2])


        #result = minimize(mine, x0=[0,0,0])

#        result =dual_annealing(mine, [[-2, 2], [-2, 2], [0, 2*3.14]])
#        res = result.x
#
#        print("Found best:")
#        print("Rotation:", res[2])
#        print("Shiftx:", res[0])
#        print("Shifty:", res[1])
#
#        print("Error: ", total_error(points, targets, inds, *res))

        H, inpts = cv.estimateAffine2D(targets, points)
        print("H:", H)
        newpoints = targets.reshape(2, 30).copy() * H

#        newpoints = transform_points(res[2], [res[0], res[1]], targets.copy())



   #
        _= [ax.scatter(np.array(p)[:,0], np.array(p)[:,1], color='red', marker='o', s=1) for p in np.array(newpoints)]
    #    E = calc_error([np.array(p-pmean) for p in points],
    #               [np.array(t-tmean) for t in targets], 
    #               S)
    #
        targets = newpoints.copy()

    _= [ax.scatter(np.array(p)[:,0], np.array(p)[:,1], color='orange', marker='o', s=3) for p in np.array(newpoints)]
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])

    plt.show()


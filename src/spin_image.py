import numpy as np
import pylab as plt
import cv2
import time
from sklearn.neighbors import KDTree
import progressbar

class PointClass(object):
    '''
    Class for a point with coordinates x, y, z
    '''
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def filter_by_cube(arr, cube_origin, cube_size):
    '''
    Inputs:
    arr - (nx3 numpy array)
    cube_origin - point class
    cube_size - (3, numpy array)

    Returns:
    Numpy array where all points in arr that lay in cube with origin
    cube_origin and size cube_size
    '''

    filtered_arr = arr[(arr[:,0] > (cube_origin.x-(cube_size[0]/2))) & (arr[:,0] < (cube_origin.x+(cube_size[0]/2))) & \
                       (arr[:,1] > (cube_origin.y-(cube_size[1]/2))) & (arr[:,1] < (cube_origin.y+(cube_size[1]/2))) & \
                       (arr[:,2] > (cube_origin.z-(cube_size[2]/2))) & (arr[:,2] < (cube_origin.z+(cube_size[2]/2)))]
    return filtered_arr

def kd_tree(p_ind, p_cloud):
    tree = KDTree(p_cloud, leaf_size=400)
    dist, ind = tree.query(p_cloud[p_ind].reshape(1,-1), k=5000)
    return ind[0]


def cylinder_test(cyl_pt1, cyl_pt2, pt, radius_sq):
    '''
    Function to check if point (pt) lies in a given cylinder where
    cyl_pt1 and cyl_pt2 are the center of the cylinder caps.

    Inputs:
    cyl_pt1: PointClass
    cyl_pt2: PointClass
    pt: PointClass
    radius_sq: Int/Long

    Returns:
    True if point lays in cylinder
    False otherwise
    '''

    dx = cyl_pt2.x - cyl_pt1.x
    dy = cyl_pt2.y - cyl_pt1.x
    dz = cyl_pt2.z - cyl_pt1.x

    lengthsq = (dx*dx) + (dy*dy) + (dz*dz)
    pdx = pt.x - cyl_pt1.x
    pdy = pt.y - cyl_pt1.y
    pdz = pt.z - cyl_pt1.z

    dot = (pdx*pdx) + (pdy*dy) + (pdz*dz)

    if dot < 0 or dot > lengthsq:
        return False
    else:
        dsq = (pdx*pdx + pdy*pdy + pdz*pdz) - dot*dot/lengthsq
        if dsq > radius_sq:
            return False
        else:
            return True

def euclidean_dist(pt1, pt2, dim=3):
    '''
    Calculates the euclidean distance between two points in
    either 2-D or 3-D depending on 'dim' variable

    Inputs:
    pt1: PointClass
    pt2: PointClass
    dim: Int

    Returns:
    distance: Long
    '''
    if dim == 2:
        return np.sqrt((pt2.x-pt1.x)**2+(pt2.y-pt1.y)**2)
    elif dim == 3:
        return np.sqrt((pt2.x-pt1.x)**2+(pt2.y-pt1.y)**2+(pt2.z-pt1.z)**2)
    else:
        raise ValueError("Dimensions must be either 2 or 3")

def generate_random_data(max_val, num_points):

    return np.random.randint(max_val, size=(num_points,3))

def mask_points(p, p_cloud, radius):

    cube_size = np.array([radius, radius, radius])

    valid_points = []

    filtered_pts = filter_by_cube(arr=p_cloud, cube_origin=p, cube_size=cube_size)

    #cyl_pt1 = PointClass(p.x, p.y, p.z-(radius/2))
    #cyl_pt2 = PointClass(p.x, p.y, p.z+(radius/2))

    #for f_pt in filtered_pts:
    #    test_pt = PointClass(x=f_pt[0], y=f_pt[1], z=f_pt[2])
    #    if cylinder_test(cyl_pt1=cyl_pt1, cyl_pt2=cyl_pt2, pt=test_pt, radius_sq=radius**2):
    #    valid_points.append(test_pt)
    #    else:
    #    pass

    return filtered_pts

def si_position(p, x, n):

    p_vec = [p.x, p.y, p.z]
    x_vec = [x.x, x.y, x.z]
    n_vec = [n[0], n[1], n[2]]

    xp_vec = [x.x-p.x, x.y-p.y, x.z-p.z]

    a = np.sqrt( (euclidean_dist(p, x, dim=3)**2) - ((np.dot(n_vec, xp_vec))**2) )
    b = np.dot(n_vec, xp_vec)

    return a, b

def get_bin(a, b, bin_size, b_max):

    i = np.floor((b_max - b)/bin_size)
    j = np.floor(a/bin_size)

    return i, j

def spin_pnt(p, p_ind, p_cloud, radius, bin_size, bilinear_interp):

    b_max = 0.5*radius

    n = [0,0,1]

    #tic = time.time()

    si = np.zeros((int(radius/bin_size)+1, int(radius/bin_size)+1), dtype=np.uint8)
    valid_points = mask_points(p, p_cloud, radius)
    #valid_points = p_cloud[kd_tree(p_ind, p_cloud)]

    for x_pnt in valid_points:
        x_pnt = PointClass(x=x_pnt[0], y=x_pnt[1],z=x_pnt[2])
        a, b = si_position(p, x_pnt, n)
        i, j = get_bin(a, b, bin_size, b_max)
        if bilinear_interp == True:
            k = a-(i*bin_size)
            l = b-(j*bin_size)
            si[int(i),int(j)] = si[int(i),int(j)] + ((1-k)*(1-l))
            si[int(i),int(j+1)] = si[int(i),int(j+1)] + (k*(1-l))
            si[int(i+1),int(j)] = si[int(i+1),int(j)] + ((1-k)*l)
            si[int(i+1),int(j+1)] = si[int(i+1),int(j+1)] + (k*l)
        else:
            si[int(i),int(j)] = si[int(j),int(i)] + 1

    #print time.time()-tic
    return si



def compute_spin_images(p_cloud, radius, bin_size, bilinear_interp):

    print '[INFO] Generating spin images...'
    bar = progressbar.ProgressBar(maxval=len(p_cloud), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    b_max = 0.5*radius

    spin_images = []

    n = [0,0,1]

    for i, pnt in enumerate(p_cloud):
        bar.update(i+1)
        si = np.zeros((int(radius/bin_size)+1, int(radius/bin_size)+1), dtype=np.uint8)
        p = PointClass(x=pnt[0],y=pnt[1],z=pnt[2])
        valid_points = mask_points(p, p_cloud, radius)

        for x_pnt in valid_points:
            a, b = si_position(p, x_pnt, n)
            i, j = get_bin(a, b, bin_size, b_max)
            if bilinear_interp == True:
                k = a-(i*bin_size)
                l = b-(j*bin_size)
                si[int(i),int(j)] = si[int(i),int(j)] + ((1-k)*(1-l))
                si[int(i+1),int(j)] = si[int(i+1),int(j)] + (k*(1-l))
                si[int(i),int(j+1)] = si[int(i),int(j+1)] + ((1-k)*l)
                si[int(i+1),int(j+1)] = si[int(i+1),int(j+1)] + (k*l)
            else:
                si[int(i),int(j)] = si[int(i),int(j)] + 1
        spin_images.append(si)
        plt.figure(), plt.imshow(si), plt.show()
    bar.finish()
    return spin_images

def main():

    p_cloud = generate_random_data(max_val=1000, num_points=10000000)

    start = time.time()
    spin_images = compute_spin_images(p_cloud=p_cloud, radius=448, bin_size=2, bilinear_interp=True)
    print '[INFO] Processed', len(spin_images), 'in', (time.time()-start)/60,'minutes'

if __name__ == '__main__':
    main()

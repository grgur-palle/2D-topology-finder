"""
This script is part of the Supplementary Material of the article:
Grgur Palle, Owen Benton: 'Exactly solvable spin-1/2 XYZ models
with highly-degenerate, partially ordered, ground states'
https://arxiv.org/abs/2101.12140

The main function that is defined in this script is the
function 'triangulation_finder'. It is documented in detail
below and explained in the Supplementary Material from above.

The conventions used in specifying the mesh and triangulations
are given in the string 'mesh_convention'.

The utility functions 'refine_mesh' and 'relax_mesh' may also
be of use when refining and relaxing meshes of 2D manifolds.

The function 'create_sphere' is included for a demonstration
of the utility functions. See the jupyter notebook
'triangulation_demonstration_and_verification.ipynb'
for examples of applications of 'triangulation_finder'.
"""

import numpy as np
from scipy.linalg import svd

# for constrained Delaunay triangulation
import triangle as tr # https://rufat.be/triangle

# for the calculation of connected components of the manifold boundary
import networkx as nx # https://networkx.github.io

# for plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

##################################################

mesh_convention = """
CONVENTION
vertices = [[x1, x2, x3, ..., xd],
            [x1, x2, x3, ..., xd],
            [x1, x2, x3, ..., xd],
            [x1, x2, x3, ..., xd], ... ]
vertices.dtype = float

vertices are assumed to be embedded in
a d-dimensional Euclidean space

a ---- b
edges = [[vert_a1, vert_b1],
         [vert_a2, vert_b2],
         [vert_a3, vert_b3], ... ]
with the convention vert_a < vert_b
edges.dtype = int

by vert_a1, vert_b2, etc., we mean the corresponding
INDEX of the vertices array, not the coordinates

        c
       /\\
      /  \\
     /____\\
    a      b
triangles = [[vert_a1, vert_b1, vert_c1, edg_ab1, edg_bc1, edg_ca1],
             [vert_a2, vert_b2, vert_c2, edg_ab2, edg_bc2, edg_ca2],
             [vert_a3, vert_b3, vert_c3, edg_ab3, edg_bc3, edg_ca3], ... ]
with the convention that vert_a < vert_b and vert_a < vert_c
triangles.dtype = int

by vert_a1, vert_b2, etc., we mean the corresponding
INDEX of the vertices array, not the coordinates

by edg_ab1, edg_ca3, etc., we mean the corresponding
INDEX of the edges array, not the index doublet (vert_a, vert_b)

the ordering vert_a -> vert_b -> vert_c -> vert_a
determines the orientation of the triangle
"""

def refine_mesh(vertices, edges, triangles):
    """
    Creates a finer mesh by subdividing all triangles according to:
               c                           c      
              /\\                          /\\      
             /  \\                        /  \\     
            /    \\       ----->       f /____\\ e  
           /      \\                    /\\    /\\   
          /        \\                  /  \\  /  \\  
         /__________\\                /____\\/____\\ 
        a            b              a     d      b
    
    The generated mesh preserves orientations and smallest index conventions,
    that is, it follows 'mesh_convention'.
    
    INPUT: (vertices, edges, triangles)
    (vertices, edges, triangles) should follow the 'mesh_convention'
    DOES NOT MODIFY INPUT
    
    OUTPUT: (new_vertices, new_edges, new_triangles)
    new_vertices.shape = (vertices.shape[0] + edges.shape[0], vertices.shape[1])
    new_vertices.dtype = float
    new_edges.shape = (2 * edges.shape[0] + 3 * triangles.shape[0], 2)
    new_edges.dtype = int
    new_triangles.shape = (4 * triangles.shape[0], 3)
    new_triangles.dtype = int
    (new_vertices, new_edges, new_triangles) follows the 'mesh_convention'
    """
    
    n_vert = vertices[:,0].size
    n_edg = edges[:,0].size
    n_triag = triangles[:,0].size
    
    # |_____|_____|
    # a     d     b
    a_ind = edges[:,0]
    b_ind = edges[:,1]
    new_vert = 0.5 * (vertices[a_ind] + vertices[b_ind])
    
    new_vertices = np.concatenate((vertices, new_vert), axis=0)
    new_vert_ind = n_vert + np.arange(n_edg)
    
    new_edg_ad = np.stack((a_ind, new_vert_ind), axis=1)
    new_edg_bd = np.stack((b_ind, new_vert_ind), axis=1)
    
    #        c
    #       /\
    #      /  \
    #   f /____\ e
    #    /\    /\
    #   /  \  /  \
    #  /____\/____\
    # a     d      b
    a_ind = triangles[:,0]
    b_ind = triangles[:,1]
    c_ind = triangles[:,2]
    d_ind = new_vert_ind[triangles[:,3]]
    e_ind = new_vert_ind[triangles[:,4]]
    f_ind = new_vert_ind[triangles[:,5]]
    new_edg_de = np.stack((d_ind, e_ind), axis=1)
    new_edg_ef = np.stack((e_ind, f_ind), axis=1)
    new_edg_fd = np.stack((f_ind, d_ind), axis=1)
    
    new_edges = np.concatenate((new_edg_ad, new_edg_bd, new_edg_de, new_edg_ef, new_edg_fd), axis=0)
    swap = new_edges[:,0] > new_edges[:,1] # ensures the convention vert_a < vert_b
    new_edges[swap,0], new_edges[swap,1] = new_edges[swap,1], new_edges[swap,0]
    
    new_edg_ad_ind = np.arange(n_edg)
    new_edg_bd_ind = n_edg + np.arange(n_edg)
    new_edg_de_ind = 2*n_edg + np.arange(n_triag)
    new_edg_ef_ind = 2*n_edg + n_triag + np.arange(n_triag)
    new_edg_fd_ind = 2*n_edg + 2*n_triag + np.arange(n_triag)
    
    ab_ind = triangles[:,3]
    bc_ind = triangles[:,4]
    ca_ind = triangles[:,5]
    
    # attention has to be paid to the proper ordering of the triangle vertices so that the orientation is preserved
    new_triag_adf = np.stack((
        a_ind, d_ind, f_ind,
        np.where(new_edges[new_edg_ad_ind[ab_ind],0] == a_ind, new_edg_ad_ind[ab_ind], new_edg_bd_ind[ab_ind]),
        new_edg_fd_ind,
        np.where(new_edges[new_edg_ad_ind[ca_ind],0] == a_ind, new_edg_ad_ind[ca_ind], new_edg_bd_ind[ca_ind])
        ), axis=1)
    
    new_triag_bed = np.stack((
        b_ind, e_ind, d_ind,
        np.where(new_edges[new_edg_ad_ind[bc_ind],0] == b_ind, new_edg_ad_ind[bc_ind], new_edg_bd_ind[bc_ind]),
        new_edg_de_ind,
        np.where(new_edges[new_edg_ad_ind[ab_ind],0] == b_ind, new_edg_ad_ind[ab_ind], new_edg_bd_ind[ab_ind])
        ), axis=1)
    
    new_triag_cfe = np.stack((
        c_ind, f_ind, e_ind,
        np.where(new_edges[new_edg_ad_ind[ca_ind],0] == c_ind, new_edg_ad_ind[ca_ind], new_edg_bd_ind[ca_ind]),
        new_edg_ef_ind,
        np.where(new_edges[new_edg_ad_ind[bc_ind],0] == c_ind, new_edg_ad_ind[bc_ind], new_edg_bd_ind[bc_ind])
        ), axis=1)
    
    new_triag_def = np.stack((d_ind, e_ind, f_ind, new_edg_de_ind, new_edg_ef_ind, new_edg_fd_ind), axis=1)
    
    new_triangles = np.concatenate((new_triag_adf, new_triag_bed, new_triag_cfe, new_triag_def), axis=0)
    
    # ensures the convention vert_a is smallest
    cycle = (new_triangles[:,1] < new_triangles[:,0]) & (new_triangles[:,1] < new_triangles[:,2])
    new_triangles[cycle,0:3] = np.roll(new_triangles[cycle,0:3], -1, axis=1)
    new_triangles[cycle,3:6] = np.roll(new_triangles[cycle,3:6], -1, axis=1)
    
    cycle = (new_triangles[:,2] < new_triangles[:,0]) & (new_triangles[:,2] < new_triangles[:,1])
    new_triangles[cycle,0:3] = np.roll(new_triangles[cycle,0:3], +1, axis=1)
    new_triangles[cycle,3:6] = np.roll(new_triangles[cycle,3:6], +1, axis=1)
    
    return new_vertices, new_edges, new_triangles

def relax_mesh(arr, ensure_arr_within_manifold, calc_arr_forces,
               min_step_size=1e-5, max_step_number=1000, min_avg_force=1e-6,
               print_all=False):
    """
    Relaxes the manifold specified through 'ensure_arr_within_manifold'
    by applying the forces from 'calc_arr_forces' on the mesh. The
    relaxation iterations (evolution of the system under the forces)
    end when one of the three conditions determined by 'min_step_size',
    'max_step_number', or 'min_avg_force' are violated.
    
    'ensure_arr_within_manifold' is a function that receives an array
    arr = [[x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd], ... ]
    and returns an array 'projected_arr' of the same shape whose i-th
    component is a point on the manifold that is nearest to arr[i]
    (i.e., a projection of arr[i] on the manifold).
    projected_arr.shape = arr.shape
    projected_arr.dtype = float
    'ensure_arr_within_manifold SHOULD NOT MODIFY THE RECEIVED 'arr'!
    
    'calc_arr_forces' is a function that receives an array
    arr = [[x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd], ... ]
    and returns an array 'forces_arr' of the same shape whose i-th component
    is the force felt by arr[i]. The forces should be repulsive, short-ranged,
    and be normalized so that they span the interval [0, 1]. It is also a
    good idea to make these forces orthogonal to the manifold constraints.
    forces_arr.shape = arr.shape
    forces_arr.dtype = float
    'calc_arr_forces' SHOULD NOT MODIFY THE RECEIVED 'arr'!
    
    INPUT: (arr, ensure_arr_within_manifold, calc_arr_forces,
            min_step_size=1e-5, max_step_number=1000, min_avg_force=1e-6, print_all=False)
    arr = [[x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd], ... ]
    arr.dtype = float
    ensure_arr_within_manifold : arr -> projected_arr
    calc_arr_forces : arr -> forces_arr
    type(min_step_size) = float, should be positive and small
    type(max_step_number) = int, should be positive and large
    type(min_avg_force) = float, should be positive and small
    type(print_all) = bool, should iteration messages be printed?
    DOES NOT MODIFY INPUT
    
    OUTPUT: relaxed_arr
    relaxed_arr.shape = arr.shape
    relaxed_arr.dtype = float
    """
    
    n = 0
    step_size = 0.25
    relaxed_arr = ensure_arr_within_manifold(arr)
    curr_F = calc_arr_forces(relaxed_arr)
    curr_avg_F = np.mean(np.sqrt(np.sum(curr_F*curr_F, axis=1)))
    
    print("Starting manifold mesh relaxation.\n")
    
    while step_size > min_step_size and n < max_step_number and curr_avg_F > min_avg_force:
        new_arr = ensure_arr_within_manifold(relaxed_arr + step_size * curr_F)
        new_F = calc_arr_forces(new_arr)
        new_avg_F = np.mean(np.sqrt(np.sum(new_F*new_F, axis=1)))
        
        if new_avg_F < curr_avg_F:
            relaxed_arr = new_arr
            curr_F, curr_avg_F = new_F, new_avg_F
        else:
            step_size *= 0.5
        n += 1
        
        if print_all:
            print("step:{:3d}; step size: {:.5e}; tolerance: {:.12e}.".format(n, step_size, curr_avg_F))
    if print_all:
        print()
    
    print("The manifold mesh has been successfully relaxed after {:d} steps.\n".format(n))
    
    return relaxed_arr

##################################################

def _normalize(arr):
    """
    Normalizes the 2D array 'arr' in place
    so that 'arr[i,0]^2 + ... + arr[i,d-1]^2 = 1'
    for all 'i >= 0' and 'i < arr.shape[0]'.
    
    INPUT: arr
    arr = [[x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd], ... ]
    arr.dtype = float
    MODIFIES INPUT
    
    OUTPUT: 
    """
    
    norm = np.sqrt(np.sum(arr*arr, axis=1))
    arr /= norm[:,np.newaxis]
    np.clip(arr, -1.0, 1.0, out=arr)

def _ensure_within_sphere(arr):
    """
    Returns an array of vertices projected
    on to the 2D unitsphere.
    
    INPUT: arr
    arr = [[x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd], ... ]
    arr.dtype = float
    DOES NOT MODIFY INPUT
    
    OUTPUT: ret
    ret.shape = arr.shape
    ret.dtype = float
    """
    
    ret = np.copy(arr)
    _normalize(ret)
    return ret

def _calc_sphere_forces(arr):
    """
    Calculates the forces among the mesh vertices.
    The repulsive force among pairs is radial
    and has magnitude: f(r) = (1 - r / r_max)^8,
    where r is the Euclidean distance and r_max = 2.
    In the second step this force is projected to be
    orthogonal to the sphere constraints.
    
    Brute force is used for the calculation of distances.
    
    INPUT: arr
    arr = [[Sx1, Sy1, Sz1],
           [Sx1, Sy1, Sz1],
           [Sx1, Sy1, Sz1],
           [Sx1, Sy1, Sz1], ...  ]
    arr.dtype = float
    DOES NOT MODIFY INPUT
    
    OUTPUT: total_F
    total_F = [[Fx1, Fy1, Fz1],
               [Fx1, Fy1, Fz1],
               [Fx1, Fy1, Fz1],
               [Fx1, Fy1, Fz1], ...  ]
    total_F.dtype = float
    """
    diff = arr[:,np.newaxis] - arr # diff[i,j] = arr[i] - arr[j]
    dist = np.sqrt(np.sum(diff*diff, axis=2)) # dist[i,j] has the range [0, 2]
    np.fill_diagonal(dist, 2.0)

    f = 1.0 - dist / 2.0
    f *= f
    f *= f
    f *= f # (1 - d / d_max)^8
    forces = (diff/dist[:,:,np.newaxis]) * f[:,:,np.newaxis]
    total_F = np.sum(forces, axis=1)

    # this ensures that the force is orthogonal to the sphere constraint
    total_F -= arr * np.sum(total_F*arr, axis=1)[:,np.newaxis]

    return total_F

def create_sphere(refine=3, min_step_size=1e-5, max_step_number=1000, min_avg_force=1e-7):
    """
    Creates a 2D unitsphere by recursively subdividing the triangles of an octahedron
    and then relaxing the mesh under repulsive forces. The 2D unitsphere is embedded
    in 3D Euclidean space. For 'refine = 0', the returned mesh describes an octahedron.
    
    INPUT: (refine=3, min_step_size=1e-5, max_step_number=1000, min_avg_force=1e-7)
    type(refine) = int, should be non-negative
    type(min_step_size) = float, should be positive and small
    type(max_step_number) = int, should be positive and large
    type(min_avg_force) = float, should be positive and small
    DOES NOT MODIFY INPUT
    
    OUTPUT: (sphere_vertices, sphere_edges, sphere_triangles)
    sphere_vertices.shape = (int >= 8, 3)
    sphere_vertices.dtype = float
    sphere_edges.shape = (int >= 12, 2)
    sphere_edges.dtype = int
    sphere_triangles.shape = (int >= 8, 3)
    sphere_triangles.dtype = int
    (sphere_vertices, sphere_edges, sphere_triangles) follows the 'mesh_convention'
    """
    
    octahedron_vertices = np.array([
        [ 1.0, 0.0, 0.0], # 0
        [-1.0, 0.0, 0.0], # 1
        [ 0.0, 1.0, 0.0], # 2
        [ 0.0,-1.0, 0.0], # 3
        [ 0.0, 0.0, 1.0], # 4
        [ 0.0, 0.0,-1.0], # 5
    ])

    octahedron_edges = np.array([
        [0, 2], # 0
        [0, 3], # 1
        [0, 4], # 2
        [0, 5], # 3
        [1, 2], # 4
        [1, 3], # 5
        [1, 4], # 6
        [1, 5], # 7
        [2, 4], # 8
        [2, 5], # 9
        [3, 4], # 10
        [3, 5], # 11
    ], dtype=int)

    octahedron_triangles = np.array([ 
        [0, 2, 4,     0, 8, 2],
        [0, 4, 3,     2, 10, 1],
        [0, 3, 5,     1, 11, 3],
        [0, 5, 2,     3, 9, 0],
        [1, 4, 2,     6, 8, 4],
        [1, 2, 5,     4, 9, 7],
        [1, 5, 3,     7, 11, 5],
        [1, 3, 4,     5, 10, 6],
    ], dtype=int)
    
    v, e, t = octahedron_vertices, octahedron_edges, octahedron_triangles
    for i in range(refine):
        v, e, t = refine_mesh(v, e, t)
    _normalize(v)
    
    v = relax_mesh(v, _ensure_within_sphere, _calc_sphere_forces, min_step_size=min_step_size,
                   max_step_number=max_step_number, min_avg_force=min_avg_force, print_all=False)
    
    return v, e, t

##################################################

def _get_plane_positions_and_orientation(sample_positions, orientation_positions, reference_dist, i):
    """
    Receives the 'sample_positions' of the vertices closest to 'i'
    from the ambient Euclidean space and projects them on to the best
    fitting 2D plane. These projected positions are returned. The
    orientation of the two local plane axes relative to the triangle
    from 'orientation_positions' is determined as well.
    
    The deviations orthogonal to the plane are also tested
    to see whether they are larger than 'reference_dist'.
    If they are, a warning is printed.
    
    INPUT: (sample_positions, orientation_positions, reference_dist, i)
    sample_positions = [[x1, x2, x3, ..., xd],
                        [x1, x2, x3, ..., xd],
                        [x1, x2, x3, ..., xd],
                        [x1, x2, x3, ..., xd], ... ]
    sample_positions.dtype = float
    orientation_positions = [[x1, x2, x3, ..., xd],
                             [y1, y2, y3, ..., yd],
                             [z1, z2, z3, ..., zd]]
    orientation_positions.dtype = float
    type(reference_dist) = float
    type(i) = int, should be non-negative
    DOES NOT MODIFY INPUT
    
    OUTPUT: (return_positions, orientation)
    return_positions.shape = (sample_positions.shape[0], 2 or 3)
    return_positions.dtype = float
    dtype(orientation) = float, = +1.0 if the in-plane (y-x) \cross (z-x) > 0, -1.0 otherwise
    """
    
    # https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    # for the best fit plane, the two directions of it must have the largest sigma[0] and sigma[1]
    while True:
        # for some unknown reason, LinAlgError("SVD did not converge") can sometimes be raised
        # when calculating SVD decomposition, probably because the initial guess (that is later
        # refined) is randomly generated
        try:
            U, sigma, V_T = svd(sample_positions, full_matrices=True, compute_uv=True, lapack_driver='gesvd')
        except np.linalg.LinAlgError as e:
            print('LinAlgError: "{}" occurred at vertex {:d}.'.format(str(e), i))
            print("Trying again...\n")
        else:
            break
    axes = V_T # sigma is sorted in decreasing order, thus axes[0] is the x axis and axes[1] is the y axis
    
    plane_positions = np.einsum("nj,ij->ni", sample_positions, axes)
    return_positions = np.empty((plane_positions.shape[0], np.minimum(plane_positions.shape[1], 3)), dtype=float)
    return_positions[:,0] = plane_positions[:,0]
    return_positions[:,1] = plane_positions[:,1]
    
    # one can derive the following identity:
    # (sigma[i])^2 = \sum_{n} (sample_positions[n,:] * V[:,i])^2, or equivalently (V_T[i,:] = V[:,i])
    # np.square(sigma[i]) = np.sum(np.square(np.einsum("nj,j->n", sample_positions, V_T[i,:])))
    # thus a large sigma[i] means a large average deviation along the V[:,i] direction
    if sigma.size > 2:
        z_sd = np.sqrt(np.sum(np.square(sigma[2:])) / sample_positions.shape[0])
        if z_sd > reference_dist:
            print("Anomalously large deviations orthogonal to the best fitting plane found at point {:d}.\n".format(i))
        return_positions[:,2] = np.sqrt(np.sum(np.square(plane_positions[:,2:]), axis=1))
    
    # calculates the orientation of our local plane relative to the reference triangle
    orientation = 1.0
    if isinstance(orientation_positions, np.ndarray):
        pos = np.einsum("nj,ij->ni", orientation_positions , axes[0:2])
        dab = pos[1,:] - pos[0,:]
        dac = pos[2,:] - pos[0,:]
        orientation = np.sign(np.cross(dab, dac))
    
    return return_positions, orientation

def _correct_edge_list(edge_list):
    """
    Ensures that all edges satisfy the convention 'e[0]' is smallest.
    
    INPUT: edge_list
    type(edge_list) = list, should be made of pairs of integers
    DOES NOT MODIFY INPUT
    
    OUTPUT: ret
    type(ret) = list, made of pairs of properly ordered integers
    """
    
    ret = list()
    for e in edge_list:
        if e[1] < e[0]:
            e = (e[1], e[0])
        else:
            e = tuple(e)
        
        ret.append(e)
    
    return ret

def _correct_triangle_list(triangle_list, orientation):
    """
    Ensures that all triangles are properly oriented and
    satisfy the convention 't[0]' is smallest.
    
    INPUT: (triangle_list, orientation)
    type(triangle_list) = list, should be made of triads of integers
    type(orientation) = float, +1.0 or -1.0
    DOES NOT MODIFY INPUT
    
    OUTPUT: ret
    type(ret) = list, made of triads of properly ordered and oriented integers
    """
    
    ret = list()
    for t in triangle_list:
        if t[1] < t[0] and t[1] < t[2]:
            t = (t[1], t[2], t[0])
        elif t[2] < t[0] and t[2] < t[1]:
            t = (t[2], t[0], t[1])
        else:
            t = tuple(t)
        
        if orientation < 0.0:
            t = (t[0], t[2], t[1])
        
        ret.append(t)
    
    return ret


def _points_to_line(arr, init=0, closed=False):
    """
    Returns an array of indices that is ordered
    according to distances and starts with 'init'.
    
    If 'ind_sort[i]' is the i-th index, then the (i+1)-th index
    'ind_sort[i+1]' is the index of the 'arr' array element that is
    1) not already included in 'ind_sort[:i+1]', and
    2) has the smallest Euclidean distance to 'arr[ind_sort[i]]'.
    
    Thus for an 'arr' that is quasi-1D, '_points_to_line'
    orders this collection of points to a line.
    
    INPUT: (arr, init=0, closed=False)
    arr = [[x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd],
           [x1, x2, x3, ..., xd], ... ]
    arr.dtype = float
    dtype(init) = int, should be >= 0 and < arr.shape[0]
    closed = bool, should init be added at the end?
    DOES NOT MODIFY INPUT
    
    OUTPUT: ind_sort
    ind_sort.shape = (arr.shape[0]) if not closed, else (arr.shape[0] + 1)
    ind_sort.dtype = int
    """
    
    size = arr.shape[0]
    
    i = init
    included = np.full((size), False, dtype=bool)
    ind_sort = list()
    while True:
        included[i] = True
        ind_sort.append(i)
        
        diff = arr[~included,:] - arr[i]
        dist = np.sqrt(np.sum(diff*diff, axis=1))
        
        if dist.size > 0:
            min_j = np.argmin(dist)        
            i = np.where(~included)[0][min_j]
        else:
            break
    
    if closed:
        ind_sort.append(init)
    
    return np.array(ind_sort, dtype=int)

def triangulation_finder(vertices, init="random", sample_size=32, cusp_sensitivity=1.6,
                         boundary_tol=20, min_triangle_angle=5,
                         to_plot=False, plot_3D=True, plot_index=0):
    """
    Finds a triangulation of a given mesh of a 2D surface by iterating through the
    mesh (starting from 'init'), finding local triangulations, and combining them
    to a global triangulation of the mesh from 'vertices'. Around every point
    a best fitting 2D plane is found using SVD decomposition. Afterwards, we
    find constrained Delaunay triangulations within these local 2D planes
    (constrained because we have to ensure that previous triangles are respected)
    only to pull them back to the ambient space (assumed Euclidean).
    
    It is assumed that 'vertices' describe a manifold that is embedded in
    a d-dimensional Euclidean space. The Euclidean distances are calculated
    using a brute force method, which is for large N often the most time
    consuming step (~ N^2). The algorithm itself is O(N).
    
    It is a good idea to keep 'sample_size' large enough for the first and second
    order neighbors to be present within every local 2D plane, otherwise spurious
    manifold boundaries might be detected, etc.
    
    The parameter 'cusp_sensitivity' determines how large the deviations along the
    direction orthogonal to the best fitting plane needs to be for a warning to be
    printed out. Smaller 'cusp_sensitivity' means higher sensitivity to cusps, that
    is, smaller deviations are needed to trigger warnings. The length scales used
    in the comparisons is the average smallest distance among vertices.
    
    The boundary of a manifold we define as vertices that relative to the nearest
    vertices have one clockwise angle that is larger than '180 - boundary_tol'
    degrees. For finer meshes we can take smaller 'boundary_tol'.
    
    Triangles that have an angle smaller than 'min_triangle_angle' (in degrees) are
    removed from the mesh. Such triangles usually appear at the boundary and
    are spurious.
    
    There is an option 'to_plot' that controls whether to plot the local 2D plane
    (if 'plot_3D = False'), or surface in 3D (if 'plot_3D = True'), around 
    the point specified by 'plot_index'. For vertices embedded in more than three
    dimensions, the third dimension is equal to the orthogonal distance from the
    best fitting local plane.
    
    INPUT: (vertices, init="random", sample_size=32, boundary_tol=20,
            min_triangle_angle=5, to_plot=False, plot_3D=True, plot_index=0)
    vertices = [[x1, x2, x3, ..., xd],
                [x1, x2, x3, ..., xd],
                [x1, x2, x3, ..., xd],
                [x1, x2, x3, ..., xd], ...]
    vertices.dtype = float
    if init = "random", then a random int from [0, vertices.shape[0]-1] is generated,
    otherwise dtype(init) = int that satisfies init >= 0 and init < vertices.shape[0]
    dtype(sample_size) = int, should be sample_size >= 0 and sample_size < vertices.shape[0]
    dtype(cusp_sensitivity) = float
    dtype(boundary_tol) = float, in degrees
    dtype(min_triangle_angle) = float, in degrees, should be small
    dtype(to_plot) = bool, should a local plane be plotted?
    dtype(plot_3D) = bool, should the local plane be plotted in 2D or 3D?
    dtype(plot_index) = int, should be plot_index >= 0 and plot_index < vertices.shape[0]
    DOES NOT MODIFY INPUT
    
    OUTPUT: (vertices, edges, triangles, other_returns)
    (vertices, edges, triangles) follows the 'mesh_convention'
    dtype(other_returns) = dict, and depending on vertices it may include
    1) improper_edges if the resulting triangulations is not proper, i.e.,
       some edges are adjacent to more than two triangles,
    2) disconnected_vertices if some vertices were not iterated through,
    3) orientation_conflicts if a consistent orientation cannot be defined,
    4) boundary_lines if boundaries were detected.
    """
    
    """
    There are ambiguities in the triangulation that may cause their doubling, for instance:
         D-----------C        
         |           |        where the sides have approximately the same length.
         |           |        This may be triangulated as either ABC+ACD or ABD+BCD,
         |           |        and depending on which local plane we are in (e.g., plane of A vs B),
         |           |        the Delaunay triangulation may give both causing edges with triple triangles.
         A-----------B        
    In our algorithm we have avoided this difficulty by using a constrained
    Delaunay triangulation that enforces already defined edges (segments).
    """
    
    """
    Herbert Edelsbrunner,
    CPS296.1: COMPUTATIONAL TOPOLOGY
    https://www2.cs.duke.edu/courses/fall06/cps296.1/
    
    "... suggests an easy algorithm to recognize a compact
    2-manifold given by its triangulation. First search all triangles
    and orient them consistently as you go until you either succeed,
    establishing orientability, or you encounter a contradiction,
    establishing non-orientability. Thereafter count the vertices,
    edges, and triangles, and the alternating sum uniquely identifies
    the 2-manifold if there is no boundary. Else count the holes, this
    time by searching the edges that belong to only one triangle each.
    For each additional hole the Euler characteristic decreases by one,
    giving chi = 2 - 2 g - h in the orientable case and chi = 2 - g - h
    in the non-orientable case. The genus, g, and the number of holes, h,
    identify a unique 2-manifold with boundary within the orientable
    and the non-orientable classes."
    https://www2.cs.duke.edu/courses/fall06/cps296.1/Lectures/sec-II-1.pdf
    """
    
    print("Calculating distances between vertices.\n")
    
    size = vertices.shape[0]
    diff = vertices[:,np.newaxis] - vertices # diff[i,j] = vertices[i] - vertices[j]
    dist = np.sqrt(np.sum(diff*diff, axis=2)) # Euclidean distance, brute force
    dist_sort = np.argsort(dist, axis=0)
    # for fixed i, dist[dist_sort[:,i], i] is sorted in ascending order
    
    # reference distance for the SVD sigma comparison
    avg_min_dist = np.mean([dist[dist_sort[1:3,i], i] for i in range(size)])
    
    ########## numerous lists, dictionaries, etc.
    edges = list() # a list of tuples of the form (vert_a, vert_b)
    triangles = list() # a list of triangles of the form (vert_a, vert_b, vert_c, edg_ab, edg_bc, edg_ca)
    
    boundary = set() # a set of edg_ab which are at the manifold boundary
    ori_conflict = set() # a set of triag_abc for which we had a conflict in defining its orientation
    
    segments = np.empty((size), dtype=object) # an array of sets that attribute to vert_a the set
    for i in range(size):                     # of all segments (vert_a, vert_b) that must be enforced
        segments[i] = set()                   # during constrained Delaunay triangulation
    
    # dictionaries for finding the appropriate indices in the lists 'edges' and 'triangles' without searching every time
    vert_to_edg = dict() # a dictionary with keys (vert_a, vert_b) and values edg_ab
    vert_to_triag = dict() # a dictionary with keys (vert_a, vert_b, vert_c) and values triag_abc
    
    edg_to_triag = dict() # a dictionary with keys edg_ab and values [triag_abc, triag_abd], that is
                          # triangles that have edg_ab as an edge; len(edg_to_triag[edg_ab]) can only
                          # be 0, 1, or 2 for a proper triangulation
    ##########
    
    to_analyze = list() # a list of tuples of the form (vert_a, (vert_a, vert_b, vert_c)) that
                        # we have to analyze; the given triangle is an orientation reference
    already_analyzed = np.full(size, False, dtype=bool)
    
    if init == "random":
        init = np.random.randint(0, size)
    to_analyze.append( (init, None) )
    
    print("Starting mesh triangulation with vertex {:d}.\n".format(init))
    
    for i, ori_ref in to_analyze:
        if already_analyzed[i]:
            continue
        already_analyzed[i] = True
        
        # the indices of the 'sample_size' closest vertices to 'i'
        sample_indices = dist_sort[0:sample_size,i] # sample_indices[0] = i
        plane_positions, orientation = _get_plane_positions_and_orientation(
                                            diff[sample_indices,i],
                                            diff[ori_ref,i] if ori_ref != None else None,
                                            cusp_sensitivity * avg_min_dist,
                                            i)
        
        # the edges (= segments) from previous triangulations
        sample_segments = set()
        for j in range(sample_size):
            sample_segments.update(segments[sample_indices[j]])
        sample_segments = np.array(list(sample_segments), dtype=int)
        
        # returned constrained Delaunay triangulations all have counterclockwise orientation
        if sample_segments.size > 0:
            index_map = np.full((size), -1)
            index_map[sample_indices] = np.arange(sample_size, dtype=int)
            
            plane_segments = np.empty_like(sample_segments)
            plane_segments[:,0] = index_map[sample_segments[:,0]]
            plane_segments[:,1] = index_map[sample_segments[:,1]]
            
            retain_mask = (plane_segments[:,0] != -1) & (plane_segments[:,1] != -1)
            plane_segments = plane_segments[retain_mask]
            
            delaunay = tr.triangulate({"vertices":plane_positions[:,0:2], "segments":plane_segments}, "pc")
        else: # no segment constraints needed
            delaunay = tr.triangulate({"vertices":plane_positions[:,0:2]}, "pc")
        
        if np.max(delaunay["triangles"].ravel()) >= sample_size:
            print("The segments given at vertex {:d} are inconsistent.".format(i))
            
            lines = np.empty((plane_segments.shape[0], 2, 2), dtype=float)
            lines[:,0] = plane_positions[plane_segments[:,0],0:2]
            lines[:,1] = plane_positions[plane_segments[:,1],0:2]
            
            mask = (delaunay["triangles"][:,0] < sample_size) & \
                   (delaunay["triangles"][:,1] < sample_size) & \
                   (delaunay["triangles"][:,2] < sample_size)
            tri = delaunay["triangles"][mask]
            
            fig = plt.figure()
            fig.gca().set_title("Local plane of vertex {:d}.".format(i))
            
            plt.triplot(plane_positions[:,0], plane_positions[:,1], tri, "b-", lw=3, alpha=0.5)
            fig.gca().add_collection(LineCollection(lines, color='k', ls="--", lw=3, alpha=1.0))
            
            for j in range(sample_size):
                plt.text(plane_positions[j,0], plane_positions[j,1],
                         "{:d}({:d})".format(sample_indices[j], j),
                         c="g", fontsize="medium", ha="center", va="center",
                         bbox=dict(color="w", alpha=0.8, boxstyle="square,pad=0.0"))
            
            blue_patch = mpatches.Patch(color="b", alpha=0.5, label="Generated.")
            black_patch = mpatches.Patch(color="k", alpha=1.0, label="Old, enforced.")
            
            plt.legend(handles=[blue_patch, black_patch], loc="best")
            plt.show()
            
            return vertices, np.zeros((1,2), dtype=int), np.zeros((1,6), dtype=int), dict()
        
        # finds the triangles that have 'i' as a vertex
        plane_triangles = delaunay["triangles"]
        near_0_mask = (plane_triangles[:,0] == 0) | (plane_triangles[:,1] == 0) | (plane_triangles[:,2] == 0)
        plane_triangles = plane_triangles[near_0_mask]
        
        plane_angles = np.empty((plane_triangles.shape[0]), dtype=float)
        for j in range(plane_triangles.shape[0]):
            t = plane_triangles[j]
            if t[1] == 0:
                t = (t[1], t[2], t[0])
            elif t[2] == 0:
                t = (t[2], t[0], t[1])
            else:
                t = tuple(t)
            
            d0b = plane_positions[t[1],0:2] - plane_positions[t[0],0:2]
            d0c = plane_positions[t[2],0:2] - plane_positions[t[0],0:2]
            plane_angles[j] = np.arctan2(np.cross(d0b, d0c), np.dot(d0b, d0c)) # > 0
            if plane_angles[j] < 0:
                print("Wrong orientation generated of triangle ({:d}, {:d}, {:d}) at vertex {:d}.".format(sample_indices[t[0]], sample_indices[t[1]], sample_indices[t[2]], i))
        
        retain_mask = (np.degrees(plane_angles) < 180.0 - boundary_tol) & \
                      (np.degrees(plane_angles) > min_triangle_angle)
        plane_triangles = plane_triangles[retain_mask]
        # in the end, plane_triangles are triangles that have 0 as a vertex, minus the spurious boundary triangles
        
        # checks whether 0 is actually at the manifold boundary
        plane_neighbors = set(plane_triangles.ravel())
        plane_neighbors.remove(0)
        plane_neighbors = np.array(list(plane_neighbors), dtype=int)
        
        plane_angles = np.arctan2(plane_positions[plane_neighbors,1], plane_positions[plane_neighbors,0])
        angle_sort = np.argsort(plane_angles)
        plane_neighbors = plane_neighbors[angle_sort]
        plane_angles = plane_angles[angle_sort]
        angle_differences = np.roll(plane_angles, -1) - plane_angles
        angle_differences[-1] += 2.0 * np.pi
        
        max_angle_ind = np.argmax(angle_differences)
        if np.degrees(angle_differences[max_angle_ind]) > 180.0 - boundary_tol:
            b, c = plane_neighbors[max_angle_ind], plane_neighbors[(max_angle_ind + 1) % angle_differences.size]
            plane_boundary_edges = np.array([(0,b), (0,c)], dtype=int)
            boundary_edges_list = _correct_edge_list(sample_indices[plane_boundary_edges])
        else:
            boundary_edges_list = list()
        # boundary_edges_list holds the two manifold boundary edges that have i as a vertex (if they exist)
        
        triangle_list = _correct_triangle_list(sample_indices[plane_triangles], orientation)
        
        # adding what we determined to the various lists and dictionaries
        for t in triangle_list:
            if t in vert_to_triag:
                continue
            
            if (t[0], t[2], t[1]) in vert_to_triag:
                ori_conflict.add( vert_to_triag[(t[0], t[2], t[1])] )
                continue
            
            edges_list = _correct_edge_list([(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            for e in edges_list:
                if e not in vert_to_edg:
                    edges.append(e)
                    vert_to_edg[e] = len(edges) - 1
                    
                    segments[e[0]].add(e)
                    segments[e[1]].add(e)
            
            edges_list = [vert_to_edg[e] for e in edges_list]
            triangles.append( (t[0], t[1], t[2], edges_list[0], edges_list[1], edges_list[2]) )
            vert_to_triag[t] = len(triangles) - 1
            
            for j in edges_list:
                if j not in edg_to_triag:
                    edg_to_triag[j] = [vert_to_triag[t]]
                else:
                    edg_to_triag[j].append(vert_to_triag[t])
                    
                    if len(edg_to_triag[j]) > 2:
                        print("Edge ({:d}, {:d}) has more than two triangles.".format(edges[j][0], edges[j][1]))
            
            for j in t:
                to_analyze.append( (j,t) )
        
        for e in boundary_edges_list:
            boundary.add(vert_to_edg[e])
        
        # lastly, plotting
        if to_plot and i == plot_index:
            if plot_3D:
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.set_title("Local plane of vertex {:d}.".format(i))
                
                t = delaunay["triangles"]
                ln = np.empty((t.shape[0]*3, 2, 3), dtype=float)
                ln[:t.shape[0],0] = plane_positions[t[:,0]]
                ln[:t.shape[0],1] = plane_positions[t[:,1]]
                ln[t.shape[0]:2*t.shape[0],0] = plane_positions[t[:,1]]
                ln[t.shape[0]:2*t.shape[0],1] = plane_positions[t[:,2]]
                ln[2*t.shape[0]:,0] = plane_positions[t[:,2]]
                ln[2*t.shape[0]:,1] = plane_positions[t[:,0]]
                ln[:,:,2] = - ln[:,:,2]
                ln[:,:,2] -= 1e-2 * avg_min_dist
                ax.add_collection(Line3DCollection(ln, color="b", ls="-", lw=2.8, alpha=0.5))
                
                t = plane_triangles
                ln = np.empty((t.shape[0]*3, 2, 3), dtype=float)
                ln[:t.shape[0],0] = plane_positions[t[:,0]]
                ln[:t.shape[0],1] = plane_positions[t[:,1]]
                ln[t.shape[0]:2*t.shape[0],0] = plane_positions[t[:,1]]
                ln[t.shape[0]:2*t.shape[0],1] = plane_positions[t[:,2]]
                ln[2*t.shape[0]:,0] = plane_positions[t[:,2]]
                ln[2*t.shape[0]:,1] = plane_positions[t[:,0]]
                ln[:,:,2] = - ln[:,:,2]
                ln[:,:,2] += 1e-1 * avg_min_dist
                ax.add_collection(Line3DCollection(ln, color="k", ls=":", lw=3.2, alpha=1.0))
                
                for j in range(sample_size):
                    ax.text(plane_positions[j,0], plane_positions[j,1], -plane_positions[j,2] + 2e-1 * avg_min_dist,
                             "{:d}({:d})".format(sample_indices[j], j),
                             c="g", fontsize="medium", ha="center", va="center",
                             bbox=dict(color="w", alpha=0.6, boxstyle="square,pad=0.0"))
                
                blue_patch = mpatches.Patch(color="b", alpha=0.5, label="Triangles.")
                black_patch = mpatches.Patch(color="k", alpha=1.0, label="To be added.")

                plt.legend(handles=[blue_patch, black_patch], loc="best")
                ax.set_xlim3d(np.min(plane_positions[:,0]), np.max(plane_positions[:,0]))
                ax.set_ylim3d(np.min(plane_positions[:,1]), np.max(plane_positions[:,1]))
                ax.set_zlim3d(-np.max(plane_positions[:,2]), np.min(plane_positions[:,2]))
                plt.show()
            else:
                lines = np.empty((plane_segments.shape[0], 2, 2), dtype=float)
                lines[:,0] = plane_positions[plane_segments[:,0],0:2]
                lines[:,1] = plane_positions[plane_segments[:,1],0:2]
                
                fig = plt.figure()
                fig.gca().set_title("Local plane of vertex {:d}.".format(i))
                
                plt.triplot(plane_positions[:,0], plane_positions[:,1], delaunay["triangles"], "b-", lw=3, alpha=0.5)
                fig.gca().add_collection(LineCollection(lines, color="r", ls="-", lw=3, alpha=0.5))
                plt.triplot(plane_positions[:,0], plane_positions[:,1], plane_triangles, "k:", lw=3, alpha=1.0)
                
                for j in range(sample_size):
                    plt.text(plane_positions[j,0], plane_positions[j,1],
                             "{:d}({:d})".format(sample_indices[j], j),
                             c="g", fontsize="medium", ha="center", va="center",
                             bbox=dict(color="w", alpha=0.8, boxstyle="square,pad=0.0"))
                
                blue_patch = mpatches.Patch(color="b", alpha=0.5, label="New triang.")
                red_patch = mpatches.Patch(color=(2.0/3.0, 0.0, 1.0/3.0), alpha=0.75, label="Old, enforced.")
                black_patch = mpatches.Patch(color="k", alpha=1.0, label="To be added.")

                plt.legend(handles=[blue_patch, red_patch, black_patch], loc="best")
                plt.show()
    
    # the arrays that will be returned
    edges = np.array(edges, dtype=int)
    triangles = np.array(triangles, dtype=int)
    
    boundary = np.array(list(boundary), dtype=int)
    boundary_graph = nx.from_edgelist(edges[boundary])
    boundary_components = [np.array(list(boundary_graph.subgraph(c).nodes), dtype=int)
                           for c in nx.connected_components(boundary_graph)]
    boundary_lines = [bc[_points_to_line(vertices[bc], closed=True)] for bc in boundary_components]
    
    ori_conflict = np.array(list(ori_conflict), dtype=int)
    
    are_edges_improper = np.full((edges.shape[0]), False, dtype=bool)
    for i in range(edges.shape[0]):
        are_edges_improper[i] = len(edg_to_triag[i]) > 2
    
    other_returns = dict()
        
    if np.count_nonzero(are_edges_improper) > 0:
        print("Mesh unsuccessfully triangulated.\n")
        print("In particular, {:d} edges are adjacent to more than two triangles.".format(np.count_nonzero(are_edges_improper)))
        print("An array of these improper edges shall be returned as 'improper_edges'.\n")
        
        other_returns["improper_edges"] = edges[are_edges_improper]
    else:
        print("Mesh successfully triangulated.")
        print("All edges are adjacent to two or less triangles.\n")
    
    if np.count_nonzero(already_analyzed) < size:
        print("The mesh has disconnected parts.")
        print("Only one part of size {:d} (total size {:d}) that is connected to vertex {:d} has been triangulated.".format(np.count_nonzero(already_analyzed), size, init))
        print("The mesh shall be split accordingly. The remaining vertices shall be returned as 'disconnected_vertices'.\n")
        
        vertices, other_vertices = vertices[already_analyzed], vertices[~already_analyzed]
        index_map = np.empty((size), dtype=int)
        index_map[already_analyzed] = np.arange(vertices.shape[0])
        index_map[~already_analyzed] = np.arange(other_vertices.shape[0])
        
        edges[:,0] = index_map[edges[:,0]] # the map is monotonously increasing so we do not need to explicitly
        edges[:,1] = index_map[edges[:,1]] # ensure that the vert_a < vert_b convention is followed
        triangles[:,0] = index_map[triangles[:,0]] # same
        triangles[:,1] = index_map[triangles[:,1]]
        triangles[:,2] = index_map[triangles[:,2]]
        for i in range(len(boundary_components)):
            boundary_lines[i] = index_map[boundary_lines[i]]
        
        other_returns["disconnected_vertices"] = other_vertices
    else:
        print("The mesh is connected and has only one part of size {:d}.\n".format(size))
    
    orientable = True
    if ori_conflict.shape[0] > 0:
        print("The manifold is not orientable.")
        print("In particular, there has been a conflict in giving an orientation to {:d} triangles.".format(ori_conflict.size))
        print("An array of these triangles shall be returned as 'orientation_conflicts'.\n")
        
        orientable = False
        other_returns["orientation_conflicts"] = ori_conflict
    else:
        print("The manifold is orientable.\n")
    
    closed = True
    if len(boundary_lines) > 0:
        print("The manifold has a boundary and therefore is not closed (= compact and without boundary).")
        print("In particular, there are {:d} boundary lines (= number of holes) present.".format(len(boundary_lines)))
        print("A list of arrays of these boundary lines shall be returned as 'boundary_lines'.\n".format(len(boundary_lines)))
        
        closed = False
        other_returns["boundary_lines"] = boundary_lines
    else:
        print("The manifold does not have a boundary and is therefore closed (= compact and without boundary).\n")
    
    v, e, t = vertices.shape[0], edges.shape[0], triangles.shape[0]
    Euler = v - e + t
    h = nx.number_connected_components(boundary_graph)
    
    print("Number of vertices: {:d}.".format(v))
    print("Number of edges: {:d}.".format(e))
    print("Number of triangles: {:d}.".format(t))
    print("Euler characteristic = vertices - edges + triangles = {:d}.".format(Euler))
    print("Number of holes = {:d}.".format(h))
    
    if orientable:
        g = (2 - Euler - h) // 2
        print("Genus = (2 - Euler char. - holes) / 2 = {:d} (for orientable surfaces only).".format(g))
    else:
        k = 2 - Euler - h
        print("Non-orientable genus = 2 - Euler char. - holes = {:d} (for non-orientable surfaces only).".format(k))
    
    return vertices, edges, triangles, other_returns

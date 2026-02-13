import numpy as np
import scipy.spatial as sptl
from openpnm._skgraph.tools import tri_to_am


def odd_layer(shape, z_index):
    x = np.repeat(np.arange(shape[0]), shape[1])*np.sqrt(3)/2
    y_list = []
    for i in range(0, shape[0]):
        if np.mod(i, 2) == 0:
            y_list.extend(np.arange(shape[1]))
        else:
            y_list.extend(np.arange(shape[1])+0.5)
    y = np.array(y_list)
    z = np.ones_like(x) * z_index * np.sqrt(8 / 3)/2
    points = np.vstack([x,y,z]).T.astype(float) + 0.5
    return points


def even_layer(shape, z_index):
    x = (np.repeat(np.arange(shape[0]-1), shape[1])+1/3) * np.sqrt(3)/2
    y_list = []
    for i in range(0, shape[0]-1):
        if np.mod(i, 2) == 0:
            y_list.extend(np.arange(shape[1])+0.5)
        else:
            y_list.extend(np.arange(shape[1]))
    y = np.array(y_list)
    z = np.ones_like(x)*z_index*np.sqrt(8/3)/2
    points = np.vstack([x,y,z]).T.astype(float) + 0.5
    return points


def hcp(shape, spacing=1, mode='tri', node_prefix='node', edge_prefix='edge'):
    r"""
     Generate a hexagonal close packed lattice

     Parameters
     ----------
     shape : array_like
         3x1 array defining number of particles in given direction. Starts by creating a baseplate of
         triangulated particles placed in line along y (2nd parameter) axis. First particle in even rows is shifted by
         in both x and y directions.

     spacing : array_like or float
         The size of a unit cell in each direction. If a scalar is given it is
         applied in all 3 directions.

     Returns
     -------
     network : dict
         A dictionary containing 'coords', 'conns' and various boolean labels
         (i.e. 'node.center')
     """
    shape = np.array(shape)
    planes=[]
    arr = np.atleast_3d(np.empty(shape))
    for k in range(0, shape[2]):
        if np.mod(k, 2) == 0:
            planes.append(odd_layer(shape, k))

        else:
            planes.append(even_layer(shape, k))

    crds = np.concatenate(planes)

    if mode.startswith('tri'):
        tri = sptl.Delaunay(points=crds)
        am = tri_to_am(tri)
        conns = np.vstack((am.row, am.col)).T
        # Trim diagonal connections between cubic pores
        L = np.sqrt(np.sum(np.diff(crds[conns], axis=1)**2, axis=2)).flatten()
        conns = conns[L <= 1.25]

    d = {}
    d[node_prefix + '.coords'] = crds*spacing
    d[edge_prefix + '.conns'] = conns
    return d

if __name__ == "__main__":
    d = hcp((3,3,3))
    import matplotlib.pyplot as plt

    coords = d["node.coords"]
    connections = d["edge.conns"]

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(xs,ys,zs, c='blue', s=50)

    for a, b in connections:
        x_line = [coords[a][0], coords[b][0]]
        y_line = [coords[a][1], coords[b][1]]
        z_line = [coords[a][2], coords[b][2]]
        ax.plot(x_line, y_line, z_line, c='red')

    plt.show()
    print("DONE")
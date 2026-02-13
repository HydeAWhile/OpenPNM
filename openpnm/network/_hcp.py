import numpy as np
from openpnm import topotools
from openpnm.network import Network
from openpnm.utils import Workspace
from openpnm.utils import Docorator
from openpnm import _skgraph as skgr
from openpnm._skgraph.generators import hcp


docstr = Docorator()
ws = Workspace()
__all__ = ['HexagonalClosePacked']


# @docstr.dedent
class HexagonalClosePacked(Network):
    r"""
       Generate a hexagonal close packed lattice

       Parameters
       ----------
       shape : array_like
           The number of rows in given direction. Stacks hexagonal grids aligned with y axis.
       spacing : array_like or float
           The size of a unit cell in each direction. If a scalar is given it is
           applied in all 3 directions.

       Returns
       -------
       network : dict
           A dictionary containing 'coords', 'conns' and various boolean labels
           (i.e. 'node.center')
           """

    def __init__(self, shape, spacing=1, **kwargs):
        super().__init__(**kwargs)
        shape = np.array(shape)
        if np.any(shape < 2):
            raise Exception('HCP lattice networks must have at least 2 '
                            'pores in all directions')
        net = hcp(shape=shape, spacing=spacing,
                  node_prefix='pore', edge_prefix='throat')
        self.update(net)
        self._post_init()
        self["pore.surface"] = skgr.tools.find_surface_nodes_cubic(self)
        Ps = self["pore.surface"]
        self["throat.surface"] = np.all(Ps[self["throat.conns"]], axis=1)
        self.update(skgr.generators.tools.label_faces_cubic(self))

        Ps = self.pores(['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'])
        Ps = self.to_mask(pores=Ps)
        self['pore.surface'] = Ps

        # Finally scale network to specified spacing
        self['pore.coords'] *= np.array(spacing)

    def add_boundary_pores(self, labels, spacing):
        r"""
        Add boundary pores to the specified faces of the network

        Pores are offset from the faces by 1/2 of the given ``spacing``,
        such that they lie directly on the boundaries.

        Parameters
        ----------
        labels : str or list[str]
            The labels indicating the pores defining each face where
            boundary pores are to be added (e.g. 'left' or
            ['left', 'right'])
        spacing : scalar or array_like
            The spacing of the network (e.g. [1, 1, 1]).  This must be
            given since it can be quite difficult to infer from the
            network, for instance if boundary pores have already added
            to other faces.

        """
        spacing = np.array(spacing)
        if spacing.size == 1:
            spacing = np.ones(3)*spacing
        for item in labels:
            Ps = self.pores(item)
            coords = np.absolute(self['pore.coords'][Ps])
            axis = np.count_nonzero(np.diff(coords, axis=0), axis=0) == 0
            offset = np.array(axis, dtype=int)/2
            if np.amin(coords) == np.amin(coords[:, np.where(axis)[0]]):
                offset = -1*offset
            topotools.add_boundary_pores(network=self, pores=Ps, offset=offset,
                                         apply_label=item + '_boundary')

if __name__ == "__main__":
    test = HexagonalClosePacked((4,4,3), spacing= 1)
    import matplotlib.pyplot as plt
    coords = test.coords
    connections = test.conns

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(xs, ys, zs, c='blue', s=50)

    for a, b in connections:
        x_line = [coords[a][0], coords[b][0]]
        y_line = [coords[a][1], coords[b][1]]
        z_line = [coords[a][2], coords[b][2]]
        ax.plot(x_line, y_line, z_line, c='red')

    plt.show()
    print("DONE")
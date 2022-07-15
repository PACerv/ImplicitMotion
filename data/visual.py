from functools import reduce

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D



class Pose3DStrip(object):
    def __init__(self, bone_hierarchy, nstrips, nframes, limits=[-10, 10], title=None):
        self.bone_hierarchy = bone_hierarchy
        self.nstrips = nstrips
        self.nframes = nframes
        self.limits = limits
        self.fig = plt.figure(figsize = (nframes * 3, nstrips*3))
        self.gs1 = gridspec.GridSpec(nstrips, nframes)
        self.gs1.update(wspace=0, hspace=0)
        self.from_idx = reduce(lambda x,y: x+y, [[i] * len(j) for i, j in enumerate(bone_hierarchy) if len(j)!=0])
        self.to_idx = reduce(lambda x,y: x+y, [j for j in bone_hierarchy if len(j)!=0])
        self.idx = 0
        self.axs = []

    def add_sequence(self, seq, title=None):
        for i in range(self.nframes):
            ax = self.fig.add_subplot(self.gs1[self.idx, i])#, projection='3d')
            self.axs.append(ax)
            ax.set_label([i])
            if title is not None:
                ax.set_ylabel([title])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            # ax.set_zticklabels([])
            ax.set_xlim(*self.limits)
            ax.set_ylim(*self.limits)
            # ax.set_xlim3d(*self.limits)
            # ax.set_ylim3d(*self.limits)
            # ax.set_zlim3d(*self.limits)
            ax.scatter(*seq[i,:,:].T, marker='.', c="grey")
            ax.elev = 0
            ax.azim = 0
            data_w_root = np.concatenate([seq[i, :, :], np.zeros([1,3], dtype=seq.dtype)], 0)
            line_data = np.stack([data_w_root[self.from_idx, :], data_w_root[self.to_idx, :]], 1)
            lines = [p for line in line_data for p in ax.plot(*line.T[:2, :],linewidth=3.0)]
        self.idx += 1

    def __del__(self):
        plt.close(self.fig)

    def save(self, path):
        plt.tight_layout()
        for ax in self.axs:
            ax.label_outer()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(path)
        plt.close(self.fig)

class Pose3DAnimator(object):
    def __init__(self, bone_hierarchy, ncols, nrows, limits=[-10, 10], title=None):
        self.fig = plt.figure(figsize=(ncols*3, nrows*3))
        if title is not None:
            self.fig.suptitle(title)
        # tight layout breaks layout
        # self.fig.set_tight_layout(True)
        self.ncols = ncols
        self.nrows = nrows
        self.limits = limits
        self.idx = 1
        self.ani = []
        self.data = []
        self.max_len = 0
        self.from_idx = reduce(lambda x,y: x+y, [[i] * len(j) for i, j in enumerate(bone_hierarchy) if len(j)!=0])
        self.to_idx = reduce(lambda x,y: x+y, [j for j in bone_hierarchy if len(j)!=0])

    def add_sequence(self, seq, title=None):
        ax = self.fig.add_subplot(self.nrows, self.ncols, self.idx, projection='3d')
        
        if title is not None:
            ax.set_title(title)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        self.idx += 1
        self.max_len = max(self.max_len, len(seq))
        ax.elev = 90
        ax.azim = 90
        ax.set_xlim3d(*self.limits)
        ax.set_ylim3d(*self.limits)
        ax.set_zlim3d(*self.limits)

        scat = ax.scatter(*seq[0,:,:].T, marker='.', c="grey")

        data_w_root = np.concatenate([seq[0, :, :], np.zeros([1,3], dtype=seq.dtype)], 0)
        line_data = np.stack([data_w_root[self.from_idx, :], data_w_root[self.to_idx, :]], 1)
        lines = [p for line in line_data for p in ax.plot(*line.T)]
        self.data.append((scat, lines, seq))

    def update_skeleton(self, idx):
        artists = []
        for scat, lines, seq in self.data:
            if idx > len(seq) -1: continue
            scat._offsets3d = (seq[idx, :, :].T)

            # add root to data
            data_w_root = np.concatenate([seq[idx, :, :], np.zeros([1,3])], 0)
            line_data = np.stack([data_w_root[self.from_idx, :], data_w_root[self.to_idx, :]], 1)

            for line, edge in zip(lines, line_data):
                line.set_data(edge[:,0], edge[:, 1])
                line.set_3d_properties(edge[:, 2])

            artists += [scat,] + lines
        return artists

    def init_func(self, *args, **kwargs):
        artists = []
        for scat, line, data in self.data:
            artists += [scat,] + line
        return artists

    def __del__(self):
        plt.close(self.fig)

    def save(self, path):
        # plt.tight_layout()
        self.ani = animation.FuncAnimation(
            self.fig, self.update_skeleton, frames=self.max_len, init_func=self.init_func, interval=30, blit=True)
        self.ani.save(path, dpi=300)
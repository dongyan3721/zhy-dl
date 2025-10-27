import matplotlib.pyplot as plt
from IPython import display

class Animator:
    """可视化训练进度"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.xlim = xlim
        self.ylim = ylim
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.X, self.Y = [], []
        if legend:
            self.lines = [self.ax.plot([], [], label=l)[0] for l in legend]
            self.ax.legend()
        display.display(self.fig)
        plt.ion()

    def add(self, x, ys):
        if not hasattr(ys, "__len__"):
            ys = [ys]
        n = len(ys)
        while len(self.X) < n:
            self.X.append([])
            self.Y.append([])
        for i, y in enumerate(ys):
            self.X[i].append(x)
            self.Y[i].append(y)
            self.lines[i].set_data(self.X[i], self.Y[i])
        self.ax.set_xlim(self.xlim if self.xlim else [0, x+1])
        if self.ylim:
            self.ax.set_ylim(self.ylim)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.relim()
        self.ax.autoscale_view()
        display.display(self.fig)
        display.clear_output(wait=True)
        plt.show()

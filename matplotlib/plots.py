__author__ = 'Anubhav Jain <ajain@lbl.gov>'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math


class HeatMapPlot():

    def __init__(self, data, xlabels, ylabels, m_props=None, discrete_int=False):
        """

        :param data: 2D list or NP array
        :param xlabels: list of str
        :param ylabels: list of str
        :param m_props:
        :param discrete_int:
        """
        self.data = np.array(data)
        self.xlabels = xlabels
        self.ylabels = ylabels

        # default parameters
        self.p = {}
        self.p['figsize'] = (8, 8)
        self.p['plotsize'] = (8, 8)
        self.p['fontname'] = 'Trebuchet MS'
        self.p['fontsize'] = 18
        self.p['xrotation'] = 0
        self.p['yrotation'] = 0
        self.p['colormap'] = 'hot'
        self.p['ncolorbins'] = None  # (int) bin the colormap into discrete areas, e.g. for mapping ints or "AUTO_INT"
        self.p['cbar_ticks'] = None  # (list of numbers) colorbar tick locs or "AUTO_INT"
        self.p['cbar_labels'] = None  # (list of str) labels for the colorbar ticks
        self.p['xlabel'] = None
        self.p['ylabel'] = None
        self.p['vmin'] = None
        self.p['vmax'] = None
        self.p['title'] = None

        # discrete int override: automatically overloads some HeatMap parameters for showing integer data
        if discrete_int:
            min = int(math.floor(self.data.min()))
            max = int(math.ceil(self.data.max()))
            intervals = (max - min) + 1
            self.p['ncolorbins'] = intervals
            self.p['vmin'] = min - 0.5
            self.p['vmax'] = max + 0.5
            self.p['cbar_ticks'] = np.linspace(min, max, intervals)

        # user parameter override
        m_props = m_props if m_props else {}
        self.p.update(m_props)

    def plot(self, export_filename=None):
        """
        :param export_filename: None for screen display, or 'myplot.png'
        """
        plt.clf()  # clear plot

        # colorbar
        cmap = plt.get_cmap(self.p['colormap'], self.p['ncolorbins'])

        # plot
        plt.figure(1, figsize=self.p['plotsize'])
        plt.pcolormesh(self.data, cmap=cmap, vmin=self.p['vmin'], vmax=self.p['vmax'])
        cbar = plt.colorbar(ticks=self.p['cbar_ticks'])

        # title, x, y labels
        if self.p['title']:
            plt.title(self.p['title'])

        if self.p['xlabel']:
            plt.xlabel(self.p['xlabel'], self.p['fontname'], self.p['fontsize'])

        if self.p['ylabel']:
            plt.ylabel(self.p['ylabel'], self.p['fontname'], self.p['fontsize'])

        if self.p['cbar_labels']:
            cbar.ax.set_yticklabels(self.p['cbar_labels'])

        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(self.p['fontsize'])
            t.set_fontname(self.p['fontname'])

        # x and y axes
        ticklok_x = [z + 0.5 for z in range(0, len(self.xlabels))]
        ticklok_y = [z + 0.5 for z in range(0, len(self.ylabels))]
        plt.xticks(ticklok_x, self.xlabels, ha='center', rotation=self.p['xrotation'], fontsize=self.p['fontsize'],
                   fontname=self.p['fontname'])
        plt.yticks(ticklok_y, self.ylabels, ha='right', rotation=self.p['yrotation'], fontsize=self.p['fontsize'],
                   fontname=self.p['fontname'])

        # set proper limits
        plt.xlim(0, len(self.xlabels))
        plt.ylim(0, len(self.ylabels))

        fig = mpl.pyplot.gcf()
        fig.set_size_inches(self.p['figsize'])
        plt.tight_layout()

        # display
        if export_filename:
            plt.savefig(export_filename)
        else:
            plt.show()


class XYPlot():

    def __init__(self, data, m_props=None):
        """

        :param data: list of dict. Each dict must have 'x' and 'y', and optional things like 'style' and 'color'
        :param m_props:
        """
        self.data = data
        # default parameters
        self.p = {}
        self.p['figsize'] = (8, 6)
        self.p['fontname'] = 'Trebuchet MS'
        self.p['fontsize'] = 14
        self.p['xlabel'] = None
        self.p['ylabel'] = None
        self.p['title'] = None
        self.p['legendloc'] = 2
        self.p['xlim'] = None
        self.p['ylim'] = None
        self.p['xticklabels'] = None
        self.p['yticklabels'] = None
        self.p['xrotation'] = None
        self.p['yrotation'] = None


        # series-specific default parameters
        defaults = [('dodgerblue', 'o', 4)]
        defaults.append(('tomato', 's', 4))
        defaults.append(('springgreen', 'D', 4))
        defaults.append(('gray', '^', 6))
        defaults.append(('hotpink', 'v', 6))
        defaults.append(('mediumorchid', 'p', 6))
        for i in range(len(data)):
            defaults.append(('black', '*', 6))

        for idx, d in enumerate(data):
            data[idx]['style'] = data[idx].get('style', 'default')

            # defaults for scatter
            data[idx]['color'] = data[idx].get('color', defaults[idx][0])
            data[idx]['marker'] = data[idx].get('marker', defaults[idx][1])
            data[idx]['markersize'] = data[idx].get('markersize', defaults[idx][2])
            data[idx]['label'] = data[idx].get('label', None)

            # defaults for line / errorbar
            data[idx]['linestyle'] = data[idx].get('linestyle', 'None')
            data[idx]['linewidth'] = data[idx].get('linewidth', None)
            data[idx]['xerr'] = data[idx].get('xerr', None)
            data[idx]['yerr'] = data[idx].get('yerr', None)


            # defaults for annotate
            data[idx]['xytext'] = data[idx].get('xytext', (0, 0))
            data[idx]['arrowprops'] = data[idx].get('arrowprops', {"arrowstyle":"->", "connectionstyle":"arc"})
            data[idx]['bbox'] = dict(boxstyle="square", fc="w", alpha=0.7)


        # user parameter override
        m_props = m_props if m_props else {}
        self.p.update(m_props)


    def plot(self, export_filename=None):
        """
        :param export_filename: None for screen display, or 'myplot.png'
        """
        plt.clf()  # clear plot

        # plot
        plt.figure(1, figsize=self.p['figsize'])
        for d in self.data:
            if d['style'] == 'annotate':
                plt.annotate(d['text'], xy=(d['x'], d['y']), xycoords='data', xytext=d['xytext'], textcoords='offset points', arrowprops=d['arrowprops'], bbox=d['bbox'], horizontalalignment='center', verticalalignment='center')
            else:
                plt.errorbar(d['x'], d['y'], linestyle=d['linestyle'], lw=d['linewidth'], color=d['color'], label=d['label'], marker=d['marker'], markersize=d['markersize'], xerr=d['xerr'], yerr=d['yerr'], ecolor='black')

        # title, x, y labels
        if self.p['title']:
            plt.title(self.p['title'])

        if self.p['xlabel']:
            plt.xlabel(self.p['xlabel'], fontname=self.p['fontname'], fontsize=self.p['fontsize'])

        if self.p['ylabel']:
            plt.ylabel(self.p['ylabel'], fontname=self.p['fontname'], fontsize=self.p['fontsize'])

        # x and y axes tick labels - for custom axes labels
        if self.p['xticklabels']:
            ticklok_x = self.p.get('tickloc_x', range(0, len(self.p['xticklabels'])))
            plt.xticks(ticklok_x, self.p['xticklabels'], ha='center', rotation=self.p['xrotation'], fontsize=self.p['fontsize'],
                   fontname=self.p['fontname'])
        else:
            plt.xticks(fontsize=self.p['fontsize'], fontname=self.p['fontname'])

        if self.p['yticklabels']:
            ticklok_y = self.p.get('tickloc_y', range(0, len(self.p['yticklabels'])))
            plt.yticks(ticklok_y, self.p['yticklabels'], ha='right', rotation=self.p['yrotation'], fontsize=self.p['fontsize'],
                   fontname=self.p['fontname'])
        else:
            plt.yticks(fontsize=self.p['fontsize'], fontname=self.p['fontname'])

        # set proper limits
        if self.p['xlim']:
            plt.xlim(self.p['xlim'][0], self.p['xlim'][1])
        if self.p['ylim']:
            plt.ylim(self.p['ylim'][0], self.p['ylim'][1])

        plt.legend(loc=self.p['legendloc'])

        fig = mpl.pyplot.gcf()
        fig.set_size_inches(self.p['figsize'])
        fig.tight_layout()

        # display
        if export_filename:
            plt.savefig(export_filename)
        else:
            plt.show()
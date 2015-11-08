#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import mpl_toolkits.axisartist as AA
from threading import Thread
import time

class PerfData:
    aData = None
    sName = ""
    sUnit = ""
    nId = -1
    nYAxisId = -1
    ymin = 0
    ymax = 10
    line = None
    ax = None

    def __init__(self, sName, sUnit, nYAxisId):
        assert isinstance(sName, str)
        assert isinstance(sUnit, str)
        assert isinstance(nYAxisId, int)
        self.aData    = []
        self.sName    = sName;
        self.sUnit    = sUnit;
        self.nYAxisId = nYAxisId;

class PerfMonitor:
    aXdata, aPerfData = [], []
    setAxisId = set()
    fig = None
    xmax = 5
    xmin = 0

    def __init__(self):
        pass

    def addData(self, prefData):
        assert isinstance(prefData, PerfData)
        prefData.nId = len(self.aPerfData)
        if len(self.aPerfData) == 0:
            self.fig = plt.figure(1)
            prefData.ax = SubplotHost(self.fig, 111)
            prefData.line, = prefData.ax.plot([], [], lw=1, label=prefData.sName)
            prefData.ax.grid()
            prefData.ax.set_ylim(prefData.ymin, prefData.ymax)
            prefData.ax.set_xlim(self.xmin, self.xmax)
            prefData.ax.set_xlabel("time")
            prefData.ax.set_ylabel(prefData.sName)
            prefData.line.set_data(self.aXdata, prefData.aData)
            self.fig.add_axes(prefData.ax)
            self.fig.patch.set_facecolor('white')
            prefData.ax.axis["left"].label.set_color(prefData.line.get_color())
            self.aPerfData.append(prefData)
        else:
            prefData.ax = self.aPerfData[0].ax.twinx()
            if len(self.setAxisId) >= 2:
                offset = (len(self.setAxisId) - 1) * 60
                new_fixed_axis = prefData.ax.get_grid_helper().new_fixed_axis
                prefData.ax.axis["right"] = new_fixed_axis(loc="right", axes=prefData.ax, offset=(offset, 0))
                prefData.ax.axis["right"].toggle(all=True)

                shrink = 1.0
                for i in range(1,len(self.setAxisId)):
                    shrink = shrink - 0.5**(2*i)
                plt.subplots_adjust(right=shrink)

            prefData.ax.set_ylabel(prefData.sName)
            prefData.line, = prefData.ax.plot([], [], lw=1, label=prefData.sName)
            prefData.ax.axis["right"].label.set_color(prefData.line.get_color())
            prefData.line.set_data(self.aXdata, prefData.aData)
            self.aPerfData.append(prefData)
            
        if not prefData.nYAxisId in self.setAxisId:
            #新たな軸を追加
            self.setAxisId.add(prefData.nYAxisId)

    def run(self, t):
        self.aXdata.append(t)
        xmin = min(self.aXdata)
        xmax = max(self.aXdata)
        removeData = xmax - xmin > 30
        if removeData:
            self.aXdata.pop(0)
        xmin = min(self.aXdata)
        self.xmin = xmin
        self.xmax = max(self.xmax, xmax + 2.0)
        self.aPerfData[0].ax.set_xlim(int(self.xmin), int(self.xmax))

        for data in self.aPerfData:
            assert isinstance(data, PerfData)
            data.aData.append(t+1+data.nId)
            if removeData:
                data.aData.pop(0)
            ymin = min(data.aData)
            ymax = max(data.aData)
            data.ymin = min(data.ymin, ymin)

            if data.ymax < ymax:
                data.ymax = ymax * 1.5
                data.ax.set_ylim(data.ymin, data.ymax)
                #data.ax.figure.canvas.draw()
            data.line.set_data(self.aXdata, data.aData)
            
        self.aPerfData[0].ax.figure.canvas.draw()


class ThreadReadStdIn(Thread):
    def __init__(self, n):
        Thread.__init__(self)
        self.n = n

    def parse_input_header(self, line):
        line = line.rstrip().split(",")

    def parse_input_line(self, line):
        elems = line.rstrip().split(",")
        print(str(elems))

    def run(self):
        line = sys.stdin.readline()
        parse_input_header(line)

        while True:
            line = sys.stdin.readline()
            parse_input_line(line)
            time.sleep(0.2)

if __name__ == "__main__":
    th = ThreadReadStdIn(1)
    th.start()
    th.join()
    #monitor = PerfMonitor()

    #data0 = PerfData("tset1", "kg", 1)
    #monitor.addData(data0)
    
    #data1 = PerfData("tset2", "kg", 2)
    #monitor.addData(data1)

    #data2 = PerfData("tset3", "kg", 3)
    #monitor.addData(data2)

    #data3 = PerfData("tset4", "kg", 4)
    #monitor.addData(data3)

    #ani = animation.FuncAnimation(monitor.fig, monitor.run, blit=False, interval=100)
    #plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  -----------------------------------------------------------------------------------------
#    QSVEnc by rigaya
#  -----------------------------------------------------------------------------------------
#   ソースコードについて
#   ・無保証です。
#   ・本ソースコードを使用したことによるいかなる損害・トラブルについてrigayaは責任を負いません。
#   以上に了解して頂ける場合、本ソースコードの使用、複製、改変、再頒布を行って頂いて構いません。
#  -----------------------------------------------------------------------------------------
import time, sys, threading, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
import mpl_toolkits.axisartist as AA

def perf_mon_log(str):
    f = open("test.txt","a")
    f.write(str + "\n")
    f.close()

class PerfData:
    """
    各データを管理する
    aData ... データ配列
    sName ... データ名
    sUnit ... データの単位
    nId   ... データのインデックス
    ax    ... そのデータが使用するグラフ軸データ
    bShow ... そのデータを表示するか
    """
    aData = None
    sName = ""
    sUnit = ""
    nId = -1
    line = None
    ax = None
    bShow = True

    def __init__(self, sName, sUnit, bShow):
        """
        sName ... データ名
        sName ... データの単位
        """
        assert isinstance(sName, str)
        assert isinstance(sUnit, str)
        self.aData    = []
        self.sName    = sName
        self.sUnit    = sUnit
        self.bShow    = bShow

class PerfYAxis:
    """
    y軸の情報を管理する
    """
    ymin = 0
    ymax = 10
    firstData = 0 #この軸を使用する最初のデータのインデックス
    sUnit = ""
    def __init__(self, ymin, ymax, sUnit, firstData):
        """
        ymin      ... 初期のyの下限値
        ymax      ... 初期のyの上限値
        sUnit     ... 軸の単位
        firstData ... この軸を使用する最初のデータのインデックス
        """
        #単位が"%"なら 0% ～ 100%の範囲を常に表示すれば良い
        self.ymin = 0   if sUnit == "%" else ymin
        self.ymax = 100 if sUnit == "%" else ymax
        self.sUnit = sUnit
        self.firstData = firstData

class PerfMonitor:
    """
    グラフ全体を管理する
    aXdata      ... [float]    x軸のデータ
    aPerfData   ... [PerfData] y軸の様々なデータ
    dictYAxis   ... { str : PerfYAxis } 単位系ごとの軸データ
    fig         ... グラフデータ
    xmin, xmax  ... x軸の範囲
    xkeepLength ... x軸のデータ保持範囲
    """
    aXdata = []
    aPerfData = []
    dictYAxis = { }
    fig = None
    xmax = 5
    xmin = 0
    xkeepLength = 30

    def __init__(self, xkeepLength = 30):
        self.xkeepLength = xkeepLength

    def addData(self, prefData):
        assert isinstance(prefData, PerfData)
        prefData.nId = len(self.aPerfData)
        if prefData.bShow:
            if len(self.aPerfData) == 0:
                perfDataYAxis = PerfYAxis(0, 10, prefData.sUnit, len(self.aPerfData))
                #データがひとつもないとき
                #新しくfigureを作成する
                self.fig = plt.figure(1)
                #ホストとして作成
                prefData.ax = SubplotHost(self.fig, 111)
                self.fig.add_subplot(prefData.ax)
                prefData.line, = prefData.ax.plot([], [], lw=1, label=prefData.sName)
                prefData.ax.grid()
                prefData.ax.set_ylim(perfDataYAxis.ymin, perfDataYAxis.ymax)
                prefData.ax.set_xlim(self.xmin, self.xmax)
                prefData.ax.set_xlabel("time (s)")
                prefData.ax.set_ylabel(prefData.sUnit)
                prefData.line.set_data(self.aXdata, prefData.aData)
            
                self.fig.patch.set_facecolor('white') #背景色の設定
                #prefData.ax.axis["left"].label.set_color(prefData.line.get_color())
                self.dictYAxis[prefData.sUnit] = perfDataYAxis
            else:
                if prefData.sUnit in self.dictYAxis:
                    #すでに同じ単位の軸が存在すれば、そこに追加する
                    perfDataYAxis = self.dictYAxis[prefData.sUnit]
                    prefData.ax = self.aPerfData[perfDataYAxis.firstData].ax
                else:
                    #異なる単位の軸なら、新たに追加する
                    perfDataYAxis = PerfYAxis(0, 10, prefData.sUnit, len(self.aPerfData))
                    prefData.ax = self.aPerfData[0].ax.twinx()
                    if len(self.dictYAxis) >= 2:
                        offset = (len(self.dictYAxis) - 1) * 60
                        new_fixed_axis = prefData.ax.get_grid_helper().new_fixed_axis
                        prefData.ax.axis["right"] = new_fixed_axis(loc="right", axes=prefData.ax, offset=(offset, 0))
                        prefData.ax.axis["right"].toggle(all=True)

                        #軸を表示する場所のため、グラフ領域を縮小する
                        shrink = 0.95
                        for i in range(1,len(self.dictYAxis)):
                            shrink = shrink - (0.075)
                        plt.subplots_adjust(right=shrink)
                    
                    prefData.ax.set_ylabel(prefData.sUnit)
                    self.dictYAxis[prefData.sUnit] = perfDataYAxis

                prefData.line, = prefData.ax.plot([], [], lw=1, label=prefData.sName)
                #prefData.ax.axis["right"].label.set_color(prefData.line.get_color())
                prefData.line.set_data(self.aXdata, prefData.aData)

        self.aPerfData.append(prefData)

    def parse_input_line(self, line):
        elems = line.rstrip().split(",")
        current_time = float(elems[0])
        self.aXdata.append(current_time)
        for i in range(1, len(elems)):
            value = float(elems[i])
            self.aPerfData[i-1].aData.append(value)

    def run(self, t):
        line = sys.stdin.readline()
        self.parse_input_line(line)
        #x軸の範囲を取得
        xmin = min(self.aXdata)
        xmax = max(self.aXdata)
        #指定以上に範囲が長ければ削除
        removeData = xmax - xmin > self.xkeepLength
        if removeData:
            self.aXdata.pop(0)
            xmin = min(self.aXdata)
        #x軸のグラフの範囲を更新
        self.xmin = xmin
        self.xmax = max(self.xmax, xmax + 2.0)
        self.aPerfData[0].ax.set_xlim(int(self.xmin), int(self.xmax))

        for data in self.aPerfData:
            assert isinstance(data, PerfData)
            if removeData:
                data.aData.pop(0)
            
            if data.bShow:
                #単位が"%"の場合は 0 - 100の固定でよい
                if data.sUnit != "%":
                    #自分の単位系全体について調整
                    perfDataYAxis = self.dictYAxis[data.sUnit]
                    ax = self.aPerfData[perfDataYAxis.firstData].ax
                    ymin = min(data.aData)
                    ymax = max(data.aData)
                    perfDataYAxis.ymin = min(perfDataYAxis.ymin, ymin)

                    #y軸のグラフの範囲を更新
                    if perfDataYAxis.ymax < ymax:
                        perfDataYAxis.ymax = ymax * 1.25
                        ax.set_ylim(perfDataYAxis.ymin, perfDataYAxis.ymax)
                        #data.ax.figure.canvas.draw()

                data.line.set_data(self.aXdata, data.aData)
            
        self.aPerfData[0].ax.figure.canvas.draw()

if __name__ == "__main__":
    nInterval = 100
    nKeepLength = 30

    #コマンドライン引数を受け取る
    iargc = 1
    while iargc < len(sys.argv):
        if sys.argv[iargc] == "-i":
            iargc += 1
            try:
                nInterval = int(sys.argv[iargc])
            except:
                nInterval = 100
        if sys.argv[iargc] == "-xrange":
            iargc += 1
            try:
                nKeepLength = int(sys.argv[iargc])
            except:
                nKeepLength = 30
        iargc += 1

    monitor = PerfMonitor(nKeepLength)

    #ヘッダー行を読み込み
    line = sys.stdin.readline()
    elems = line.rstrip().split(",")

    #"()"内を「単位」として抽出するための正規表現
    r = re.compile(r'.*\((.+)\)')
    for counter in elems[1:]:
        #"()"内を「単位」として抽出
        m = r.search(counter)
        unit = "" if m == None else m.group(1)
        #データとして追加 (単位なしや平均は表示しない)
        monitor.addData(PerfData(counter, unit, not (m == None or counter.find("avg") >= 0)))
    
    #凡例を作成
    counter_names = []
    for data in monitor.aPerfData:
        if data.bShow:
            counter_names.append(data.sName)

    plt.legend(tuple(counter_names),   #凡例に表示すべき内容
        shadow = True,                 #影の表示
        loc = "upper left",            #表示位置の指定
        bbox_to_anchor = (0.05, 1.10),
        borderaxespad = 0,
        prop = {'size' : 10})          #フォントサイズの調整

    #アニメーションの開始
    ani = animation.FuncAnimation(monitor.fig, monitor.run, blit=False, interval=nInterval)
    plt.show()

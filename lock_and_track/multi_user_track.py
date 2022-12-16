import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import scipy.cluster.hierarchy as hcluster

data = [json.loads(val) for val in open(
    "/home/*/guicomposer/runtime/gcruntime.v6/mmWave_Demo_Visualizer/26072022_1733_Sandipan_Triple_roof.txt", "r")]

df = pd.DataFrame()
for d in data:
    df = df.append(d, ignore_index=True)

if 'x_coord' in df.columns:
    df = df[['activeFrameCPULoad', 'activity', 'datenow', 'dopplerIdx',
             'interChirpProcessingMargin', 'interFrameCPULoad',
             'interFrameProcessingMargin', 'interFrameProcessingTime',
             'numDetectedObj', 'peakVal', 'range', 'rangeIdx', 'rp_y', 'timenow',
             'transmitOutputTime', 'x_coord', 'y_coord']]
    df.columns = ['activeFrameCPULoad', 'activity', 'datenow', 'dopplerIdx',
                  'interChirpProcessingMargin', 'interFrameCPULoad',
                  'interFrameProcessingMargin', 'interFrameProcessingTime',
                  'numDetectedObj', 'peakVal', 'range', 'rangeIdx', 'rp_y', 'timenow',
                  'transmitOutputTime', 'x', 'y']
df_final = df[['timenow', 'x', 'y', 'dopplerIdx', 'peakVal']]


def Sim(df_final):
    for t, X, Y, D, P in df_final.values:
        for x, y, d, p in zip(X, Y, D, P):
            yield t, x, y, d, p


class Point:
    def __init__(self):
        self.t = 0
        self.x = 0
        self.y = 0
        self.d = 0
        self.p = 0

    def update(self, t, x, y, d, p):
        self.t = t
        self.x = x
        self.y = y
        self.d = d
        self.p = p

    def __repr__(self):
        return f'{self.t},{self.x},{self.y},{self.d},{self.p}'

    @property
    def values(self):
        return self.t, self.x, self.y, self.d, self.p


class HistoryQueue:
    def __init__(self, qsize):
        self.p = None
        self.d = None
        self.y = None
        self.x = None
        self.t = None
        self.lastPoint = None
        self.qsize = qsize
        self.setup()

    def isFull(self):
        return True if self.length == self.qsize else False

    @property
    def length(self):
        return len(self.p)

    def setup(self):
        self.lastPoint = Point()
        self.t = deque(maxlen=self.qsize)
        self.x = deque(maxlen=self.qsize)
        self.y = deque(maxlen=self.qsize)
        self.d = deque(maxlen=self.qsize)
        self.p = deque(maxlen=self.qsize)

    def append(self, t, x, y, d, p):
        self.lastPoint.update(t, x, y, d, p)
        self.t.append(t)
        self.x.append(x)
        self.y.append(y)
        self.d.append(d)
        self.p.append(p)

    def __repr__(self):
        return f't:{self.t}\nx:{self.x}\ny:{self.y}\nd:{self.d}\np:{self.p}\n'


class ClusterFinder:
    def __init__(self, th=1, metric='distance'):
        self.length = None
        self.algo = hcluster
        self.th = th
        self.metric = metric

    def getCluster(self, x, y, lp):
        data = np.concatenate([np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1)], axis=1)
        clusters = self.algo.fclusterdata(data, t=self.th, criterion=self.metric)
        self.length = len(np.unique(clusters))
        user_id = clusters[-1]
        return [*lp.values, user_id]


class StreamSplitter:
    def __init__(self, qsize=40):
        self.splits = defaultdict(lambda: HistoryQueue(qsize))

    def push(self, t, x, y, d, p, user_id):
        self.splits[user_id].append(t, x, y, d, p)

    @property
    def all_ID(self):
        return self.splits.keys()

    @property
    def valid_ID(self):
        valid_ids = []
        for Id in self.splits.keys():
            if (not self.splits[Id].isFull()) or (np.abs(self.splits[Id].d).sum() == 0):
                continue
            valid_ids.append(Id)
        return valid_ids


hq = HistoryQueue(qsize=1000)
cf = ClusterFinder(th=1, metric='distance')
splitter = StreamSplitter(qsize=400)
fig = plt.figure()
ax2 = fig.add_subplot(111)
fig.show()
fig.canvas.draw()
for e in Sim(df_final):
    if e[3] != 0:
        hq.append(e[0], e[1], e[2], e[3], e[4])
    if not hq.isFull():
        continue
    splitter.push(*cf.getCluster(hq.x, hq.y, hq.lastPoint))
    # print('all id: ', splitter.all_ID)
    # print('valid id', splitter.valid_ID)

    #     ax1.cla()
    #     for k in splitter.all_ID:
    #         ax1.scatter(splitter.splits[k].x,splitter.splits[k].y,label=k)
    #     ax1.legend(ncol=10)
    ax2.clear()
    for k in splitter.valid_ID:
        ax2.scatter(splitter.splits[k].x, splitter.splits[k].y, label=k)
    if len(splitter.valid_ID) > 0:
        ax2.legend(ncol=10)
    fig.canvas.draw()
    # clear_output(wait = True)
    plt.pause(0.005)
plt.show()

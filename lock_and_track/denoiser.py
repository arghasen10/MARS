import math
import numpy as np
from scipy.stats import mode
from collections import deque
from scipy.signal import butter, lfilter, freqz

class statOut:
    def __init__(self, x=None, y=None, azim=None, stat=None):
        self.x = x
        self.y = y
        self.azim = azim
        self.stat = stat

    def isSkipped(self):
        return True if self.stat is None else False

    def isSync(self):
        return True if self.stat == 'SYNC' else False

    def isAvailable(self):
        return True if self.stat == 'AVL' else False

    def __repr__(self):
        return f'{self.x},{self.y},{self.azim},{self.stat}'


class CordQueue:
    def __init__(self, maxlen):
        self.x = deque(maxlen=maxlen)
        self.y = deque(maxlen=maxlen)

    def appendCord(self, x, y):
        self.x.append(x)
        self.y.append(y)

    @property
    def length(self):
        return len(self.x)

    @property
    def maxlen(self):
        return self.x.maxlen


class DeNoising:
    def __init__(self, qsize=8, order=6, fs=30.0, cutoff=0.7, offset=10):
        self.qsize = qsize
        self.order = order
        self.fs = fs
        self.offset = offset
        self.cutoff = cutoff
        self.setup()

    def setup(self):
        self.config = self.butter_lowpass(self.cutoff, self.fs, self.order)
        self.mode_queue = CordQueue(maxlen=self.qsize)
        self.lpf_queue = CordQueue(maxlen=int(self.fs + self.offset))

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data):
        y = lfilter(self.config[0], self.config[1], data)
        return y

    def process(self, x, y, p):
        if p < 50:
            return statOut()
        else:
            self.mode_queue.appendCord(x, y)
            if self.mode_queue.length < self.mode_queue.maxlen:
                return statOut(stat="SYNC")
            else:
                self.lpf_queue.appendCord(mode(self.mode_queue.x).mode[0], mode(self.mode_queue.y).mode[0])
                if self.lpf_queue.length < self.lpf_queue.maxlen:
                    return statOut(stat="SYNC")
                else:
                    x_dn = self.butter_lowpass_filter(self.lpf_queue.x)[-1]
                    y_dn = self.butter_lowpass_filter(self.lpf_queue.y)[-1]
                    azim = math.atan(x_dn / y_dn) * 180 / np.pi
                    return statOut(x_dn, y_dn, azim, "AVL")

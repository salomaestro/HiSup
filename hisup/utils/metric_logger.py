# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import defaultdict
from collections import deque

import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        keys = sorted(self.meters)
        # for name, meter in self.meters.items():
        for name in keys:
            meter = self.meters[name]
            loss_str.append(
                "{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def to_dict(self):
        loss_dict = {}
        keys = sorted(self.meters)
        for name in keys:
            meter = self.meters[name]
            name_global_avg = name + '_global_avg'
            name_median = name + '_median'
            loss_dict[name_median] = meter.median
            loss_dict[name_global_avg] = meter.global_avg
        return loss_dict

    def tensorborad(self, iteration, writter, phase='train'):
        for name, meter in self.meters.items():
            if 'loss' in name:
                # writter.add_scalar('average/{}'.format(name), meter.avg, iteration)
                writter.add_scalar('{}/global/{}'.format(phase,name), meter.global_avg, iteration)
                # writter.add_scalar('median/{}'.format(name), meter.median, iteration)


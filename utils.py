from __future__ import annotations
from collections import defaultdict
import pickle
import sys
import torch
import json
from typing import Any
import os
import time
import psutil
import os.path
import pathlib
from typing import Any, Callable, List, Iterable, Optional, TypeVar, Dict, IO, Tuple

from collections import Counter

import torch

import time


T = TypeVar("T", str, bytes)

def verify_str_arg(value: T, arg: Optional[str] = None, valid_values: Iterable[T] = None, custom_msg: Optional[str] = None,) -> T:
    if not isinstance(value, torch._six.string_classes):
        if arg is None:
            msg = 'Expected type str, but got type {type}.'
        else:
            msg = 'Expected type str for argument {arg}, but got type {type}.'

        msg = msg.format(type=type(value), arg=arg)
        raise ValueError(msg)

    if valid_values is None:
        return value

    if value not in valid_values:
        if custom_msg is not None:
            msg = custom_msg
        else:
            msg = ('Unknown value for argument.'
                    'Valid values are {{{valid_values}}}.')
            msg = msg.format(value=value, arg=arg, valid_values=iterable_to_str(valid_values))

        raise ValueError(msg)

    return value

class Timer:
    def __init__(self):
        self.time_dict = Counter()

    def log(self, key, value):
        self.time_dict[key] += value

    def start(self):
        self.t0 = time.time_ns()

    def end(self, key):
        t1 = time.time_ns()
        if self.t0 is not None:
            self.log(key, t1 - self.t0)
        self.t0 = None

    def __repr__(self):
        total = sum(i for i in self.time_dict.values())
        string = 'key,absolute_time,relative_time\n'

        for key in self.time_dict:
            absolute_time = str(self.time_dict[key])
            relative_time = str(self.time_dict[key] / total)

            string += key 
            string += ','
            string += absolute_time
            string += ','
            string += relative_time
            string += '\n'

        return string

from pytorch_lightning.callbacks import Callback

class TimerPrintCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(trainer.train_dataloader.dataset.datasets.augment.timer)

    def on_validation_epoch_end(self, trainer, pl_module):
        for dl in trainer.val_dataloaders:
            print(dl.dataset.augment.timer)


class HParamsSaveCallback(Callback):
    def __init__(self, full_hparams_yaml):
        self.full_hparams_yaml = full_hparams_yaml

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["full_config"] = self.full_hparams_yaml

def get_subset_of_size(dataset, count=96):
    indices = []
    target_count = Counter()
    for (idx, target) in enumerate(dataset.targets):
        if target_count[target] < count:
            target_count[target] += 1
            indices.append(idx)

    return torch.utils.data.Subset(dataset, indices)


def get_mem_info(pid: int) -> dict[str, int]:
  res = defaultdict(int)
  for mmap in psutil.Process(pid).memory_maps():
    res['rss'] += mmap.rss
    res['pss'] += mmap.pss
    res['uss'] += mmap.private_clean + mmap.private_dirty
    res['shared'] += mmap.shared_clean + mmap.shared_dirty
    if mmap.path.startswith('/'):
      res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
  return res


class MemoryMonitor():
  def __init__(self, pids: list[int] = None):
    if pids is None:
      pids = [os.getpid()]
    self.pids = pids

  def add_pid(self, pid: int):
    assert pid not in self.pids
    self.pids.append(pid)

  def _refresh(self):
    self.data = {pid: get_mem_info(pid) for pid in self.pids}
    return self.data

  def str(self):
    self._refresh()
    keys = list(list(self.data.values())[0].keys())
    res = []
    for pid in self.pids:
      s = f"PID={pid}"
      for k in keys:
        v = self.format(self.data[pid][k])
        s += f", {k}={v}"
      res.append(s)
    return "\n".join(res)

  @staticmethod
  def format(size: int) -> str:
    for unit in ('', 'K', 'M', 'G'):
      if size < 1024:
        break
      size /= 1024.0
    return "%.1f%s" % (size, unit)


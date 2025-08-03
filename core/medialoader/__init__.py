# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

class LoadSample(ABC):
    def __init__(self):
        self.mode = None
        self.w = None
        self.h = None
        self.fps = None

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

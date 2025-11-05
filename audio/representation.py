# representation.py --- Audio feature computation: Mel spectrograms and MFCC extraction.
# Copyright (C) 2025  Yigit Akoymak
# Created: 2025-11-05
# This file is part of the Lyvio.
#
# The Music Genre CNN project is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Lyvio project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this project.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from visualization import plot_waveform
import matplotlib.pyplot as plt

class AudioFeatureExtractor:
    def __init__(self, sample_rate: int  = 16000, duration: int = 30):
        self.sample_rate = sample_rate
        self.duration = duration

    @staticmethod
    def pre_emphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
        if signal.size == 0:
            return signal.astype(np.float32)
        emphasized = np.append(signal[0], signal[1:] - alpha * signal[:-1])
        return emphasized.astype(np.float32)

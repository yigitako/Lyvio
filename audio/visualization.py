# visualization.py --- Visualization tools for audio waveforms, spectrograms, and feature embeddings.
# Copyright (C) 2025  Yigit Akoymak
# Created: 05-11-2025
# This file is part of the Music Genre CNN project.
#
# The Music Genre CNN project is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The Music Genre CNN project is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this project.  If not, see <https://www.gnu.org/licenses/>.

import librosa.display
import numpy as np

def plot_waveform(y: np.ndarray, sr: int):
    """Visualize the waveform of an audio signal."""
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_waveform_comparison(
    original: np.ndarray,
    emphasized: np.ndarray,
    sr: int,
    alpha: float = 0.97
):
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(original, sr=sr)

    plt.title("Original Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(emphasized, sr=sr)
    plt.title(f"Pre-emphasized Waveform (α={alpha})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


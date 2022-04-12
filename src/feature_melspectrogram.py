from argparse import ArgumentParser
from pathlib import Path

from essentia.streaming import (
    MonoLoader,
    TensorflowInputMusiCNN,
    TensorflowInputVGGish,
    FrameCutter,
)
import essentia.standard as es
from essentia import Pool, run, reset
import numpy as np


class MelSpectrogramMusiCNN:
    def __init__(self):
        self.sample_rate = 16000
        self.hop_size = 256
        self.frame_size = 512

        self.pool = Pool()
        self.loader = MonoLoader(sampleRate=self.sample_rate)
        self.frameCutter = FrameCutter(frameSize=self.frame_size, hopSize=self.hop_size)
        self.mels = TensorflowInputMusiCNN()

        self.loader.audio >> self.frameCutter.signal
        self.frameCutter.frame >> self.mels.frame
        self.mels.bands >> (self.pool, "mel_bands")

    def compute(self, audio_file):
        self.loader.configure(sampleRate=self.sample_rate, filename=str(audio_file))
        run(self.loader)

        melbands = self.pool["mel_bands"].copy()
        self.pool.clear()
        reset(self.loader)

        return melbands


class MelSpectrogramVGGish:
    def __init__(self):
        self.sample_rate = 16000
        self.hop_size = 160
        self.frame_size = 400

        self.pool = Pool()
        self.loader = MonoLoader(sampleRate=self.sample_rate)
        self.frameCutter = FrameCutter(frameSize=self.frame_size, hopSize=self.hop_size)
        self.mels = TensorflowInputVGGish()

        self.loader.audio >> self.frameCutter.signal
        self.frameCutter.frame >> self.mels.frame
        self.mels.bands >> (self.pool, "mel_bands")

    def compute(self, audio_file):
        self.loader.configure(sampleRate=self.sample_rate, filename=audio_file)
        run(self.loader)

        melbands = self.pool["mel_bands"].copy()
        self.pool.clear()
        reset(self.loader)

        return melbands


class MelSpectrogramOpenL3:
    def __init__(self, hop_time):
        self.hop_time = hop_time

        self.sr = 48000
        self.n_mels = 128
        self.frame_size = 2048
        self.hop_size = 242
        self.a_min = 1e-10
        self.d_range = 80
        self.db_ref = 1.0

        self.patch_samples = int(1 * self.sr)
        self.hop_samples = int(self.hop_time * self.sr)

        self.w = es.Windowing(
            size=self.frame_size,
            normalized=False,
        )
        self.s = es.Spectrum(size=self.frame_size)
        self.mb = es.MelBands(
            highFrequencyBound=self.sr / 2,
            inputSize=self.frame_size // 2 + 1,
            log=False,
            lowFrequencyBound=0,
            normalize="unit_tri",
            numberBands=self.n_mels,
            sampleRate=self.sr,
            type="magnitude",
            warpingFormula="slaneyMel",
            weighting="linear",
        )

    def compute(self, audio_file):
        audio = es.MonoLoader(filename=str(audio_file), sampleRate=self.sr)()

        batch = []
        for audio_chunk in es.FrameGenerator(
            audio, frameSize=self.patch_samples, hopSize=self.hop_samples
        ):
            melbands = np.array(
                [
                    self.mb(self.s(self.w(frame)))
                    for frame in es.FrameGenerator(
                        audio_chunk,
                        frameSize=self.frame_size,
                        hopSize=self.hop_size,
                        validFrameThresholdRatio=0.5,
                    )
                ]
            )

            melbands = 10.0 * np.log10(np.maximum(self.a_min, melbands))
            melbands -= 10.0 * np.log10(np.maximum(self.a_min, self.db_ref))
            melbands = np.maximum(melbands, melbands.max() - self.d_range)
            melbands -= np.max(melbands)

            batch.append(melbands.copy())
        return np.vstack(batch)


def feature_melspectrogram(feature_type, hop_time=1):
    if feature_type == "musicnn_melspectrogram":
        extractor = MelSpectrogramMusiCNN()
    elif feature_type == "vggish_melspectrogram":
        extractor = MelSpectrogramVGGish()
    elif feature_type == "openl3_melspectrogram":
        extractor = MelSpectrogramOpenL3(hop_time)
    else:
        raise NotImplementedError(f"{feature_type} not supported yet")
    return extractor

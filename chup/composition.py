from pathlib import Path
from tempfile import TemporaryFile
from . import decomposition as d
import boto3
import numpy as np
import soundfile as sf

class Composition:
    s3 = boto3.client('s3')

    @classmethod
    def concatenate(cls, compositions):
        sample_rate = compositions[0].sample_rate
        samples = [composition.samples for composition in compositions]
        return Composition(0, np.concatenate(samples), sample_rate)

    @classmethod
    def from_s3(cls, bucket, key):
        with TemporaryFile() as temp:
            cls.s3.download_fileobj(bucket, key, temp)
            temp.seek(0)
            return cls.from_soundfile(temp, dtype='float32')

    @classmethod
    def from_soundfile(cls, *args, **kwargs):
        samples, sample_rate = sf.read(*args, **kwargs)
        return cls(0, samples, sample_rate)

    def __init__(self, lead_in, samples, sample_rate):
        self.lead_in = lead_in
        self.samples = samples
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.samples)

    def compose(self):
        return self

    def decompose(self, wavelet, levels):
        if levels < 2:
            return self
        else:
            return d.Decomposition.from_composition(wavelet, self, levels)

    def decorrelated(self):
        input = self.samples
        shape = input.shape
        half = shape[0] >> 1
        full = half << 1
        temp_shape = (half, 2) + shape[1:]
        output_shape = (full,) + shape[1:]
        print('{} -> {} -> {}'.format(shape, temp_shape, output_shape))
        temp = np.fliplr(np.reshape(input[:full], temp_shape))
        output = np.reshape(temp, output_shape)
        return Composition(self.lead_in, output, self.sample_rate)

    def muted(self):
        return Composition(self.lead_in, np.zeros_like(self.samples), self.sample_rate)

    def reversed(self):
        return Composition(self.lead_in, np.flipud(self.samples), self.sample_rate)

    def to_s3(self, bucket, key, **kwargs):
        with TemporaryFile() as temp:
            self.to_soundfile(temp, **kwargs)
            temp.seek(0)
            self.s3.upload_fileobj(temp, bucket, key)

    def to_soundfile(self, file, **kwargs):
        return sf.write(file, self.samples, self.sample_rate, **kwargs)

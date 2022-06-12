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
        return Composition(np.concatenate(samples), sample_rate)

    @classmethod
    def from_s3(cls, bucket, key):
        with TemporaryFile() as temp:
            cls.s3.download_fileobj(bucket, key, temp)
            temp.seek(0)
            return cls.from_soundfile(temp, dtype='float32')

    @classmethod
    def from_soundfile(cls, *args, **kwargs):
        samples, sample_rate = sf.read(*args, **kwargs)
        return cls(samples, sample_rate)

    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate

    def decompose(self, wavelet):
        return d.Decomposition.from_composition(wavelet, self)

    def muted(self):
        return Composition(np.zeros_like(self.samples), self.sample_rate)

    def to_s3(self, bucket, key, **kwargs):
        with TemporaryFile() as temp:
            self.to_soundfile(temp, **kwargs)
            temp.seek(0)
            self.s3.upload_fileobj(temp, bucket, key)

    def to_soundfile(self, file, **kwargs):
        return sf.write(file, self.samples, self.sample_rate, **kwargs)

from tempfile import TemporaryFile

import boto3
import numpy as np
import pyopencl as cl
import soundfile as sf
from .decomposition import Decomposition

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
        data, sample_rate = sf.read(*args, **kwargs)
        print('got data')
        samples = np.ndarray(data.shape[:1], cl.cltypes.float2, data.data)
        print('got samples')
        return cls(samples, sample_rate)

    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate

    def __repr__(self):
        return '<{0} samples at {1}Hz>'.format(len(self.samples), self.sample_rate)

    def decompose(self):
        Decomposition.from_composition(self)

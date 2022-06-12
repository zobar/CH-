import numpy as np
import pyopencl as cl
from . import composition as c

class Decomposition:

    @classmethod
    def from_composition(cls, wavelet, composition):
        (approximation_samples, detail_samples) = wavelet.decompose(composition.samples)

        approximation_sample_rate = round(composition.sample_rate / 2)
        approximation = c.Composition(approximation_samples, approximation_sample_rate)

        detail_sample_rate = composition.sample_rate - approximation_sample_rate
        detail = c.Composition(detail_samples, detail_sample_rate)

        return cls(wavelet, approximation, detail)

    def __init__(self, wavelet, approximation, detail):
        self.wavelet = wavelet
        self.approximation = approximation
        self.detail = detail

    def coefficients(self):
        return np.array([self.approximation.samples, self.detail.samples])

    def compose(self):
        samples = self.wavelet.compose(self.coefficients())
        return c.Composition(samples, self.sample_rate())

    def sample_rate(self):
        return self.approximation.sample_rate + self.detail.sample_rate

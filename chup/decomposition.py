import numpy as np
import pyopencl as cl
from . import composition as c

class Decomposition:

    @classmethod
    def from_composition(cls, wavelet, composition, levels):
        (approximation_samples, detail_samples) = wavelet.decompose(composition.samples)

        approximation_sample_rate = round(composition.sample_rate / 2)
        approximation = c.Composition(0, approximation_samples, approximation_sample_rate).decompose(wavelet, levels - 1)

        detail_sample_rate = composition.sample_rate - approximation_sample_rate
        detail = c.Composition(0, detail_samples, detail_sample_rate)

        return cls(wavelet, approximation, detail)

    def __init__(self, wavelet, approximation, detail):
        self.wavelet = wavelet
        self.approximation = approximation
        self.detail = detail

    def compose(self):
        approximation = self.approximation.compose()
        detail = self.detail.compose()

        lead_in_a = approximation.lead_in
        lead_in_d = detail.lead_in
        lead_in = max(lead_in_a, lead_in_d)

        end_a = len(approximation) + lead_in_a
        end_d = len(detail) + lead_in_d
        end = min(end_a, end_d)

        a_cropped = approximation.samples[lead_in - lead_in_a:end - lead_in_a]
        d_cropped = detail.samples[lead_in - lead_in_d:end - lead_in_d]

        coefficients = np.array([a_cropped, d_cropped])
        samples = self.wavelet.compose(coefficients)

        full_lead_in = self.wavelet.lead_in() + (lead_in * 2)

        return c.Composition(full_lead_in, samples, self.sample_rate)

    def decorrelated(self):
        return Decomposition(self.wavelet, self.approximation.decorrelated(), self.detail.decorrelated())

    def reversed(self):
        return Decomposition(self.wavelet, self.approximation.reversed(), self.detail.reversed())

    @property
    def sample_rate(self):
        return self.approximation.sample_rate + self.detail.sample_rate

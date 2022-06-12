#!/usr/bin/python3

from argparse import ArgumentParser
from tempfile import TemporaryFile
from .composition import Composition
from .library import Library
from . import wavelet

import boto3
import numpy as np
import soundfile as sf

s3 = boto3.client('s3')
list_objects_v2 = s3.get_paginator('list_objects_v2')

parser = ArgumentParser(prog='CH▲')
parser.add_argument('bucket')
parser.add_argument('input', nargs='+')
args = parser.parse_args()

keys = [info['Key']
        for prefix in args.input
        for page in list_objects_v2.paginate(Bucket=args.bucket, Prefix=prefix)
        for info in page.get('Contents', [])]

library = Library.from_s3(args.bucket, keys)
mega = Composition.concatenate(library.compositions)

for w in wavelet.rbio:
    print(w.name)

    decomposition = mega.decompose(w)
    gpu_approximation = decomposition.approximation.samples
    gpu_detail = decomposition.detail.samples

    approximation, detail = decomposition.wavelet.decompose_cpu(mega.samples)
    lead_in = (decomposition.wavelet.length - 2) >> 1
    cpu_approximation = np.resize(approximation[lead_in:], gpu_approximation.shape)
    cpu_detail = np.resize(detail[lead_in:], gpu_detail.shape)

    max_a_delta = np.max(np.abs(gpu_approximation - cpu_approximation))
    max_d_delta = np.max(np.abs(gpu_detail - cpu_detail))

    print('Maximum approximation error: {}'.format(max_a_delta))
    print('Maximum detail error: {}'.format(max_d_delta))

    composition = decomposition.compose()
    lead_in = decomposition.wavelet.length - 2
    round_trip = composition.samples
    original = np.resize(mega.samples[lead_in:], round_trip.shape)
    delta = np.max(np.abs(original - round_trip))
    print('Maximum reconstruction error: {}'.format(delta))

    print()

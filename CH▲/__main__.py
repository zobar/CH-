#!/usr/bin/python3

from argparse import ArgumentParser
from tempfile import TemporaryFile
from .composition import Composition
from .library import Library

import boto3
import numpy as np
import soundfile as sf

s3 = boto3.client('s3')
list_objects_v2 = s3.get_paginator('list_objects_v2')

parser = ArgumentParser(prog='CH▲')
parser.add_argument('bucket')
parser.add_argument('input', nargs='+')
args = parser.parse_args()

for prefix in args.input:
    keys = [info['Key']
            for page in list_objects_v2.paginate(Bucket=args.bucket, Prefix=prefix)
            for info in page.get('Contents', [])]
    library = Library.from_s3(args.bucket, keys)
    mega = Composition.concatenate(library.compositions)
    mega.decompose()

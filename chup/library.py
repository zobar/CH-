from .composition import Composition

class Library:

    @classmethod
    def from_s3(cls, bucket, keys):
        compositions = [Composition.from_s3(bucket, key) for key in keys]
        return cls(compositions)

    def __init__(self, compositions):
        self.compositions = compositions

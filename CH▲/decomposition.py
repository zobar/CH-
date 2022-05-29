import pyopencl as cl

class Decomposition:

    ctx = cl.create_some_context()
    prg = cl.Program(ctx, '''
    ''').build()

    @classmethod
    def from_composition(cls, composition):
        channels = composition.channels


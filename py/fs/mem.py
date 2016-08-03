import fs._fs as c

class Mem:
    """Mem is a memory allocation utility.

    Args:
       name (str): name of the allocated memory
       size (unsigend long long): size of the allocation

    Raises:
       MemoryError: if sufficient memory is not available

    Attributes:
       name (str)
       mem_using (int)
       mem_allocated (int)
    """

    def __init__(self, name, size=0):
        self._mem= c._mem_alloc(name, size)
        self.name= name
        mem= c._mem(self._mem)
        self.mem_allocated= mem[0]
        self.mem_using= mem[1]

    def __repr__(self):
        return "Mem %s, using %d/%d bytes" % (
            self.name, self.mem_using, self.mem_allocated)

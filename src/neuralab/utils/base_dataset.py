from itertools import pairwise
from typing import Generator, Sequence
from flax import struct

class BatchIterable[T: Sequence](Sequence[T]):
    __slots__ = ("inner","batch_size")

    def __init__(self, inner: T, batch_size: int):
        self.batch_size = batch_size
        self.inner = inner

    def __len__(self) -> int:
        return (len(self.inner) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i: int) -> Sequence[T]:
        return self.inner[i * self.batch_size : (i + 1) * self.batch_size]

    def __iter__(self) -> Generator[Sequence[T], None, None]:
        for n, m in pairwise(range(0, len(self.inner) + 1, self.batch_size)):
            yield self.inner[n:m]

@struct.dataclass()
class BaseDataset(Sequence, struct.PyTreeNode):
    ...
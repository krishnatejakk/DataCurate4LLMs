from .base_compressor import BaseCompressor

class MissingDependencyException(Exception):
    """
    Is raised when an underlying dependency is not
    found when loading a library.
    """

    def __init__(self, compression_library: str) -> None:
        self.message = f"""
        Compression Library ({compression_library})
        is missing an underlying dependency. Try
        installing those missing dependencies and
        load this again.

        Common missing dependencies for:

        * lzma:
            * brew install xz
            * sudo apt-get install lzma liblzma-dev libbz2-dev

        * bz2:
            * sudo apt-get install lzma liblzma-dev libbz2-dev

        """
        super().__init__(self.message)


class GZipCompressor(BaseCompressor):
    """
    gzip compressor that inherits from
    `npc_gzip.compressors.base.BaseCompressor`

    >>> compressor: BaseCompressor = GZipCompressor()
    >>> example: str = "Hello there!"
    >>> compressed_length: int = compressor.get_compressed_length(example)
    >>> bits_per_character: float = compressor.get_bits_per_character(example)
    >>> assert isinstance(compressed_length, int)
    >>> assert isinstance(bits_per_character, float)
    """

    def __init__(self) -> None:
        super().__init__(self)

        try:
            import gzip
        except ModuleNotFoundError as e:
            raise MissingDependencyException("gzip") from e

        self.compressor = gzip
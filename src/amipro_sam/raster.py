"""Strict, inert raster PNG validation for caller-supplied ``Image`` blocks.

Only one canonical form is accepted: 8-bit truecolour RGB, non-interlaced, filter type 0
on every scanline, with only ``IHDR``, consecutive ``IDAT`` and ``IEND`` plus purely
informational ancillary chunks. Anything that could change rendered pixels (for example
``tRNS`` transparency or colour-management chunks) is rejected rather than ignored.

Renderers never forward the source bytes: they re-encode the validated pixels with
:func:`encode_rgb_png`, so third-party libraries only ever parse a freshly generated PNG.
Limits are shared with the WMF raster path.
"""

from __future__ import annotations

import struct
import zlib

from .model import Image
from .wmf import _SAFE_DIMENSION, _SAFE_PIXELS, _SAFE_PNG_BYTES

__all__ = [
    "RasterDecodeError",
    "decode_rgb_png",
    "encode_rgb_png",
    "image_display_size",
    "image_png",
]

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_INFORMATIONAL_CHUNKS = {b"tEXt", b"zTXt", b"iTXt", b"tIME", b"pHYs"}


class RasterDecodeError(ValueError):
    """A caller-supplied image is not an accepted canonical raster PNG."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _check_dimensions(width: int, height: int) -> None:
    if (
        not 1 <= width <= _SAFE_DIMENSION
        or not 1 <= height <= _SAFE_DIMENSION
        or width * height > _SAFE_PIXELS
    ):
        raise RasterDecodeError("unsafe-dimensions", "raster dimensions are outside safe limits")


def encode_rgb_png(
    width_px: int, height_px: int, rgb_data: bytes, *, compression_level: int = 9
) -> bytes:
    """Encode top-down packed RGB as a canonical PNG (byte-identical to ``wmf_png``)."""

    _check_dimensions(width_px, height_px)
    row = width_px * 3
    if len(rgb_data) != row * height_px:
        raise RasterDecodeError("inconsistent-pixels", "RGB storage does not match dimensions")
    scanlines = bytearray((row + 1) * height_px)
    for index in range(height_px):
        start = index * (row + 1)
        scanlines[start + 1 : start + 1 + row] = rgb_data[index * row : (index + 1) * row]
    result = (
        _SIGNATURE
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width_px, height_px, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(bytes(scanlines), level=compression_level))
        + _chunk(b"IEND", b"")
    )
    if len(result) > _SAFE_PNG_BYTES:
        raise RasterDecodeError("generated-output-limit", "generated PNG is too large")
    return result


def decode_rgb_png(data: bytes) -> tuple[int, int, bytes]:
    """Validate a canonical RGB PNG and return (width, height, top-down RGB)."""

    if not isinstance(data, bytes | bytearray) or len(data) > _SAFE_PNG_BYTES:
        raise RasterDecodeError("input-limit", "PNG input is missing or too large")
    if not data.startswith(_SIGNATURE):
        raise RasterDecodeError("not-png", "data does not start with the PNG signature")
    position = len(_SIGNATURE)
    header: tuple[int, ...] | None = None
    idat: list[bytes] = []
    idat_closed = False
    seen_end = False
    while position < len(data):
        if position + 8 > len(data):
            raise RasterDecodeError("truncated", "PNG chunk header is truncated")
        length, kind = struct.unpack_from(">I4s", data, position)
        body_start = position + 8
        body_end = body_start + length
        if body_end + 4 > len(data):
            raise RasterDecodeError("truncated", "PNG chunk body is truncated")
        body = bytes(data[body_start:body_end])
        (crc,) = struct.unpack_from(">I", data, body_end)
        if zlib.crc32(kind + body) & 0xFFFFFFFF != crc:
            raise RasterDecodeError("bad-crc", f"PNG chunk {kind!r} fails its CRC")
        position = body_end + 4
        if header is None:
            if kind != b"IHDR" or length != 13:
                raise RasterDecodeError("bad-header", "PNG must start with a 13-byte IHDR")
            header = struct.unpack(">IIBBBBB", body)
            width, height, depth, colour, compression, filtering, interlace = header
            if (depth, colour, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                raise RasterDecodeError(
                    "unsupported-format",
                    "only 8-bit non-interlaced RGB PNG (colour type 2) is accepted",
                )
            _check_dimensions(width, height)
            continue
        if kind == b"IDAT":
            if idat_closed:
                raise RasterDecodeError("idat-not-consecutive", "IDAT chunks are not consecutive")
            idat.append(body)
            continue
        if idat:
            idat_closed = True
        if kind == b"IEND":
            seen_end = True
            if position != len(data):
                raise RasterDecodeError("trailing-data", "bytes follow the IEND chunk")
            break
        if kind == b"tRNS":
            raise RasterDecodeError(
                "png-transparency-unsupported", "tRNS transparency is not accepted"
            )
        if kind not in _INFORMATIONAL_CHUNKS:
            raise RasterDecodeError("png-chunk-unsupported", f"PNG chunk {kind!r} is not accepted")
    if header is None or not idat or not seen_end:
        raise RasterDecodeError("truncated", "PNG lacks IHDR, IDAT or IEND")
    width, height = header[0], header[1]
    expected = height * (width * 3 + 1)
    inflater = zlib.decompressobj()
    try:
        raw = inflater.decompress(b"".join(idat), expected + 1)
    except zlib.error as exc:
        raise RasterDecodeError("bad-deflate", "IDAT data does not inflate") from exc
    if (
        len(raw) != expected
        or not inflater.eof
        or inflater.unused_data
        or inflater.unconsumed_tail
    ):
        raise RasterDecodeError(
            "bad-deflate", "IDAT must inflate to exactly the image size as one complete stream"
        )
    row = width * 3
    rgb = bytearray(row * height)
    for index in range(height):
        start = index * (row + 1)
        if raw[start] != 0:
            raise RasterDecodeError("unsupported-filter", "only scanline filter type 0 is accepted")
        rgb[index * row : (index + 1) * row] = raw[start + 1 : start + 1 + row]
    return width, height, bytes(rgb)


def image_png(image: Image) -> bytes:
    """Re-encode a caller-supplied canonical PNG ``Image`` as a fresh PNG."""

    if image.media_type not in (None, "image/png"):
        raise RasterDecodeError("unsupported-media-type", "only image/png is accepted")
    if not isinstance(image.data, bytes):
        raise RasterDecodeError("missing-data", "image carries no data")
    return encode_rgb_png(*decode_rgb_png(image.data))


def image_display_size(
    image: Image, *, max_width_in: float = 6.5, max_height_in: float = 8.0
) -> tuple[float, float]:
    """Bounded display size: physical inches when given, else px/96; fitted to the body."""

    width, height, _rgb = decode_rgb_png(image.data)
    physical_width = image.width_in if _positive(image.width_in) else None
    physical_height = image.height_in if _positive(image.height_in) else None
    if physical_width is None or physical_height is None:
        physical_width, physical_height = width / 96.0, height / 96.0
    maximum_width = max_width_in if _positive(max_width_in) else 6.5
    maximum_height = max_height_in if _positive(max_height_in) else 8.0
    scale = min(1.0, maximum_width / physical_width, maximum_height / physical_height)
    return physical_width * scale, physical_height * scale


def _positive(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and 0 < value < 1_000
    )

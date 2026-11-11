"""Caller-supplied canonical RGB PNG ``Image`` blocks (strict validation, fresh re-encoding).

All images are generated in the tests; no corpus data is used.
"""

from __future__ import annotations

import re
import struct
import zlib
from io import BytesIO
from zipfile import ZipFile

import pytest

from amipro_sam import raster
from amipro_sam.model import Document, Image, Paragraph, TextRun, WmfGraphic
from amipro_sam.parser import parse_bytes
from amipro_sam.raster import (
    RasterDecodeError,
    decode_rgb_png,
    encode_rgb_png,
    image_display_size,
    image_png,
)
from amipro_sam.renderers import RENDERERS
from amipro_sam.renderers import pdf as pdf_renderer
from amipro_sam.wmf import _SAFE_DIMENSION, _SAFE_PIXELS, _SAFE_PNG_BYTES, wmf_png

SIGNATURE = b"\x89PNG\r\n\x1a\n"
RGB = bytes(range(18))  # 3 x 2 pixels


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def ihdr(width: int, height: int, depth: int = 8, colour: int = 2, interlace: int = 0) -> bytes:
    return chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, interlace))


def scanlines(width: int, height: int, rgb: bytes, filters: list[int] | None = None) -> bytes:
    row = width * 3
    filters = filters or [0] * height
    return b"".join(bytes([filters[i]]) + rgb[i * row : (i + 1) * row] for i in range(height))


def png(*chunks: bytes) -> bytes:
    return SIGNATURE + b"".join(chunks)


def canonical(width: int = 3, height: int = 2, rgb: bytes = RGB) -> bytes:
    return png(
        ihdr(width, height),
        chunk(b"IDAT", zlib.compress(scanlines(width, height, rgb))),
        chunk(b"IEND", b""),
    )


def code_of(data: bytes) -> str:
    with pytest.raises(RasterDecodeError) as raised:
        decode_rgb_png(data)
    return raised.value.code


def test_round_trip_and_wmf_identity() -> None:
    assert decode_rgb_png(encode_rgb_png(3, 2, RGB)) == (3, 2, RGB)
    assert encode_rgb_png(3, 2, RGB) == wmf_png(WmfGraphic(3, 2, RGB, "sha"))


def test_limits_are_shared_with_wmf() -> None:
    assert (raster._SAFE_DIMENSION, raster._SAFE_PIXELS, raster._SAFE_PNG_BYTES) == (
        _SAFE_DIMENSION,
        _SAFE_PIXELS,
        _SAFE_PNG_BYTES,
    )


@pytest.mark.parametrize(("width", "height"), [(4096, 976), (2000, 2000)])
def test_dimensions_at_the_caps_are_accepted(width: int, height: int) -> None:
    rgb = bytes(width * height * 3)
    assert decode_rgb_png(encode_rgb_png(width, height, rgb))[:2] == (width, height)


@pytest.mark.parametrize(("width", "height"), [(4097, 1), (1, 4097), (2000, 2001), (0, 1)])
def test_dimensions_past_the_caps_are_rejected(width: int, height: int) -> None:
    assert code_of(png(ihdr(width, height), chunk(b"IEND", b""))) == "unsafe-dimensions"


def test_input_size_cap_is_exact() -> None:
    base = canonical()
    filler = _SAFE_PNG_BYTES - len(base) - 12  # one tEXt chunk has 12 bytes of overhead
    text = chunk(b"tEXt", b"k\x00" + b"x" * (filler - 2))
    exact = base[:-12] + text + base[-12:]
    assert len(exact) == _SAFE_PNG_BYTES
    assert decode_rgb_png(exact)[2] == RGB
    longer = base[:-12] + chunk(b"tEXt", b"k\x00" + b"x" * (filler - 1)) + base[-12:]
    assert code_of(longer) == "input-limit"


def test_split_idat_is_accepted() -> None:
    data = zlib.compress(scanlines(3, 2, RGB))
    assert (
        decode_rgb_png(
            png(ihdr(3, 2), chunk(b"IDAT", data[:5]), chunk(b"IDAT", data[5:]), chunk(b"IEND", b""))
        )[2]
        == RGB
    )


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (canonical()[:-5], "truncated"),
        (canonical()[:-1] + b"\x00", "bad-crc"),
        (canonical() + b"junk", "trailing-data"),
        (
            png(
                ihdr(3, 2),
                chunk(b"IDAT", zlib.compress(scanlines(3, 2, RGB) + b"\x00")),
                chunk(b"IEND", b""),
            ),
            "bad-deflate",
        ),
        (
            png(
                ihdr(3, 2),
                chunk(b"IDAT", zlib.compress(scanlines(3, 2, RGB)[:-1])),
                chunk(b"IEND", b""),
            ),
            "bad-deflate",
        ),
        (
            png(
                ihdr(3, 2),
                chunk(b"IDAT", zlib.compress(scanlines(3, 2, RGB))[:6]),
                chunk(b"tEXt", b"k\x00v"),
                chunk(b"IDAT", zlib.compress(scanlines(3, 2, RGB))[6:]),
                chunk(b"IEND", b""),
            ),
            "idat-not-consecutive",
        ),
        (
            png(
                ihdr(3, 2),
                chunk(b"PLTE", b"\x00" * 3),
                chunk(b"IDAT", zlib.compress(scanlines(3, 2, RGB))),
                chunk(b"IEND", b""),
            ),
            "png-chunk-unsupported",
        ),
        (png(ihdr(3, 2, colour=0), chunk(b"IEND", b"")), "unsupported-format"),
        (png(ihdr(3, 2, colour=6), chunk(b"IEND", b"")), "unsupported-format"),
        (png(ihdr(3, 2, depth=16), chunk(b"IEND", b"")), "unsupported-format"),
        (png(ihdr(3, 2, interlace=1), chunk(b"IEND", b"")), "unsupported-format"),
    ],
)
def test_malformed_inputs_are_rejected(data: bytes, code: str) -> None:
    assert code_of(data) == code


def test_bomb_cannot_inflate_past_the_image_size() -> None:
    bomb = zlib.compress(scanlines(3, 2, RGB) + b"\x00" * 10_000_000)
    assert code_of(png(ihdr(3, 2), chunk(b"IDAT", bomb), chunk(b"IEND", b""))) == "bad-deflate"


def test_concatenated_junk_and_missing_checksum_streams_are_rejected() -> None:
    body = scanlines(3, 2, RGB)
    two_streams = zlib.compress(body) + zlib.compress(b"")
    junk_tail = zlib.compress(body) + b"\x01\x02"
    compressor = zlib.compressobj()
    no_checksum = compressor.compress(body) + compressor.flush(zlib.Z_SYNC_FLUSH)
    for stream in (two_streams, junk_tail, no_checksum):
        assert (
            code_of(png(ihdr(3, 2), chunk(b"IDAT", stream), chunk(b"IEND", b""))) == "bad-deflate"
        )


def test_transparency_and_appearance_chunks_are_rejected_text_is_not() -> None:
    idat = chunk(b"IDAT", zlib.compress(scanlines(3, 2, RGB)))
    end = chunk(b"IEND", b"")
    assert (
        code_of(png(ihdr(3, 2), chunk(b"tRNS", b"\x00\x00" * 3), idat, end))
        == "png-transparency-unsupported"
    )
    assert (
        code_of(png(ihdr(3, 2), chunk(b"gAMA", struct.pack(">I", 45455)), idat, end))
        == "png-chunk-unsupported"
    )
    assert decode_rgb_png(png(ihdr(3, 2), chunk(b"tEXt", b"Comment\x00hi"), idat, end))[2] == RGB


def test_wrong_implementations_are_caught() -> None:
    # A header-only validator would accept both; only filter-0 scanlines are canonical.
    filtered = png(
        ihdr(3, 2), chunk(b"IDAT", zlib.compress(scanlines(3, 2, RGB, [0, 1]))), chunk(b"IEND", b"")
    )
    assert code_of(filtered) == "unsupported-filter"
    pil = pytest.importorskip("PIL.Image")
    gradient = pil.new("RGB", (64, 64))
    gradient.putdata([(x * 4, y * 4, (x + y) * 2) for y in range(64) for x in range(64)])
    buffer = BytesIO()
    gradient.save(buffer, format="PNG")
    assert code_of(buffer.getvalue()) == "unsupported-filter"


def test_display_size() -> None:
    image = Image(data=canonical(), media_type="image/png", width_in=1.5, height_in=1.0)
    assert image_display_size(image) == (1.5, 1.0)
    assert (
        image_display_size(Image(data=canonical(), width_in=20, height_in=10), max_width_in=6.5)[0]
        == 6.5
    )
    assert image_display_size(Image(data=canonical())) == (3 / 96, 2 / 96)


def _document(*blocks) -> Document:
    return Document("t.sam", "cp1252", blocks=list(blocks))


def _marked_png() -> bytes:
    base = canonical()
    return base[:-12] + chunk(b"tEXt", b"Comment\x00SOURCEMARKER") + base[-12:]


def test_pdf_draws_the_image() -> None:
    payload = RENDERERS["pdf"](
        _document(Image(data=canonical(), media_type="image/png", width_in=1.5, height_in=1.0))
    )

    assert payload.count(b"/Subtype /Image") == 1
    assert re.search(rb"/Width 3\b", payload) and re.search(rb"/Height 2\b", payload)


def _pdf_library_inputs(monkeypatch, image: Image) -> list[bytes]:
    received: list[bytes] = []
    original = pdf_renderer.ReportLabImage

    def spy(source, *args, **kwargs):
        received.append(source.getvalue())
        return original(source, *args, **kwargs)

    monkeypatch.setattr(pdf_renderer, "ReportLabImage", spy)
    RENDERERS["pdf"](_document(image))
    return received


def test_reportlab_receives_only_the_fresh_encoding(monkeypatch) -> None:
    source = _marked_png()
    fresh = encode_rgb_png(*decode_rgb_png(source))

    assert _pdf_library_inputs(monkeypatch, Image(data=source, media_type="image/png")) == [fresh]


def test_spy_detects_a_source_forwarding_renderer(monkeypatch) -> None:
    # Negative control: if the renderer forwarded the source, the spy must see it.
    source = _marked_png()
    monkeypatch.setattr(pdf_renderer, "image_png", lambda image: image.data)

    received = _pdf_library_inputs(monkeypatch, Image(data=source, media_type="image/png"))
    assert received == [source] and b"SOURCEMARKER" in received[0]


def test_odt_and_docx_embed_only_the_fresh_encoding(monkeypatch) -> None:
    source = _marked_png()
    fresh = encode_rgb_png(*decode_rgb_png(source))
    document = _document(Image(data=source, media_type="image/png", width_in=1.0, height_in=0.5))

    with ZipFile(BytesIO(RENDERERS["odt"](document))) as archive:
        pictures = [
            archive.read(name) for name in archive.namelist() if name.startswith("Pictures/")
        ]
    assert pictures == [fresh]

    docx = pytest.importorskip("docx")
    received: list[bytes] = []
    original = docx.text.run.Run.add_picture

    def spy(self, stream, *args, **kwargs):
        received.append(stream.getvalue())
        stream.seek(0)
        return original(self, stream, *args, **kwargs)

    monkeypatch.setattr(docx.text.run.Run, "add_picture", spy)
    with ZipFile(BytesIO(RENDERERS["docx"](document))) as archive:
        media = [
            archive.read(name) for name in archive.namelist() if name.startswith("word/media/")
        ]
    assert received == [fresh] and media == [fresh]


def test_invalid_png_is_a_visible_placeholder() -> None:
    bad = Image(
        data=png(ihdr(3, 2, colour=6), chunk(b"IEND", b"")), media_type="image/png", alt_text="pic"
    )
    payload = RENDERERS["pdf"](_document(bad))

    assert b"/Subtype /Image" not in payload
    flowable = pdf_renderer._image_flowable(bad, None, None)
    assert "PNG not rendered: unsupported-format" in flowable.getPlainText()
    with ZipFile(BytesIO(RENDERERS["odt"](_document(bad)))) as archive:
        assert not [name for name in archive.namelist() if name.startswith("Pictures/")]


def test_bmp_image_placeholder_is_unchanged() -> None:
    bmp = Image(data=b"BM" + b"\x00" * 60, media_type="image/bmp")

    assert pdf_renderer._image_placeholder(bmp) == (
        "[Image: Embedded image (embedded image preserved as a placeholder)]"
    )
    assert b"/Subtype /Image" not in RENDERERS["pdf"](_document(bmp))


@pytest.mark.parametrize("first", ["image", "wmf"])
def test_page_number_restart_binds_in_picture_first_odt(first: str) -> None:
    variant = "\n\t\t".join(("15840", "12240", "37", "720", "900", "1", "1080", "540", "19"))
    layout = (
        f"[lay]\n\tInvented\n\t1284\n\t[rght]\n\t\t{variant}\n"
        "\t[frght]\n\t\t[txt]\n\t\t\tF <:P11,5,>\n\t\t>\n[elay]\n"
    )
    text = "[ver]\n\t4\n[sty]\n\t\n" + layout + "[edoc]\nbody\n>\n"
    document = parse_bytes(text.replace("\n", "\r\n").encode("cp1252"))
    picture = (
        Image(data=canonical(), media_type="image/png", width_in=1, height_in=0.5)
        if first == "image"
        else WmfGraphic(3, 2, RGB, "sha", width_in=1, height_in=0.5)
    )
    document.blocks.insert(0, picture)

    with ZipFile(BytesIO(RENDERERS["odt"](document))) as archive:
        content = archive.read("content.xml").decode("utf-8")
    first_paragraph = re.search(
        r"<office:text[^>]*>\s*<text:p text:style-name=\"([^\"]+)\"", content
    )
    assert first_paragraph, "first body paragraph must carry a style"
    name = first_paragraph.group(1)
    style = re.search(
        rf'<style:style style:name="{re.escape(name)}"[^>]*>.*?</style:style>', content, re.S
    )
    assert style and 'style:master-page-name="Standard"' in style.group(0)
    assert 'style:page-number="5"' in style.group(0)
    assert content.count(f'text:style-name="{name}"') == 1


def test_textual_outputs_keep_placeholders() -> None:
    document = _document(
        Image(data=canonical(), media_type="image/png", alt_text="pic"),
        Paragraph(runs=[TextRun("x")]),
    )

    assert "pic" in RENDERERS["text"](document).decode("utf-8")
    assert decode_rgb_png(image_png(document.blocks[0])) == (3, 2, RGB)

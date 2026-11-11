"""Additive source provenance: `record_span` and `anchor_source` on main-stream blocks.

All documents are invented and built in memory; every assertion compares the exact raw
bytes the recorded span selects.
"""

from __future__ import annotations

import json

import pytest

from amipro_sam.model import (
    Footnote,
    Frame,
    Header,
    PageBreak,
    Paragraph,
    SourceSpan,
    UnsupportedObject,
)
from amipro_sam.parser import parse_bytes
from amipro_sam.renderers import RENDERERS

CHARSET = "[charset]\n\t82\n\tANSI (Windows, IBM CP 1250)\n"
TEXT_FRAME = (
    "[frm]\n\t0\n\t524288\n\t0\n\t0\n\t1440\n\t1440\n\t[frmname]\n\t\tF\n\t[txt]\nframe text\n>\n"
)
TABLE_FRAME = (
    "[frm]\n\t1\n\t524288\n\t0\n\t0\n\t4000\n\t1000\n"
    "\t[tbl]\n\t\t1 1 0 0 0 0 0 0 0\n\t\t[h]\n\t\t\t0 300\n\t\t[e]\n"
    "\t\t[w]\n\t\t\t0 4000\n\t\t[e]\n\t\t[data]\n\t\t\t0 0\ncell\n>\n"
)


def _doc(body: str, *, extra: str = "", encoding: str = "cp1250", charset: str = CHARSET):
    text = "[ver]\n\t4\n[sty]\n\t\n" + charset + extra + "[edoc]\n" + body + "\n>\n"
    data = text.replace("\n", "\r\n").encode(encoding)
    return data, parse_bytes(data, encoding=None if charset else encoding)


def _slice(data: bytes, span: SourceSpan | None) -> bytes | None:
    return None if span is None else data[span.byte_offset : span.end_byte_offset]


def test_multiline_paragraph_span_covers_all_lines() -> None:
    data, document = _doc("first line\nsecond line\n\nnext")
    first, second = [b for b in document.blocks if isinstance(b, Paragraph)]

    assert _slice(data, first.record_span) == b"first line\r\nsecond line"
    assert _slice(data, first.source) == b"first line"  # unchanged: first physical line only
    assert _slice(data, second.record_span) == b"next"


def test_bare_style_line_merged_into_next_line() -> None:
    data, document = _doc("@S@\nText")
    (paragraph,) = [b for b in document.blocks if isinstance(b, Paragraph)]

    assert _slice(data, paragraph.record_span) == b"@S@\r\nText"


def test_empty_bare_style_record_emits_nothing() -> None:
    data, document = _doc("A\n\n@S@\n\nB")
    paragraphs = [b for b in document.blocks if isinstance(b, Paragraph)]

    assert [_slice(data, p.record_span) for p in paragraphs] == [b"A", b"B"]


def test_anchor_only_paragraph_frame() -> None:
    data, document = _doc("<:A0>", extra=TEXT_FRAME)
    frame = next(b for b in document.blocks if isinstance(b, Frame))

    assert frame.source.byte_offset < data.index(b"[edoc]")  # still the header record
    assert _slice(data, frame.anchor_source) == b"<:A0>"
    assert _slice(data, frame.record_span) == b"<:A0>"


def test_inline_anchor_blocks_share_the_record_span() -> None:
    data, document = _doc("before<:A0>after", extra=TEXT_FRAME)
    kinds = [type(b).__name__ for b in document.blocks]
    spans = {_slice(data, b.record_span) for b in document.blocks}

    assert kinds[:3] == ["Paragraph", "Frame", "Paragraph"]
    assert spans == {b"before<:A0>after"}


def test_table_anchor_inline_and_alone() -> None:
    data, document = _doc("before<:t0>after", extra=TABLE_FRAME)
    frame = next(b for b in document.blocks if isinstance(b, Frame))
    assert _slice(data, frame.anchor_source) == b"<:t0>"
    assert _slice(data, frame.record_span) == b"before<:t0>after"

    data, document = _doc("<:t0>", extra=TABLE_FRAME)
    frame = next(b for b in document.blocks if isinstance(b, Frame))
    assert _slice(data, frame.anchor_source) == b"<:t0>"


def test_missing_and_mismatched_anchors_record_their_occurrence() -> None:
    data, document = _doc("x<:t9>y")
    missing = next(b for b in document.blocks if isinstance(b, UnsupportedObject))
    assert _slice(data, missing.anchor_source) == b"<:t9>"

    data, document = _doc("x<:A0>y", extra=TABLE_FRAME)
    occurrences = [
        _slice(data, b.anchor_source)
        for b in document.blocks
        if isinstance(b, Frame | UnsupportedObject) and b.anchor_source is not None
    ]
    assert occurrences and set(occurrences) == {b"<:A0>"}


def test_anchor_offsets_count_bytes_not_characters() -> None:
    data, document = _doc(
        "BEFOREπ<:A0>AFTER\nline2 π<:A1>",
        extra=TEXT_FRAME + TEXT_FRAME.replace("\t0\n\t524288", "\t1\n\t524288"),
        encoding="utf-8",
        charset="",
    )
    anchors = [_slice(data, b.anchor_source) for b in document.blocks if isinstance(b, Frame)]

    # The second anchor sits on the record's second physical line after a 2-byte
    # character: a character-index or "\n"-join implementation would miss it.
    assert anchors == [b"<:A0>", b"<:A1>"]


def test_container_record_span_runs_from_opener_to_terminator() -> None:
    data, document = _doc("before<:F\none\n\ntwo\n>\nafter")
    footnote = next(b for b in document.blocks if isinstance(b, Footnote))
    paragraphs = [b for b in document.blocks if isinstance(b, Paragraph)]

    assert _slice(data, footnote.record_span) == b"<:F\r\none\r\n\r\ntwo\r\n>"
    assert [_slice(data, p.record_span) for p in paragraphs] == [b"before", b"after"]


@pytest.mark.parametrize(
    ("body", "record"),
    [
        ("FIRST<:p<* >>\n\nSECOND", b"FIRST<:p<* >>"),  # trailing break after text
        ("FIRST\n\n<:p<* >>\n\nSECOND", b"<:p<* >>"),  # standalone break
        ("FIRST\n\n<:p<* >>SECOND", b"<:p<* >>SECOND"),  # attached (break before)
    ],
)
def test_page_break_records(body: str, record: bytes) -> None:
    data, document = _doc(body)
    carrier = next(
        b
        for b in document.blocks
        if isinstance(b, PageBreak) or (isinstance(b, Paragraph) and b.page_break_before)
    )

    assert _slice(data, carrier.record_span) == record


def test_non_main_stream_blocks_have_no_provenance() -> None:
    fixed = TEXT_FRAME.replace("524288", "537002048")
    layout = "[lay]\n\tStandard\n\t[hrght]\n\t\t[txt]\n\t\t\tHeader text\n\t\t>\n"
    _data, document = _doc("body", extra=fixed + layout)

    for block in document.blocks:
        if isinstance(block, Frame) and block.anchor_source is None:
            assert block.record_span is None
            assert all(getattr(inner, "record_span", None) is None for inner in block.blocks)
        if isinstance(block, Header):
            assert block.record_span is None


def test_provenance_is_json_additive_and_ignored_by_equality() -> None:
    data, document = _doc("hello")
    paragraph = document.blocks[0]
    clone = Paragraph(runs=paragraph.runs, source=paragraph.source)

    assert paragraph.record_span is not None and clone == paragraph
    payload = json.loads(RENDERERS["json"](document).decode("utf-8"))
    assert "record_span" in json.dumps(payload)


@pytest.mark.parametrize("encoding", ["utf-7", "iso2022_jp"])
def test_stateful_codecs_keep_conversion_and_drop_within_line_offsets(encoding: str) -> None:
    data, document = _doc("ππ<:A0>", extra=TEXT_FRAME, encoding=encoding, charset="")
    frame = next(b for b in document.blocks if isinstance(b, Frame))

    assert frame.anchor_source is None
    assert document.blocks  # conversion still succeeds


def test_multi_character_mapping_codec_falls_back() -> None:
    raw = (
        "[ver]\r\n\t4\r\n[sty]\r\n\t\r\n[edoc]\r\nA".encode("ascii")
        + b"\x88\x62"
        + b"<:A9>B\r\n>\r\n"
    )
    document = parse_bytes(raw, encoding="big5hkscs")
    missing = next(b for b in document.blocks if isinstance(b, UnsupportedObject))

    assert missing.anchor_source is None


def test_utf8_bom_and_embedded_tail_keep_exact_spans() -> None:
    data, document = _doc("café", encoding="utf-8-sig", charset="")
    assert _slice(data, document.blocks[0].record_span) == "café".encode()

    body_data, body_doc = _doc("hello")
    tail = b"PAYLOAD\r\n[Embedded]\r\n"
    tailed = body_data + tail + b"%08d\r\n" % (len(body_data) + len(b"PAYLOAD\r\n"))
    document = parse_bytes(tailed)
    assert _slice(tailed, document.blocks[0].record_span) == b"hello"


def test_leading_preamble_offsets_stay_absolute() -> None:
    data, _document = _doc("hello")
    shifted = b"junk\r\n" + data
    document = parse_bytes(shifted)

    assert _slice(shifted, document.blocks[0].record_span) == b"hello"


def test_partial_utf16_unit_does_not_break_conversion() -> None:
    text = "[ver]\r\n\t4\r\n[sty]\r\n\t\r\n[edoc]\r\nhello\r\n>\r\n"
    raw = b"\xff\xfe" + text.encode("utf-16-le") + b"\x41"
    document = parse_bytes(raw)

    assert any(isinstance(b, Paragraph) for b in document.blocks)

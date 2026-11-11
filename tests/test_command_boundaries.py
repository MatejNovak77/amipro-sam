"""Escape-aware inline command boundaries and `<:p...>` page breaks.

All documents are invented and built in memory by ``_sam`` (a minimal header with a
CP1250 ``[charset]`` declaration, no styles, and an ``[edoc]`` body), following
SAM-INLINE-NESTING-001 and SAM-INLINE-PAGEBREAK-001/002.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from io import BytesIO
from zipfile import ZipFile

import pytest

from amipro_sam.errors import PreservationLossError
from amipro_sam.model import PageBreak, Paragraph
from amipro_sam.parser import (
    _find_inline_command_close,
    _page_break_flag,
    parse_bytes,
)
from amipro_sam.renderers import RENDERERS

ODF_TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
ODF_STYLE = "urn:oasis:names:tc:opendocument:xmlns:style:1.0"
ODF_FO = "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _sam(body: str) -> bytes:
    source = (
        "[ver]\n\t4\n[sty]\n\t\n[charset]\n\t82\n\tANSI (Windows, IBM CP 1250)\n"
        "[edoc]\n" + body + "\n>\n"
    )
    return source.replace("\n", "\r\n").encode("ascii")


def _paragraphs(document) -> list[Paragraph]:
    return [block for block in document.blocks if isinstance(block, Paragraph)]


def _codes(document) -> list[str]:
    return [item.code for item in document.diagnostics]


def _families(document) -> list[str | None]:
    return [run.style.font_family for p in _paragraphs(document) for run in p.runs]


@pytest.mark.parametrize(
    "tag",
    [
        "+!",
        ":f240,Wingdings,",
        ':X3,-32768;if Defined x x else "R" endif',
        ":#1440,720",
        ":S+-1",
        ":I1440,0,720,0",
    ],
)
def test_close_without_nested_angle_matches_first_gt(tag: str) -> None:
    text = f"<{tag}>after"

    assert _find_inline_command_close(text, 0) == (text.find(">", 1), False)


def test_font_family_escape_decoded_in_codepage() -> None:
    document = parse_bytes(_sam("BEFORE<:f240,2Arial<\\a>,>AFTER"))

    assert [p.text for p in _paragraphs(document)] == ["BEFOREAFTER"]
    assert _families(document)[-1] == "Arialá"
    assert "inline-command-boundary-fallback" not in _codes(document)
    assert "unsupported-inline-tags" not in _codes(document)


def test_nested_page_number_field_is_one_command() -> None:
    # A valid field (SAM-INLINE-PAGENUM-001) whose leading text holds an escape: the
    # escape's ">" must not close the command, so no tail leaks into the text.
    document = parse_bytes(_sam("<:P11,1,Str<\\a>nka >TAIL"))
    (paragraph,) = _paragraphs(document)

    assert paragraph.text == "Stránka [#]TAIL"
    assert "inline-command-boundary-fallback" not in _codes(document)


def test_unterminated_nested_command_falls_back() -> None:
    document = parse_bytes(_sam("A<:p<* >B"))

    assert "B" in "".join(p.text for p in _paragraphs(document))
    assert _codes(document).count("inline-command-boundary-fallback") == 1


def test_bare_angle_in_payload_keeps_legacy_split() -> None:
    text = "<:fA<x>B>"

    assert _find_inline_command_close(text, 0) == (text.find(">", 1), False)
    assert "inline-command-boundary-fallback" not in _codes(parse_bytes(_sam(text)))


def test_doubled_angle_in_font_payload_is_strict_clean() -> None:
    document = parse_bytes(_sam("BEFORE<:f240,2A<<B,>AFTER"), strict=True)

    assert _families(document)[-1] == "A<B"


@pytest.mark.parametrize("pairs", [250, 253, 256])
def test_capped_doubled_angles_stay_strict_clean(pairs: int) -> None:
    document = parse_bytes(_sam("X<:f240,2" + "<<" * pairs + ",>Y"), strict=True)

    assert _families(document)[-1] == "<" * pairs
    assert "inline-command-boundary-fallback" not in _codes(document)


def test_capped_nested_escapes_are_diagnosed() -> None:
    document = parse_bytes(_sam("<:f240," + "<\\a>" * 200 + ",>TAIL"))

    assert "inline-command-boundary-fallback" in _codes(document)
    assert "".join(p.text for p in _paragraphs(document)).endswith("TAIL")


def test_nested_scan_is_bounded_on_large_input() -> None:
    document = parse_bytes(_sam(("<:f" + "<\\a>" * 100) * 2000))

    assert document is not None


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        (":p<* >", (0x00, "")),
        (":p@", (0x40, "")),
        (":p</@>Name", (0x80, "Name")),
        (":p ", (0x20, "")),
        (":pMYSTERY", None),
        (":p</@>", None),
    ],
)
def test_page_break_flag_decoding(tag: str, expected: tuple[int, str] | None) -> None:
    assert _page_break_flag(tag) == expected


def test_plain_page_break_after_text_emits_break_block() -> None:
    document = parse_bytes(_sam("FIRST<:p<* >>\n\nSECOND"), strict=True)
    kinds = [type(block).__name__ for block in document.blocks]

    assert kinds == ["Paragraph", "PageBreak", "Paragraph"]
    second = _paragraphs(document)[1]
    assert second.text == "SECOND"
    assert second.page_break_before is False
    assert "unsupported-inline-tags" not in _codes(document)


def test_page_break_at_paragraph_start_breaks_before() -> None:
    document = parse_bytes(_sam("FIRST\n\n<:p<* >>SECOND"), strict=True)

    assert [type(block).__name__ for block in document.blocks] == ["Paragraph", "Paragraph"]
    assert _paragraphs(document)[1].page_break_before is True


def test_page_break_alone_in_paragraph_emits_break_block() -> None:
    document = parse_bytes(_sam("FIRST\n\n<:p<* >>\n\nSECOND"), strict=True)

    assert [type(block).__name__ for block in document.blocks] == [
        "Paragraph",
        "PageBreak",
        "Paragraph",
    ]


@pytest.mark.parametrize("tag", ["<:p@>", "<:p</@>Name>", "<:p >"])
def test_flagged_page_breaks_keep_unapplied_semantics(tag: str) -> None:
    document = parse_bytes(_sam(f"FIRST{tag}\n\nSECOND"))

    assert any(isinstance(block, PageBreak) for block in document.blocks)
    assert "Unsupported" not in "".join(p.text for p in _paragraphs(document))
    unapplied = [
        item for item in document.diagnostics if item.code == "inline-command-semantics-unapplied"
    ]
    assert unapplied
    with pytest.raises(PreservationLossError):
        parse_bytes(_sam(f"FIRST{tag}\n\nSECOND"), strict=True)


def test_malformed_page_break_payload_stays_unsupported() -> None:
    document = parse_bytes(_sam("A<:pMYSTERY>B"))

    assert "unsupported-inline-tags" in _codes(document)


def test_text_render_orders_break_after_first_paragraph() -> None:
    text = RENDERERS["text"](parse_bytes(_sam("FIRST<:p<* >>\n\nSECOND"))).decode("utf-8")

    assert text.index("FIRST") < text.index("\f") < text.index("SECOND")


def test_mid_paragraph_break_is_reported() -> None:
    document = parse_bytes(_sam("FIRST<:p<* >>SECOND"))

    assert [p.text for p in _paragraphs(document)] == ["FIRSTSECOND"]
    assert isinstance(document.blocks[-1], PageBreak)
    assert "page-break-mid-paragraph-approximated" in _codes(document)
    with pytest.raises(PreservationLossError):
        parse_bytes(_sam("FIRST<:p<* >>SECOND"), strict=True)


def _odt_sequence(payload: bytes) -> list[str]:
    with ZipFile(BytesIO(payload)) as archive:
        root = ET.fromstring(archive.read("content.xml"))
    breaking = {"PageBreak"}
    for style in root.iter(f"{{{ODF_STYLE}}}style"):
        for properties in style.iter(f"{{{ODF_STYLE}}}paragraph-properties"):
            if properties.attrib.get(f"{{{ODF_FO}}}break-before") == "page":
                breaking.add(style.attrib[f"{{{ODF_STYLE}}}name"])
    sequence = []
    for element in root.iter(f"{{{ODF_TEXT}}}p"):
        if element.attrib.get(f"{{{ODF_TEXT}}}style-name") in breaking:
            sequence.append("BREAK")
        text = "".join(element.itertext())
        if text:
            sequence.append(text)
    return sequence


def _docx_sequence(payload: bytes) -> list[str]:
    with ZipFile(BytesIO(payload)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    sequence = []
    for element in root.iter():
        page_break = (
            element.tag == f"{{{WORD}}}br" and element.attrib.get(f"{{{WORD}}}type") == "page"
        )
        if page_break or element.tag == f"{{{WORD}}}pageBreakBefore":
            sequence.append("BREAK")
        elif element.tag == f"{{{WORD}}}t" and element.text:
            sequence.append(element.text)
    return sequence


def _between(sequence: list[str]) -> bool:
    first = next(i for i, item in enumerate(sequence) if "FIRST" in item)
    second = next(i for i, item in enumerate(sequence) if "SECOND" in item)
    return "BREAK" in sequence[first + 1 : second]


@pytest.mark.parametrize(
    "body", ["FIRST<:p<* >>\n\nSECOND", "FIRST\n\n<:p<* >>SECOND"], ids=["after", "before"]
)
def test_break_placement_in_paginated_renderers(body: str) -> None:
    document = parse_bytes(_sam(body))

    assert len(re.findall(rb"/Type\s*/Page\b", RENDERERS["pdf"](document))) == 2
    assert _between(_odt_sequence(RENDERERS["odt"](document)))
    pytest.importorskip("docx")
    assert _between(_docx_sequence(RENDERERS["docx"](document)))


def test_break_at_document_end_is_trimmed_without_loss() -> None:
    document = parse_bytes(_sam("ONLY<:p<* >>"), strict=True)

    assert isinstance(document.blocks[-1], PageBreak)
    assert len(re.findall(rb"/Type\s*/Page\b", RENDERERS["pdf"](document))) == 1

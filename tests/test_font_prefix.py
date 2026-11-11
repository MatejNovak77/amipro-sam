"""`<:f…>` family fields start with a pitch-and-family prefix (SAM-INLINE-FONT-PREFIX-001).

All documents are invented and built in memory with a CP1250 `[charset]` declaration.
"""

from __future__ import annotations

import json
from io import BytesIO
from zipfile import ZipFile

import pytest

from amipro_sam.model import Paragraph
from amipro_sam.parser import parse_bytes
from amipro_sam.renderers import RENDERERS


def _sam(body: str, *, extra: str = "") -> bytes:
    text = (
        "[ver]\n\t4\n[sty]\n\t\n[charset]\n\t82\n\tANSI (Windows, IBM CP 1250)\n"
        + extra
        + "[edoc]\n"
        + body
        + "\n>\n"
    )
    return text.replace("\n", "\r\n").encode("cp1250")


def _family_after(field: str, *, strict: bool = False):
    document = parse_bytes(_sam(f"A<:f240,{field},>B"), strict=strict)
    paragraph = next(block for block in document.blocks if isinstance(block, Paragraph))
    run = next(run for run in paragraph.runs if run.text.endswith("B"))
    codes = [item.code for item in document.diagnostics]
    return run.style.font_family, codes


@pytest.mark.parametrize(
    ("field", "face"),
    [
        ('"GaramondE', "GaramondE"),
        ("2Times New Roman CE", "Times New Roman CE"),
        ("BArial CE", "Arial CE"),
        ("QCourier New CE", "Courier New CE"),
        ("bScript", "Script"),
        (" Courier New CE", "Courier New CE"),
        ("RModern", "Modern"),
    ],
)
def test_every_corpus_prefix_is_dropped(field: str, face: str) -> None:
    family, codes = _family_after(field, strict=True)

    assert family == face
    assert "inline-font-prefix-unrecognized" not in codes


def test_only_one_character_is_dropped() -> None:
    # A rule that strips every leading pitch-and-family character would yield
    # "rial CE" and "oman"; the prefix is exactly one positional character.
    assert _family_after("BArial CE")[0] == "Arial CE"
    assert _family_after("2Roman")[0] == "Roman"


def test_non_digit_prefix_is_dropped_too() -> None:
    # The historical digit-only rule left '"GaramondE' unchanged.
    assert _family_after('"GaramondE')[0] == "GaramondE"


@pytest.mark.parametrize(("field", "face"), [("NArial", "Arial"), ("Wingdings", "ingdings")])
def test_unobserved_prefix_is_dropped_with_advisory_note(field: str, face: str) -> None:
    # 'N' is reported by AmiPro-Linux; neither prefix is in the observed set, yet the
    # field position still holds the prefix, and strict parsing must not reject it.
    family, codes = _family_after(field, strict=True)

    assert family == face
    assert "inline-font-prefix-unrecognized" in codes


def test_prefix_is_split_before_unescaping() -> None:
    assert _family_after("@@Face")[0] == "@Face"
    assert _family_after("B<\\a>rial")[0] == "árial"
    assert _family_after("B@@Face")[0] == "@Face"


def test_escaped_prefix_is_consumed_whole() -> None:
    family, codes = _family_after("<\\a>Arial", strict=True)

    assert family == "Arial"
    assert "inline-font-prefix-unrecognized" in codes


@pytest.mark.parametrize("field", ['"', "<\\a>"])
def test_prefix_only_field_keeps_style_font(field: str) -> None:
    style = (
        "[tag]\n\tBody Text\n\t2\n\t[fnt]\n\t\tStyleFace\n\t\t240\n\t\t0\n\t\t49152\n"
        "\t[algn]\n\t\t1\n\t\t1\n\t\t0\n\t\t288\n\t\t288\n"
        "\t[spc]\n\t\t1\n\t\t273\n\t\t1\n\t\t0\n\t\t0\n\t\t1\n\t\t100\n\tBody Text\n\t0\n\t0\n"
    )
    document = parse_bytes(_sam(f"A<:f240,{field},>B", extra=style))
    paragraph = next(block for block in document.blocks if isinstance(block, Paragraph))

    assert paragraph.runs[-1].style.font_family == "StyleFace"
    assert paragraph.runs[-1].style.font_size_pt == 12.0
    assert "Unsupported" not in paragraph.text


def test_empty_family_still_restores_style_font() -> None:
    assert _family_after("")[0] is None


def test_face_length_cap_applies_after_the_prefix() -> None:
    assert _family_after("B" + "x" * 256)[0] == "x" * 256
    document = parse_bytes(_sam("A<:f240,B" + "x" * 257 + ",>B"))
    assert "Unsupported inline command" in document.blocks[0].text


def test_fallback_truncated_field_is_diagnosed_not_crashed() -> None:
    document = parse_bytes(_sam("A<:f240,B" + "<\\a>" * 200 + ",>TAIL"))

    assert "inline-command-boundary-fallback" in [item.code for item in document.diagnostics]
    assert document.blocks[0].text.endswith("TAIL")


def test_fnt_style_family_is_not_stripped() -> None:
    style = (
        "[tag]\n\tBody Text\n\t2\n\t[fnt]\n\t\tArial CE\n\t\t240\n\t\t0\n\t\t49152\n"
        "\t[algn]\n\t\t1\n\t\t1\n\t\t0\n\t\t288\n\t\t288\n"
        "\t[spc]\n\t\t1\n\t\t273\n\t\t1\n\t\t0\n\t\t0\n\t\t1\n\t\t100\n\tBody Text\n\t0\n\t0\n"
    )
    document = parse_bytes(_sam("text", extra=style))

    assert document.styles["Body Text"].character.font_family == "Arial CE"


def test_odt_docx_and_json_carry_the_bare_family() -> None:
    document = parse_bytes(_sam("A<:f240,BArial CE,>B"))

    with ZipFile(BytesIO(RENDERERS["odt"](document))) as archive:
        content = archive.read("content.xml").decode("utf-8")
    assert "Arial CE" in content and "BArial CE" not in content

    payload = json.loads(RENDERERS["json"](document).decode("utf-8"))
    families = set()

    def collect(node: object) -> None:
        if isinstance(node, dict):
            if "font_family" in node and node["font_family"] is not None:
                families.add(node["font_family"])
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(payload)
    # The raw source text is preserved elsewhere in the dump; the decoded field is bare.
    assert "Arial CE" in families and "BArial CE" not in families

    pytest.importorskip("docx")
    with ZipFile(BytesIO(RENDERERS["docx"](document))) as archive:
        word = archive.read("word/document.xml").decode("utf-8")
    assert "Arial CE" in word and "BArial CE" not in word

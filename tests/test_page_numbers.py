"""Page-number fields `<:PSP,N,LEAD>` (SAM-INLINE-PAGENUM-001..004).

All documents are invented and built in memory. Layout fixtures follow the nine-field
`[rght]` shape used by tests/test_geometry.py.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from io import BytesIO
from zipfile import ZipFile

import pytest

from amipro_sam.errors import PreservationLossError
from amipro_sam.model import (
    Footer,
    PageNumberRun,
    Paragraph,
    document_page_number_restart,
    format_page_number,
    page_number_text,
    page_number_value,
)
from amipro_sam.parser import parse_bytes
from amipro_sam.renderers import RENDERERS
from amipro_sam.renderers import pdf as pdf_renderer

ODF_TEXT = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
ODF_STYLE = "urn:oasis:names:tc:opendocument:xmlns:style:1.0"
ODF_TABLE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
VARIANT = "\n".join(("15840", "12240", "37", "720", "900", "1", "1080", "540", "19"))


def _sam(body: str, *, footer: str | None = None, left: str = "720", bottom: str = "900") -> bytes:
    layout = ""
    if footer is not None:
        variant = VARIANT.replace("\t720\n", "").split("\n")
        variant[3], variant[4] = left, bottom
        geometry = "\n\t\t".join(variant)
        layout = (
            f"[lay]\n\tInvented\n\t1284\n\t[rght]\n\t\t{geometry}\n"
            f"\t[frght]\n\t\t[txt]\n\t\t\t{footer}\n\t\t>\n"
            f"\t[flft]\n\t\t[txt]\n\t\t\t{footer}\n\t\t>\n[elay]\n"
        )
    text = "[ver]\n\t4\n[sty]\n\t\n" + layout + "[edoc]\n" + body + "\n>\n"
    return text.replace("\n", "\r\n").encode("cp1252")


THREE_PAGES = "one<:p<* >>\n\ntwo<:p<* >>\n\nthree"


def _fields(document) -> list[PageNumberRun]:
    found = []
    for block in document.blocks:
        paragraphs = [block] if isinstance(block, Paragraph) else getattr(block, "blocks", [])
        for paragraph in paragraphs:
            found.extend(r for r in getattr(paragraph, "runs", []) if isinstance(r, PageNumberRun))
    return found


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("<:P11,2,>", ("1", 1, 2, "")),
        ("<:P10,0,>", ("1", 0, 0, "")),
        ("<:Pi1,1,Page >", ("i", 1, 1, "Page ")),
        ("<:PA0,5,str<\\a>nka >", ("A", 0, 5, "stránka ")),
        ("<:P19999,9999,>", ("1", 9999, 9999, "")),
    ],
)
def test_field_grammar(tag: str, expected: tuple[str, int, int, str]) -> None:
    document = parse_bytes(_sam(f"A{tag}B"))
    (field,) = _fields(document)

    assert (
        field.number_style,
        field.start_page,
        field.start_number,
        field.leading_text,
    ) == expected
    assert document.blocks[0].text == f"A{expected[3]}[#]B"
    assert "unsupported-inline-tags" not in [item.code for item in document.diagnostics]


@pytest.mark.parametrize(
    "tag",
    ["<:Pq1,1,>", "<:P1,1>", "<:P110000,1,>", "<:P11,10000,>", "<:P11,1," + "x" * 256 + ">"],
)
def test_malformed_fields_stay_unsupported(tag: str) -> None:
    document = parse_bytes(_sam(f"A{tag}B"))

    assert _fields(document) == []
    assert "unsupported-inline-tags" in [item.code for item in document.diagnostics]
    assert "[Page number]" not in document.blocks[0].text


def test_leading_text_at_cap_is_accepted() -> None:
    document = parse_bytes(_sam("<:P11,1," + "x" * 255 + ">"))

    assert len(_fields(document)[0].leading_text) == 255


@pytest.mark.parametrize(
    ("value", "style", "expected"),
    [
        (4, "1", "4"),
        (4, "I", "IV"),
        (14, "i", "xiv"),
        (3888, "I", "MMMDCCCLXXXVIII"),
        (3999, "I", "MMMCMXCIX"),
        (4000, "I", "4000"),
        (1, "A", "A"),
        (26, "a", "z"),
        (27, "a", "aa"),
        (53, "A", "AAA"),
        (0, "I", "0"),
    ],
)
def test_format_page_number(value: int, style: str, expected: str) -> None:
    assert format_page_number(value, style) == expected


@pytest.mark.parametrize(
    ("start_page", "start_number", "values"),
    [
        (0, 0, [1, 2, 3]),  # N = 0: ordinary numbering
        (1, 0, [1, 2, 3]),
        (0, 5, [5, 6, 7]),  # P = 0: p + N - 1
        (1, 5, [5, 6, 7]),
        (2, 5, [None, 5, 6]),  # nothing before the start page
        (2, 0, [None, 2, 3]),
    ],
)
def test_displayed_value(start_page: int, start_number: int, values: list[int | None]) -> None:
    run = PageNumberRun("[#]", start_page=start_page, start_number=start_number)

    assert [page_number_value(run, page) for page in (1, 2, 3)] == values
    assert page_number_text(run, 1) == ("" if values[0] is None else str(values[0]))


def test_restart_and_unsupported_offsets() -> None:
    assert document_page_number_restart(parse_bytes(_sam("x<:P11,5,>")).blocks) == 5
    assert document_page_number_restart(parse_bytes(_sam("x<:P10,0,>")).blocks) is None

    conflicting = parse_bytes(_sam("x<:P11,5,>y<:P11,7,>"))
    hidden_start = parse_bytes(_sam("x<:P12,5,>"))
    for document in (conflicting, hidden_start):
        assert document_page_number_restart(document.blocks) is None
        assert "page-number-offset-unsupported" in [d.code for d in document.diagnostics]
    with pytest.raises(PreservationLossError):
        parse_bytes(_sam("x<:P12,5,>"), strict=True)


def test_valid_field_is_strict_clean() -> None:
    document = parse_bytes(_sam("x<:P11,5,str. >y"), strict=True)

    assert _fields(document)[0].leading_text == "str. "


def test_field_stays_a_separate_run_and_survives_copy() -> None:
    document = parse_bytes(_sam("same<:P11,1,>same<:P11,1,><:P11,1,>"))
    runs = document.blocks[0].runs

    assert [type(run).__name__ for run in runs] == [
        "TextRun",
        "PageNumberRun",
        "TextRun",
        "PageNumberRun",
        "PageNumberRun",
    ]
    clone = copy.copy(runs[1])
    assert isinstance(clone, PageNumberRun) and clone == runs[1]


def test_json_keeps_typed_field() -> None:
    payload = json.loads(RENDERERS["json"](parse_bytes(_sam("x<:Pi1,3,p. >"))).decode("utf-8"))

    assert '"type": "PageNumberRun"' in json.dumps(payload)
    assert '"leading_text": "p. "' in json.dumps(payload)


@pytest.mark.parametrize("name", ["text", "markdown", "html"])
def test_textual_outputs_show_placeholder(name: str) -> None:
    output = RENDERERS[name](parse_bytes(_sam("x<:P11,5,str. >y"))).decode("utf-8")
    # Markdown escapes the brackets; it still displays as "str. [#]".
    placeholder = "str. \\[#\\]" if name == "markdown" else "str. [#]"

    assert placeholder in output and "Page number" not in output


def test_furniture_lines_substitute_real_numbers() -> None:
    document = parse_bytes(_sam(THREE_PAGES, footer="<:P11,158,str. >"))
    footer = next(block for block in document.blocks if isinstance(block, Footer))

    assert pdf_renderer._furniture_lines(footer, page_number=1) == ["str. 158"]
    assert pdf_renderer._furniture_lines(footer, page_number=2) == ["str. 159"]
    hidden = next(
        block
        for block in parse_bytes(_sam(THREE_PAGES, footer="<:P12,5,>")).blocks
        if isinstance(block, Footer)
    )
    assert pdf_renderer._furniture_lines(hidden, page_number=1) == []
    assert pdf_renderer._furniture_lines(hidden, page_number=2) == ["5"]


def test_widest_rendering_is_measured_not_largest() -> None:
    widest = pdf_renderer._widest_page_number("I", 1, 3880, "")
    candidates = [format_page_number(value, "I") for value in range(3880, 3880 + 128)]

    assert widest in candidates
    assert widest != candidates[-1]  # the largest value is not the widest numeral
    assert widest == "MMMDCCCLXXXVIII"


def _pdf_pages(payload: bytes, tmp_path) -> list[str]:
    path = tmp_path / "out.pdf"
    path.write_bytes(payload)
    text = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True, check=True
    ).stdout
    return [page for page in text.split("\f") if page.strip()]


poppler = pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext absent")


@poppler
def test_pdf_furniture_shows_real_numbers(tmp_path) -> None:
    pages = _pdf_pages(
        RENDERERS["pdf"](parse_bytes(_sam(THREE_PAGES, footer="F <:P11,5,str. >"))), tmp_path
    )

    assert [re.findall(r"str\. (\d+)", page) for page in pages] == [["5"], ["6"], ["7"]]


@poppler
def test_narrow_margin_never_clips_the_number(tmp_path) -> None:
    # 72 pt body width (left margin eats the page) and a 13 pt footer margin.
    document = parse_bytes(
        _sam(THREE_PAGES, footer="xxxxxxxxxxx <:PI1,3880,>", left="11160", bottom="260")
    )
    pages = _pdf_pages(RENDERERS["pdf"](document), tmp_path)
    joined = "\n".join(pages)

    for value in (3880, 3881, 3882):
        assert format_page_number(value, "I") in joined


@poppler
def test_in_flow_footer_shows_its_page_number(tmp_path) -> None:
    layout_without_geometry = (
        "[lay]\n\tInvented\n\t0\n\t[frght]\n\t\t[txt]\n\t\t\tF <:P11,5,str. >\n\t\t>\n"
    )
    text = "[ver]\n\t4\n[sty]\n\t\n" + layout_without_geometry + "[edoc]\n" + THREE_PAGES + "\n>\n"
    document = parse_bytes(text.replace("\n", "\r\n").encode("cp1252"))
    pages = _pdf_pages(RENDERERS["pdf"](document), tmp_path)

    assert "str. 7" in pages[-1]
    assert "[#]" not in "".join(pages)


def _zip_xml(payload: bytes, name: str) -> ET.Element:
    with ZipFile(BytesIO(payload)) as archive:
        return ET.fromstring(archive.read(name))


def test_odt_native_field_and_bound_restart() -> None:
    payload = RENDERERS["odt"](parse_bytes(_sam(THREE_PAGES, footer="F <:Pi1,5,str. >")))
    styles = _zip_xml(payload, "styles.xml")
    content = _zip_xml(payload, "content.xml")

    field = styles.find(f".//{{{ODF_TEXT}}}page-number")
    assert field is not None
    assert field.attrib[f"{{{ODF_STYLE}}}num-format"] == "i"
    assert field.attrib[f"{{{ODF_TEXT}}}select-page"] == "current"
    body = content.find(".//{urn:oasis:names:tc:opendocument:xmlns:office:1.0}text")
    first = list(body)[0]
    style_name = first.attrib[f"{{{ODF_TEXT}}}style-name"]
    style = next(
        node
        for node in content.iter(f"{{{ODF_STYLE}}}style")
        if node.attrib[f"{{{ODF_STYLE}}}name"] == style_name
    )
    assert style.attrib[f"{{{ODF_STYLE}}}master-page-name"] == "Standard"
    properties = style.find(f"{{{ODF_STYLE}}}paragraph-properties")
    assert properties.attrib[f"{{{ODF_STYLE}}}page-number"] == "5"
    users = [
        p
        for p in body.iter(f"{{{ODF_TEXT}}}p")
        if p.attrib.get(f"{{{ODF_TEXT}}}style-name") == style_name
    ]
    assert len(users) == 1  # one-use: a shared master-page style would restart at every use


def test_odt_without_offset_has_no_restart() -> None:
    payload = RENDERERS["odt"](parse_bytes(_sam(THREE_PAGES, footer="F <:P10,0,>")))
    content = _zip_xml(payload, "content.xml")

    assert not [
        node
        for node in content.iter(f"{{{ODF_STYLE}}}paragraph-properties")
        if f"{{{ODF_STYLE}}}page-number" in node.attrib
    ]


def test_docx_page_field_and_section_start() -> None:
    pytest.importorskip("docx")
    payload = RENDERERS["docx"](parse_bytes(_sam(THREE_PAGES, footer="F <:PI1,157,>")))
    document = _zip_xml(payload, "word/document.xml")
    footers = []
    with ZipFile(BytesIO(payload)) as archive:
        footers = [
            ET.fromstring(archive.read(n))
            for n in archive.namelist()
            if n.startswith("word/footer")
        ]

    instructions = [
        node.attrib[f"{{{WORD}}}instr"]
        for root in footers
        for node in root.iter(f"{{{WORD}}}fldSimple")
    ]
    assert instructions and all(value.strip() == "PAGE \\* ROMAN" for value in instructions)
    number_type = document.find(f".//{{{WORD}}}sectPr/{{{WORD}}}pgNumType")
    assert number_type is not None and number_type.attrib[f"{{{WORD}}}start"] == "157"
    children = [child.tag.split("}")[1] for child in document.find(f".//{{{WORD}}}sectPr")]
    if "cols" in children:
        assert children.index("pgNumType") < children.index("cols")


def test_docx_conflicting_offsets_get_no_restart() -> None:
    pytest.importorskip("docx")
    payload = RENDERERS["docx"](parse_bytes(_sam("x<:P11,5,>y<:P11,7,>")))

    assert _zip_xml(payload, "word/document.xml").find(f".//{{{WORD}}}pgNumType") is None


def test_record_budget_is_charged_for_fields() -> None:
    from amipro_sam.errors import ResourceLimitError
    from amipro_sam.limits import ParseLimits

    with pytest.raises(ResourceLimitError):
        parse_bytes(_sam("<:P11,1,>" * 50), limits=ParseLimits(max_records=10))

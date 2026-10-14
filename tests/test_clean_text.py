from clean_text import (
    clean_page_text,
    cut_references,
    find_header_footer_patterns,
    is_boilerplate,
    normalize,
    remove_headers_footers,
)


def make_pages(texts):
    return [{"source": "doc.pdf", "page": i + 1, "text": t} for i, t in enumerate(texts)]


def test_normalize_replaces_every_number_with_zero():
    assert normalize("NASA/TM-2010-216736 9") == "NASA/TM-0-0 0"


def test_cut_references_cuts_after_heading_in_second_half():
    pages = make_pages(["intro", "body", "Conclusions\nReferences\n1. Smith", "after"])
    kept, page_number = cut_references(pages)
    assert page_number == 3
    assert len(kept) == 3
    assert kept[2]["text"] == "Conclusions"


def test_cut_references_ignores_heading_in_first_half():
    pages = make_pages(["References\nlisted in the table of contents", "body", "more", "end"])
    kept, page_number = cut_references(pages)
    assert page_number is None
    assert len(kept) == 4


def test_is_boilerplate_detects_known_marker():
    assert is_boilerplate("blah REPORT   DOCUMENTATION\nPAGE blah")
    assert not is_boilerplate("A normal paragraph about turbines.")


def test_clean_page_text_joins_hyphen_split_words():
    text, removed, joined = clean_page_text("Test- \ning of the system", set())
    assert "Testing" in text
    assert joined == 1


def test_clean_page_text_keeps_capitalised_line_separate():
    text, _, _ = clean_page_text("The first sentence ends here.\nNew Heading", set())
    assert text.splitlines() == ["The first sentence ends here.", "New Heading"]


def test_clean_page_text_joins_line_that_continues_a_sentence():
    text, _, _ = clean_page_text("The blade grows\nunder load.", set())
    assert text == "The blade grows under load."


def test_repeated_header_is_found_and_removed():
    pages = make_pages(
        ["REPORT 12 3\nbody one", "REPORT 12 4\nbody two", "REPORT 12 5\nbody three"]
    )
    patterns = find_header_footer_patterns(pages)
    assert "REPORT 0 0" in patterns
    cleaned, removed = remove_headers_footers(pages[0]["text"], patterns)
    assert removed == 1
    assert cleaned == "body one"
from chunk_text import build_chunks, chunk_sentences, chunk_words


def numbered_words(count):
    return " ".join(f"w{i}" for i in range(count))


def test_chunk_words_respects_size_and_overlap():
    chunks = chunk_words(numbered_words(300), size=100, overlap=20)
    first = chunks[0].split()
    second = chunks[1].split()
    assert len(first) == 100
    assert second[0] == "w80"
    assert all(len(c.split()) <= 100 for c in chunks)
    assert chunks[-1].split()[-1] == "w299"


def test_chunk_words_empty_text_gives_no_chunks():
    assert chunk_words("", size=100, overlap=20) == []


def test_chunk_sentences_never_exceeds_size():
    sentence = " ".join(["word" for _ in range(19)]) + "."
    text = "\n".join([sentence for _ in range(12)])
    chunks = chunk_sentences(text, size=60, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 60 for c in chunks)


def test_chunk_sentences_repeats_last_sentence_as_overlap():
    lines = [f"Sentence number {i} has exactly seven words." for i in range(8)]
    chunks = chunk_sentences("\n".join(lines), size=25, overlap=10)
    assert len(chunks) > 1
    last_sentence_of_first = chunks[0].split(". ")[-1]
    assert last_sentence_of_first.strip(".") in chunks[1]


def test_build_chunks_keeps_source_page_and_builds_ids():
    pages = [{"source": "report.pdf", "page": 7, "text": numbered_words(40) + "."}]
    chunks = build_chunks(pages, "words")
    assert chunks[0]["chunk_id"] == "report_p7_c1"
    assert chunks[0]["page"] == 7
    assert chunks[0]["source"] == "report.pdf"


def test_build_chunks_drops_chunks_that_are_too_short():
    pages = [{"source": "report.pdf", "page": 1, "text": "Just a few words here."}]
    assert build_chunks(pages, "words") == []
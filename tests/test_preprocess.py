from docrouter.preprocess import MAX_CHARS, clean_text


def test_removes_source_markers():
    text = "On XX/XX/2019 my card XXXX was charged. Order {{Order Number}} at www.shop.com"
    out = clean_text(text)
    assert "XXXX" not in out and "XX/XX" not in out
    assert "{{" not in out and "www." not in out


def test_whitespace_and_length():
    assert clean_text("  a \n\n b\t c ") == "a b c"
    assert len(clean_text("word " * 5000)) <= MAX_CHARS


def test_handles_none():
    assert clean_text(None) == ""

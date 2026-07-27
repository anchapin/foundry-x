from target import repeat


def test_repeat_basic():
    assert repeat("hi", 3) == "hi, hi, hi"


def test_repeat_zero():
    assert repeat("hello", 0) == ""


def test_repeat_negative():
    with pytest.raises(ValueError, match="non-negative"):
        repeat("test", -1)


def test_repeat_one():
    assert repeat("x", 1) == "x"


def test_repeat_longer():
    result = repeat("ab", 4)
    assert result == "ab, ab, ab, ab"

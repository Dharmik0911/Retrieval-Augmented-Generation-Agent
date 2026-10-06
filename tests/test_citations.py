import pytest
from agent import Claim, Draft, render

DOCS = [{"id": 2, "source": "report.pdf", "page": 4, "status": "preliminary"}]


def test_valid_citation():
    result = render(Draft(claims=[Claim(text="Example supported statement.", source_ids=[2])],
                          limitations=""), DOCS)
    assert "[S2]" in result
    assert "PDF page 4 (preliminary)" in result


def test_unknown_citation_rejected():
    with pytest.raises(ValueError):
        render(Draft(claims=[Claim(text="Example", source_ids=[99])], limitations=""), DOCS)


def test_missing_citation_rejected():
    with pytest.raises(ValueError):
        render(Draft(claims=[Claim(text="Example", source_ids=[])], limitations=""), DOCS)


def test_empty_answer_abstains():
    assert "sufficient evidence" in render(Draft(claims=[], limitations=""), DOCS)

from src.latex_check import (
    check_braces,
    check_dollar_delimiters,
    check_environments,
    check_page,
    count_unclear,
)


# --- dollar delimiters ---

def test_paired_inline_math_passes():
    assert check_dollar_delimiters("The value $x^2$ is positive.") == []


def test_paired_display_math_passes():
    assert check_dollar_delimiters("$$K_r(j)=\\sum_i x_i$$") == []


def test_unclosed_inline_dollar_fails():
    assert len(check_dollar_delimiters("The value $x^2 is positive.")) == 1


def test_unclosed_display_dollar_fails():
    assert len(check_dollar_delimiters("$$x = 1 + y")) == 1


# --- braces ---

def test_balanced_braces_in_math_pass():
    assert check_braces("$\\binom{j}{i}$") == []


def test_unbalanced_braces_in_math_fail():
    assert len(check_braces("$\\binom{j}{i$")) == 1


def test_braces_in_prose_are_not_checked():
    assert check_braces("A stray { brace in prose is not math.") == []


# --- environments ---

def test_matched_environment_passes():
    assert check_environments("\\begin{pmatrix} a \\end{pmatrix}") == []


def test_unmatched_begin_fails():
    assert len(check_environments("\\begin{pmatrix} a")) == 1


# --- unclear marks ---

def test_count_unclear_marks():
    assert count_unclear("x [UNCLEAR] y [UNCLEAR]") == 2
    assert count_unclear("no marks here") == 0


# --- whole page ---

def test_clean_page_is_ok():
    report = check_page("Plain text with $x$ and $$y = 2$$.")
    assert report == {"ok": True, "failures": [], "unclear_count": 0}


def test_broken_page_is_flagged_not_ok():
    report = check_page("The value $x^2 is here. [UNCLEAR]")
    assert report["ok"] is False
    assert report["unclear_count"] == 1
    assert len(report["failures"]) >= 1

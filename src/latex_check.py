"""Deterministic checks on one page of transcribed Markdown. No LLM, no API calls.

A page FAILS if any check finds a problem. Failing pages are flagged in
results/ingest_report.json and must not be used silently.
"""

import re

# A math span is either $$...$$ (display) or $...$ (inline). The $$ branch comes
# first, so a display block is never read as two inline spans.
MATH_SPAN = re.compile(r"\$\$(.*?)\$\$|\$([^$]*?)\$", re.DOTALL)
ENV_TAG = re.compile(r"\\(begin|end)\{([^}]*)\}")


def check_dollar_delimiters(text: str) -> list[str]:
    """Return a failure message for each unpaired $ or $$ in the text.

    Inline math is $...$. Display math is $$...$$. Every opening delimiter
    needs a closing one of the same kind.
    """
    failures = []
    display = text.count("$$")
    inline = text.replace("$$", "").count("$")  # remove display pairs, count what is left
    if display % 2:
        failures.append(f"unpaired $$ (found {display})")
    if inline % 2:
        failures.append(f"unpaired $ (found {inline})")
    return failures


def check_braces(text: str) -> list[str]:
    """Return a failure message if { and } do not balance inside the math.

    Braces in plain prose are not checked. Only braces inside $...$ or $$...$$.
    """
    failures = []
    for match in MATH_SPAN.finditer(text):
        span = match.group(1) if match.group(1) is not None else match.group(2)
        opens, closes = span.count("{"), span.count("}")
        if opens != closes:
            failures.append(f"unbalanced braces in math: {span.strip()!r}")
    return failures


def check_environments(text: str) -> list[str]:
    """Return a failure message for each \\begin{X} without a matching \\end{X}, and the reverse."""
    failures = []
    stack = []  # names of environments opened but not yet closed
    for kind, name in ENV_TAG.findall(text):
        if kind == "begin":
            stack.append(name)
        elif stack and stack[-1] == name:
            stack.pop()
        else:
            failures.append(f"\\end{{{name}}} without a matching \\begin")
    failures.extend(f"\\begin{{{name}}} never closed" for name in stack)
    return failures


def count_unclear(text: str) -> int:
    """Return how many times the [UNCLEAR] marker appears in the page."""
    return text.count("[UNCLEAR]")


def check_page(text: str) -> dict:
    """Run every check on one page. Return a report dict:

    {
        "ok": bool,              # True only when there are no failures
        "failures": [str, ...],  # all messages from the three check_* functions
        "unclear_count": int,
    }
    """
    failures = (
        check_dollar_delimiters(text)
        + check_braces(text)
        + check_environments(text)
    )
    return {
        "ok": not failures,
        "failures": failures,
        "unclear_count": count_unclear(text),
    }

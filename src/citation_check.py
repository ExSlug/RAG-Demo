"""Deterministic citation check on a model's answer.

No LLM, no API calls. It reads only the labels in the answer, not the meaning.

The answer cites sources with tags like  [FYP_notes.pdf p.14]  (one page per tag).
The check applies these rules (the project spec, section 10):

  missing_citation            the answer has no tag and is not exactly ABSTAIN
  unretrieved_citation        a tag points to a (file, page) that is not in the retrieved chunks
  answered_when_unanswerable  the question has no answer in the notes but the model answered

tests/test_citation_check.py has examples for each rule.
"""

import re

# The one place this phrase is defined. generate.py puts it in the prompt;
# the check compares against it. Never type it anywhere else.
ABSTAIN = "Not in the notes."


def extract_citations(answer: str) -> set[tuple[str, int]]:
    """Return {(file, page)} for every [file p.N] tag in the answer.

    Ignore text that only looks like a tag, such as [p.4] with no file name.
    """
    citations = set(re.findall(r'\[([^\]\s]+) p\.(\d+)\]', answer))
    return {(file, int(page)) for file, page in citations}


def check_citations(answer: str, retrieved_chunks: list[dict], answerable: bool) -> dict:
  """Check one answer. Return:

  {
      "ok": bool,                 # True when `failures` is empty
      "failures": [str, ...],     # names from the table above, each at most once
      "cited": [(file, page), ...]  # every page the answer cites, sorted
  }

  Rules, in plain words:
  - An answer that is exactly ABSTAIN (ignoring spaces and newlines at the ends) is an
    abstention. If answerable is False, that is correct. If answerable is True, the
    citation check passes it (a wrong refusal is measured somewhere else).
  - If answerable is False and the model did not abstain: failure answered_when_unanswerable.
  - If answerable is True and the model did not abstain:
      no tag at all                       -> missing_citation
      a tag not among the retrieved pages -> unretrieved_citation
    Each retrieved chunk has "file" and "pages" (list of ints).
  """
  stripped = answer.strip()
  ok = True
  failures = []
  cited = []


  if answerable:
    if stripped == ABSTAIN:
      # citation check passes it
      pass

    else: # not abstain
      # missing_citation
      answer_citation = extract_citations(answer)
      if answer_citation == set(): 
        ok = False
        failures = ["missing_citation"]

      answer_citation = sorted(answer_citation)
      
      # unretrieved_citation
      citation_set = set([])
      for chunk in retrieved_chunks:
        for page in chunk["pages"]:
          citation_set.update([(chunk["file"], int(page))])
      
      for tuple in answer_citation:
        if tuple not in citation_set:
          ok = False
          failures = ["unretrieved_citation"]
      cited = list(answer_citation)
      
  else: # not answerable
    if stripped == ABSTAIN:
      pass
    
    else: # not abstain
      ok = False
      failures = ["answered_when_unanswerable"]
      
       
  return {
    "ok": ok,
    "failures": failures,
    "cited": cited
  }

         
         
    
    

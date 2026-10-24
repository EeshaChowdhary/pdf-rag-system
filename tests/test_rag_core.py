"""Deterministic checks for retrieval, abstention, citations, and leakage.

These tests do not call Gemini and do not read or rewrite gold answers.
"""

import json
import unittest
from pathlib import Path

from ai_engine import spread_page_images
from evals.run_evals import is_hit, validate_dataset
from pdf_rag import (
    build_evidence_prompt,
    claims_outside_context,
    cited_pages,
    evidence_sufficient,
    looks_like_abstention,
    matches_required_facts,
    prompt_contains_reference,
    should_abstain,
    source_footer,
)
from utils import embed_chunks, retrieve_with_scores
from web_search import search_status

ROOT = Path(__file__).resolve().parents[1]


class DatasetTests(unittest.TestCase):
    def test_all_39_questions_have_existing_pdfs(self):
        questions = json.loads((ROOT / "evals" / "questions.json").read_text(encoding="utf-8"))
        validate_dataset(questions)
        self.assertEqual(len(questions), 39)

    def test_page_hit_uses_pdf_page_and_skips_missing_labels(self):
        hit = {"answerable": True, "page": 2, "source": "notes.pdf"}
        chunks = [
            {"page": 5, "source": "notes.pdf"},
            {"page": 2, "source": "notes.pdf"},
        ]
        self.assertTrue(is_hit(hit, chunks))
        self.assertFalse(is_hit(hit, [{"page": 1, "source": "notes.pdf"}]))
        unlabeled = {"answerable": True, "page": None, "source": "notes.pdf"}
        self.assertIsNone(is_hit(unlabeled, chunks))
        unanswerable = {"answerable": False, "page": 2, "source": "notes.pdf"}
        self.assertIsNone(is_hit(unanswerable, chunks))


class EvidenceTests(unittest.TestCase):
    def test_supported_fact_matches_and_incorrect_answer_does_not(self):
        context = "Heap is slow compare to stack and there is a chance of memory leak."
        supported = "The heap is slower than the stack and there is a chance of memory leak."
        incorrect = "Heap allocation is always faster than the stack."
        self.assertTrue(matches_required_facts(supported, ["memory leak"]))
        self.assertFalse(matches_required_facts(incorrect, ["memory leak"]))
        self.assertEqual(claims_outside_context(["memory leak"], context), [])
        self.assertEqual(
            claims_outside_context(["fragmented", "memory leak"], context),
            ["fragmented"],
        )

    def test_unanswerable_abstention(self):
        refusal = "The document has no mention of a pilot test or a user count."
        invented = "The pilot test had 25 users."
        self.assertTrue(looks_like_abstention(refusal))
        self.assertFalse(looks_like_abstention(invented))
        self.assertTrue(should_abstain("agent", [0.2], [], attempts=2))
        self.assertFalse(should_abstain("pdf", [0.2], [], attempts=2))
        self.assertFalse(should_abstain("agent", [0.8], [], attempts=1))
        self.assertFalse(should_abstain(
            "agent",
            [0.2],
            [{"title": "Paper", "url": "https://example.edu/paper"}],
            attempts=2,
        ))
        self.assertTrue(evidence_sufficient([0.2, 0.51]))
        self.assertFalse(evidence_sufficient([0.2, 0.4]))

    def test_page_citations_and_source_footer(self):
        self.assertEqual(cited_pages("Explained on page 40 and again on Page 41."), [40, 41])
        footer = source_footer([40, 41], [{"title": "Spec", "url": "https://example.edu/spec"}])
        self.assertIn("page 40", footer)
        self.assertIn("https://example.edu/spec", footer)

    def test_reference_answer_is_not_part_of_the_evidence_prompt(self):
        gold = "73.18%, so 75% is impossible."
        prompt = build_evidence_prompt(
            [{"text": "--- Page 40 ---\nmaximum attendance is 73.18%", "page": 40}],
            [],
        )
        self.assertFalse(prompt_contains_reference(prompt, gold))
        self.assertNotIn("Reference answer", prompt)

    def test_retrieval_ranks_the_relevant_chunk_first(self):
        chunks = [
            "banana bread recipe with sugar and butter",
            "graph neural network for money laundering detection",
        ]
        scored = retrieve_with_scores(
            "money laundering graph neural network",
            chunks,
            embed_chunks(chunks),
            top_k=1,
        )
        self.assertIn("laundering", scored[0][0])
        self.assertGreater(scored[0][1], 0.4)

    def test_scanned_fallback_is_not_the_first_four_pages_only(self):
        images = [{"page_num": number} for number in range(1, 11)]
        selected = [image["page_num"] for image in spread_page_images(images, limit=5)]
        self.assertEqual(selected[0], 1)
        self.assertEqual(selected[-1], 10)
        self.assertNotEqual(selected, [1, 2, 3, 4])

    def test_web_search_without_credentials_stays_unavailable(self):
        status = search_status()
        self.assertFalse(status["available"])
        self.assertIsNone(status["provider"])
        self.assertNotIn("GEMINI_API_KEY", json.dumps(status))


if __name__ == "__main__":
    unittest.main()

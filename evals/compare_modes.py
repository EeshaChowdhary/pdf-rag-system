"""Compare baseline PDF RAG, the improved PDF run, and agentic RAG.

Baseline results.json is read only. Agentic web search is skipped when no
search credential is configured. Gold answers are not sent to generation.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from evals.run_evals import is_hit, judge, load_document  # noqa: E402
from pdf_rag import AGENT, answer_from_index, looks_like_abstention  # noqa: E402
from web_search import search_status  # noqa: E402

REPRESENTATIVE = [1, 3, 14, 25, 30, 39]


def _metrics(rows):
    scored = [row for row in rows if row.get("correct") is not None]
    grounded = [row for row in scored if row.get("grounded") is not None]
    hits = [row for row in rows if row.get("hit") is not None]
    errors = [row for row in rows if row.get("error")]
    unscored = [row for row in rows if row.get("correct") is None and not row.get("error")]
    times = [row["secs"] for row in rows if isinstance(row.get("secs"), (int, float))]
    unanswerable = [row for row in rows if row.get("answerable") is False and not row.get("error")]

    def abstained(row):
        if "abstained" in row:
            return row.get("abstained") is True
        return looks_like_abstention(row.get("answer_given") or "")

    good_abstain = [
        row for row in unanswerable
        if abstained(row) and row.get("correct") is True
    ]
    tokens = 0
    token_rows = 0
    for row in rows:
        for key in ("answer_usage", "judge_usage"):
            usage = row.get(key) or {}
            if isinstance(usage.get("total_tokens"), int):
                tokens += usage["total_tokens"]
                token_rows += 1
    return {
        "n": len(rows),
        "correct": sum(row.get("correct") is True for row in scored),
        "scored": len(scored),
        "grounded": sum(row.get("grounded") is True for row in grounded),
        "grounded_n": len(grounded),
        "hits": sum(row.get("hit") is True for row in hits),
        "hit_n": len(hits),
        "errors": len(errors),
        "unscored": len(unscored),
        "latency": (sum(times) / len(times)) if times else None,
        "abstained_ok": len(good_abstain),
        "unanswerable": len(unanswerable),
        "tokens": tokens if token_rows else None,
    }


def _line(name, metrics):
    if metrics is None:
        return f"| {name} | not run | | | | | | |"

    def ratio(num, den):
        if not den:
            return "n/a"
        return f"{num}/{den} ({100 * num / den:.1f}%)"

    latency = f"{metrics['latency']:.2f}s" if metrics["latency"] is not None else "n/a"
    tokens = str(metrics["tokens"]) if metrics["tokens"] is not None else "unavailable"
    return (
        f"| {name} | {metrics['n']} | {ratio(metrics['correct'], metrics['scored'])} | "
        f"{ratio(metrics['grounded'], metrics['grounded_n'])} | "
        f"{ratio(metrics['hits'], metrics['hit_n'])} | "
        f"{ratio(metrics['abstained_ok'], metrics['unanswerable'])} | "
        f"{latency} | errors {metrics['errors']}, unscored {metrics['unscored']}, tokens {tokens} |"
    )


def run_agentic(questions):
    status = search_status()
    rows = []
    for number in REPRESENTATIVE:
        item = questions[number - 1]
        record = dict(item)
        record["question_number"] = number
        started = time.time()
        try:
            document = load_document(item["source"])
            result = answer_from_index(
                item["q"],
                document["chunks"],
                document["embeddings"],
                document["page_images"],
                document["scanned"],
                AGENT,
            )
            record["secs"] = round(time.time() - started, 2)
            record["answer_given"] = result["answer"]
            record["retrieved_pages"] = result["pages"]
            record["abstained"] = result["abstained"] or looks_like_abstention(result["answer"])
            record["web_available"] = bool(result["web"].get("available"))
            record["web_results"] = len(result["web"].get("results") or [])
            record["notes"] = result["notes"]
            record["answer_usage"] = result.get("usage")
            record["hit"] = is_hit(item, result["hits"])
            if result["hits"]:
                verdict = judge(item, result["answer"], result["hits"])
                record["correct"] = verdict["correct"]
                record["grounded"] = verdict["grounded"]
                record["reason"] = verdict["reason"]
                record["judge_usage"] = verdict.get("usage")
            else:
                record["correct"] = None
                record["grounded"] = None
                record["reason"] = "Not scored automatically: no retrieved text context."
        except Exception as exc:
            record["secs"] = round(time.time() - started, 2)
            record["error"] = str(exc.__class__.__name__)
            record["correct"] = None
            record["grounded"] = None
            record["hit"] = None
        rows.append(record)
        time.sleep(4.5)
    return rows, status


def main():
    questions = json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))
    baseline_path = ROOT / "results.json"
    improved_path = ROOT / "results_improved.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None
    improved = json.loads(improved_path.read_text(encoding="utf-8")) if improved_path.is_file() else None

    agent_rows, web_status = run_agentic(questions)
    agent_path = ROOT / "results_agentic.json"
    agent_path.write_text(json.dumps(agent_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    def subset(rows):
        if rows is None:
            return None
        return [rows[number - 1] for number in REPRESENTATIVE if number - 1 < len(rows)]

    report = [
        "# RAG comparison",
        "",
        "Baseline is the preserved `evals/results.json` PDF-only run.",
        "Improved is `evals/results_improved.json` when that file exists.",
        "Agentic results are in `evals/results_agentic.json`.",
        "Gold answers were not passed to answer generation.",
        "",
        "## Full improved run" if improved else "## Full improved run",
        "",
    ]
    if improved is None:
        report.append("Improved evaluation file is not present yet, so that column is not run.")
    else:
        report.append("Metrics below for the improved system use every saved question, not only the sample.")
    report.extend([
        "",
        "| System | Questions | Correct | Grounded | Page hit | Abstention on unanswerable | Latency and usage |",
        "|---|---|---|---|---|---|---|",
        _line("Baseline PDF-only, all saved questions", _metrics(baseline) if baseline else None),
        _line("Improved PDF RAG, all saved questions", _metrics(improved) if improved else None),
        "",
        "## Representative questions",
        "",
        "Questions " + ", ".join(str(number) for number in REPRESENTATIVE) + ".",
        "",
        "| System | Questions | Correct | Grounded | Page hit | Abstention on unanswerable | Latency and usage |",
        "|---|---|---|---|---|---|---|",
        _line("Baseline PDF-only", _metrics(subset(baseline)) if baseline else None),
        _line("Improved PDF RAG", _metrics(subset(improved)) if improved else None),
        _line("Agentic RAG", _metrics(agent_rows)),
        "",
        "## External search",
        "",
    ])
    if web_status.get("available"):
        report.append(f"Web search provider configured: {web_status['provider']}.")
    else:
        report.append(
            "External search was not run. "
            + (web_status.get("reason") or "No search credential is configured.")
            + " Agentic answers used PDF evidence only and could abstain."
        )
    report.extend([
        "",
        "API dollar cost was not returned by Gemini. Token totals are included only when the API response had usage metadata.",
        "",
        "The agentic judge saw the reference answer only after the model answer was generated.",
        "",
    ])
    for row in agent_rows:
        label = "ERROR" if row.get("error") else ("ABSTAIN" if row.get("abstained") else "ANSWERED")
        report.append(
            f"- Q{row['question_number']} {label} in {row.get('secs', 'n/a')}s. "
            f"Web results: {row.get('web_results', 0)}. "
            f"{row.get('error') or row.get('unscored_reason', '')}"
        )

    destination = ROOT / "comparison_report.md"
    destination.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()

# Evaluation audit

Audit of `evals/results.json` as saved by the completed run. Reference answers and result records were not changed. No scores were recomputed by calling Gemini again. Support checks used the saved answers, the saved page lists, the PDF text, and the scanned-page transcripts that the runner indexed.

## Recorded metrics, recomputed from the file

The file contains 39 records. Each has `correct`, `grounded`, and `hit` filled in. None has an `error`.

| Metric | Stored summary | Recount from the 39 records |
|---|---|---|
| Questions | 39 | 39 |
| Correct | 39/39 (100%) | 39 true, 0 false, 0 null |
| Grounded | 39/39 (100%) | 39 true, 0 false, 0 null |
| Page hit | 23/24 (95.8%) | 23 true, 1 false, 15 null |
| Errors | 0 | 0 |
| Unscored | 0 | 0 |

`23/24` is 95.833..., which the runner prints as 95.8%. The arithmetic matches the records.

The one stored miss is question 3. Its labeled page is 1. The retrieved pages are 5, 3, 4, 6, and 2. Recomputing `hit` as "labeled page is among `retrieved_pages`" matches the stored `hit` on every question. `pages_sent` matches `retrieved_pages` on every question.

## Page-hit denominator and page numbers

A hit is counted only when `answerable` is true and `page` is a number. That is 24 questions:

- Questions 1–13, the answerable compiler-note items. Labeled pages are 1, 1, 1, 2, 5, 10, 12, 13, 16, 18, 20, 22, and 24.
- Questions 28–38, the answerable FinShield items. Labeled pages are 1 or 2, except question 38, which is page 3.

The other 15 questions are excluded, and that exclusion is what the runner implements:

- Questions 14, 27, and 39 are unanswerable and have `page: null`.
- Questions 15–26 are answerable, but `page` is null. Several of them carry `printed_page` (`cover`, `abstract`, or a printed page). Those printed numbers are not compared with retrieved PDF pages.

So 95.8% is not a hit rate over all 39 questions, and not over all 36 answerable questions. It is 23 of the 24 questions that store a numeric PDF page.

Page numbers are 1-based PDF page indexes, the same indexes used when text is extracted and when scanned pages are rendered. They are not the printed page numbers inside the mini-project report. For the FinShield article, PDF page 1 is the page marked "Page 1 of 4". For the scanned notes, PDF page 1 is the first sheet, and the transcript of that sheet is the three-address-code question. Question 3 shows the check is real: the quadruple fields also appear on the retrieved later pages, the answer is still right, and `hit` is still false because page 1 was not retrieved.

## Answer leakage

The reference answer and the labeled page are not passed into answer generation.

`generate_answer` uses `item["source"]` only to open that question's PDF. It then calls `ask_question_vision` or `ask_question_text` with the question text and the retrieved pages or chunks. `ai_engine.py` puts those excerpts or images in the prompt. It does not receive `item["answer"]` or `item["page"]`.

The reference answer is used only in `judge`, after the model answer has already been produced. The judge is also told, for unanswerable items, that the information is not in the document. That is label information for scoring, not an input to the RAG answer.

Two boundaries are worth keeping separate from leakage:

- Each question is run against its labeled file only. Retrieval cannot return a page from the wrong PDF, so this is not a three-document search.
- Questions 25 and 26 include the calculator inputs in the question (298, 202, 10 days, 75%). They do not include the reference results, 73.18% and 11 classes. Those results are on the transcribed screenshot pages that were retrieved (PDF pages 40 and 41).

## Unanswerable questions

The notes do not contain graph-colouring, colour, color, or a time-complexity statement. The mini-project text does not contain "pilot". The FinShield article does not contain "GPU". Those absences match the reference answers.

Question 14 refuses. It says the pages do not mention the time complexity of graph-colouring register allocation, then restates the register-allocation and register-assignment definitions from pages 11 and 12, which are in the retrieved transcript. It adds the parenthetical "such as NP-completeness" as an example of the missing fact. It does not claim the document says the algorithm is NP-complete. The refusal itself is valid. The parenthetical is not a document fact.

Question 27 refuses. It says the document does not mention a pilot test or a user count. It then points at Student 1 and Student 2 on retrieved pages 39 and 41 and treats them as interface examples. It does not invent a participant count.

Question 39 refuses. It says the paper does not give a GPU count or other training hardware. The architecture and metric remarks it adds are on the retrieved FinShield pages. It does not invent a GPU count.

The judge was not blind on these three. Its prompt says the information is unavailable and to mark the answer wrong if it invents an answer. In these records the model did refuse, so the `correct: true` marks agree with the answers. The prompt would also make it easier for a weak refusal to be accepted. That did not happen here.

## Answerable questions and retrieved content

For every answerable question, the distinctive reference facts were found in the text of the retrieved pages (PDF text, or the scan transcript for the handwritten notes and the five screenshot pages). The check was the reference wording, not similarity to the answer alone.

Examples that are easy to misread, and what the pages actually contain:

- Question 3 missed page 1, but `op`, `arg1`, `arg2`, and `result` are on the retrieved quadruple pages. Correctness does not depend on the missed page.
- Question 5's final line `t5 = t3 + t4` is on retrieved page 5.
- Question 9's example is `r1 := r2 * 2` replaced by `r1 := r2 << 1` on page 16. The notes use the symbol, not the word "shift".
- Question 13's reference facts are on page 24: the heap is slow compared with the stack, and there is a chance of a memory leak. `malloc`, `new`, `free`, and `delete` are there too.
- Question 25's 73.18% figure is in the page 40 screenshot transcript, which was retrieved. Question 26's "11 more classes" and 75.00% are in the page 41 transcript.
- FinShield figures (95%, 50,000 transactions, 5,000 users, 8%, the four graph metrics, F1 0.86 and 0.92, false-positive 0.12 and detection 0.91 at threshold 0.6, 0.25 down to 0.05, "Requires structured graph input", and the three future-work items) sit on the retrieved article pages.

Question 1's worked example (`t2 = a + b`, and the following steps) matches the scanned table on page 3, including that table's own arithmetic. It is document content, not an outside correction.

## Genuine issue: groundedness is lenient about extras

Question 13 is the clear case. The required facts are supported. The answer also says heap memory "can be fragmented" and that a leak can make the program crash or slow the system down. Neither "fragment" nor "crash" appears anywhere in the 24-page transcript. The retrieved pages for that question are 23, 24, 11, 20, and 21.

The saved judge reason says those additions are "consistent with general programming knowledge" and still sets `grounded` to true. The judge prompt says a claim is grounded only when the retrieved context supports it. On this record the judge did not apply that rule to the extra sentences. Correctness can stay true, because the reference facts are present and not contradicted. The groundedness flag is too generous.

Other long tutor-style answers add explanation around a supported fact. Question 9's remark that a left shift is faster for the processor is a gloss of the note's "costly" versus "cheaper", not a second numeric claim. Question 14's NP-completeness parenthetical is the other extra that is not in the notes. Neither one invents the asked-for fact.

## Recommendations

Keep the stored scores. They match the file, and the reference facts are in the retrieved pages. Do not edit `results.json` to raise or lower them from this audit.

1. Score groundedness without treating outside knowledge as support. If any concrete claim is absent from the retrieved pages, `grounded` should be false even when the reference fact is present. Question 13 would change under that rule. The other 38 were not re-judged sentence by sentence.
2. Ask the judge for unsupported extras as its own field, so a matching reference answer cannot hide them. The current prompt already says this, and question 13 shows the model did not follow it.
3. Give the scanned-page prompt the same limit the text prompt has: answer only from the supplied pages, and say when the fact is absent. Question 14 refused anyway. The vision prompt does not require that.
4. Publish the page-hit denominator beside the percentage. 95.8% does not cover the 12 answerable mini-project questions, because they have no numeric `page`. A separate check could map `printed_page` to a PDF page, without changing reference answers.
5. Leave gold pages and reference answers out of `ask_question_text` and `ask_question_vision`. That separation already holds.

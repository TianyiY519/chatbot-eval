# Tasks (from meetings Nov 3–4)

Goal (Jorge): evidence comparing the SharePoint/Copilot Studio agent and the in-house
Open WebUI bot on correctness, safety and reliability, using the web front ends only.

## Done (code, tested locally)
- [x] Folder → test set in one command (`make_testset.py`); PDF, PPTX, DOCX, TXT, MD
- [x] Ragas generation, source checks, balanced 100 per set, sets A/B on disjoint pages
- [x] English + French (question and reference answer); es/ar/zh/ru supported for later
- [x] OpenRouter as LLM backend (also Ollama, Gemini, Azure, OpenAI) via `.env`
- [x] Playwright web runner: fresh chat per question, resume, screenshots on error,
      persistent login; 50/50 cases collected from the mock bot (median 3.7 s)
- [x] Safety set: 15 prompts × EN/FR with expected behaviour
- [x] DeepEval scoring: correctness, hallucination, relevancy, toxicity, bias, safety
      handling + deterministic language check; judge self-check with planted bad answers
- [x] Bot-vs-bot comparison table (`06_compare.py`)

- [x] OpenRouter key from Jorge in `.env` ($30 limit, expires 2026-12-05)
- [x] Judge calibration on the mock run (GPT-6 Luna): every planted error caught, no correct
      answer failed; fixed GEval logprobs error and DeepEval 4.x score directions
- [x] Trial generation (GPT-6 Luna): 22/26 drafts pass, 9 of 10 documents covered, no
      section-number or methodology-trivia questions (trial set kept locally)
      Fixes: one Ragas run per document, correct-grammar questions only, multi-hop source
      matching, fallback page sample when a document fails (UNICEF Annual Report 2023)

## Next, in order
3. [ ] Final generation with `openai/gpt-6.1-terra`: `make_testset.py` → sets A and B (100 × EN+FR each);
       spot-check ~20 rows of `testset_A_review.csv` (`human_ok` column)
4. [ ] Push code + `data/testset_*` to GitHub; send Jorge the link
5. [ ] Practise the web runner on a public Open WebUI chatbot (Jorge's suggestion)
6. [ ] With Jorge: run `--inspect` on the SharePoint agent and the Open WebUI bot, fill
       `targets.json`, pilot 10 questions each
7. [ ] Full runs on both bots → `05_evaluate.py` → `06_compare.py` → results to Jorge

## Out of scope for now
Load testing (Locust), agent tracing, backend/API integration.

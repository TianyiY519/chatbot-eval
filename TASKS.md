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

## Waiting on
- [ ] OpenRouter API key from Jorge → put in `.env`

## Next, in order
1. [ ] Judge calibration: `05_evaluate.py --run mock_demo --canaries`; failures must be
       exactly the planted ones (see README). Adjust thresholds if not.
2. [ ] Trial generation: `make_testset.py --sets trial --keep 10` on a free model; read the CSV
3. [ ] Final generation with a strong model: `make_testset.py` → sets A and B (100 × EN+FR each);
       spot-check ~20 rows of `testset_A_review.csv` (`human_ok` column)
4. [ ] Push code + `data/testset_*` to GitHub; send Jorge the link
5. [ ] Practise the web runner on a public Open WebUI chatbot (Jorge's suggestion)
6. [ ] With Jorge: run `--inspect` on the SharePoint agent and the Open WebUI bot, fill
       `targets.json`, pilot 10 questions each
7. [ ] Full runs on both bots → `05_evaluate.py` → `06_compare.py` → results to Jorge

## Out of scope for now
Load testing (Locust), agent tracing, backend/API integration.

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

## Done (week of Nov 9–16)
- [x] Final generation with `openai/gpt-6.1-terra`: sets A and B, 100 × EN+FR each, 9–11 questions
      per document
- [x] Human review applied with `07_revise_testset.py`: 31 questions replaced, 4 reworded, every
      change logged; `question_type` added (A: 84 source_specific / 16 natural_user, B: 87 / 13);
      checked spares kept (13 for A, 9 for B)
- [x] Code, final sets and review trail on GitHub
- [x] End-to-end pilot on a local Open WebUI bot (10 documents as knowledge base): 20/20 answered
      through the web page (median 20.3 s), scored with DeepEval; planted wrong answers 10/10 caught
- [x] Web runner fixes found in the pilot: close pop-ups (`dismiss`), wait for the bot to finish
      (`busy`), remove interface text from replies (`strip`)

## Next, in order
1. [ ] Full rehearsal on the local bot: all of set A, EN + FR (~1 hour, ~$1)
2. [ ] With Jorge: access to the SharePoint agent and the Open WebUI bot; run `--inspect`, fill
       `targets.json`, pilot 10 questions each
3. [ ] Full runs on both bots → `05_evaluate.py` → `06_compare.py` → results to Jorge

## Open questions for Jorge
- French number format in reference answers: keep 12.5 or switch to 12,5
- Add a check that flags answers given without the documents (seen in the pilot: such an answer
  can still pass Correctness)

## Out of scope for now
Load testing (Locust), agent tracing, backend/API integration.

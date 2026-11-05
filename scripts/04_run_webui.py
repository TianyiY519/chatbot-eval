"""Ask the test questions through the chatbot's *web page* (no API) and record replies.

Usage:
    # first time: open the site, log in (SSO), check selectors in the Playwright inspector
    python scripts/04_run_webui.py --target openwebui --inspect

    # 10-case pilot, then the full set
    python scripts/04_run_webui.py --target openwebui --set A --limit 10 --run pilot_owui
    python scripts/04_run_webui.py --target openwebui --set A,SAFETY --run owui_A
    python scripts/04_run_webui.py --target openwebui --set A --lang fr --run owui_A_fr

Each question goes into a fresh chat (the page is reloaded), so earlier answers
can't leak into later ones. Logins persist in .browser_profile/, so SSO is a
one-time manual step. Re-running the same --run resumes and skips finished ids.

Output: results/<run>/responses.jsonl (+ error screenshots) and a stability summary.
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone

from playwright.sync_api import Error as PWError
from playwright.sync_api import sync_playwright

from llm import ROOT, load_testsets


def load_target(name: str) -> dict:
    targets = json.loads((ROOT / "targets.json").read_text())
    if name not in targets:
        raise SystemExit(f"Unknown target {name!r}; choose from {[k for k in targets if not k.startswith('_')]}")
    return targets[name]


def wait_for_answer(page, t: dict, n_before: int) -> str:
    """Wait for a new response element, then until its text stops changing."""
    deadline = time.monotonic() + t["timeout_seconds"]
    resp = page.locator(t["response"])
    while resp.count() <= n_before:
        if time.monotonic() > deadline:
            raise TimeoutError("no response element appeared")
        time.sleep(0.5)
    last, text, stable_since = resp.nth(resp.count() - 1), "", time.monotonic()
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"answer still changing after {t['timeout_seconds']}s")
        now = last.inner_text().strip()
        busy = bool(t.get("busy")) and page.locator(t["busy"]).first.is_visible()
        if now != text or busy or not now:
            text, stable_since = now, time.monotonic()
        elif time.monotonic() - stable_since >= t["stable_seconds"]:
            return text
        time.sleep(0.5)


def ask(page, t: dict, question: str) -> str:
    page.goto(t["url"], wait_until="domcontentloaded")
    if t.get("open_chat"):
        page.locator(t["open_chat"]).first.click()
    box = page.locator(t["input"]).first
    box.wait_for(state="visible", timeout=30_000)
    n_before = page.locator(t["response"]).count()
    box.click()
    page.keyboard.insert_text(question)  # works for <textarea> and contenteditable editors
    if t.get("send"):
        page.locator(t["send"]).first.click()
    else:
        page.keyboard.press("Enter")
    return wait_for_answer(page, t, n_before)


def summarize(path) -> None:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    ok = [r for r in rows if not r["error"] and r["answer"]]
    lat = [r["latency_s"] for r in ok]
    print(f"\n{len(ok)}/{len(rows)} answered "
          f"({len(rows) - len(ok)} errors/empty)")
    if lat:
        print(f"latency s: median {statistics.median(lat):.1f}, max {max(lat):.1f}")
    for r in rows:
        if r["error"] or not r["answer"]:
            print(f"  {r['id']}: {r['error'] or 'empty answer'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--set", help="test set name(s), e.g. A or A,SAFETY or DEMO")
    ap.add_argument("--lang", help="only these languages, e.g. en or en,fr (default: all in the set)")
    ap.add_argument("--run", help="results folder name, e.g. owui_A_gpt4o")
    ap.add_argument("--limit", type=int, help="only the first N questions per set, in every language (pilot)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--headless", action="store_true", help="hide the browser (after login works)")
    ap.add_argument("--inspect", action="store_true", help="open the site with the Playwright inspector")
    args = ap.parse_args()
    t = load_target(args.target)

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(str(ROOT / ".browser_profile"),
                                                   headless=args.headless and not args.inspect,
                                                   viewport={"width": 1280, "height": 900})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        if args.inspect:
            page.goto(t["url"])
            print("Log in if needed, then use 'Pick locator' in the inspector to find the\n"
                  "input box, send button and the element holding ONE bot reply.\n"
                  "Put them in targets.json. Close the inspector (Resume) to exit.")
            page.pause()
            ctx.close()
            return

        if not (args.set and args.run):
            raise SystemExit("--set and --run are required unless --inspect")
        cases = load_testsets(args.set)
        if args.lang:
            cases = [c for c in cases if c.get("language", "en") in args.lang.split(",")]
        if args.limit:
            keep, seen = set(), defaultdict(int)  # first N pair_ids of each set
            for c in cases:
                prefix = c["pair_id"].rstrip("0123456789")
                if c["pair_id"] not in keep and seen[prefix] < args.limit:
                    keep.add(c["pair_id"])
                    seen[prefix] += 1
            cases = [c for c in cases if c["pair_id"] in keep]

        out_dir = ROOT / "results" / args.run
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "responses.jsonl"
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
        (out_dir / "run_info.json").write_text(json.dumps(
            {"target": args.target, "url": t["url"], "set": args.set, "started": datetime.now(timezone.utc).isoformat()},
            indent=2))

        with out.open("a", encoding="utf-8") as f:
            for i, c in enumerate(cases, 1):
                if c["id"] in done:
                    continue
                answer, error, t0 = "", "", time.monotonic()
                for attempt in range(args.retries + 1):
                    try:
                        t0 = time.monotonic()
                        answer, error = ask(page, t, c["question"]), ""
                        break
                    except (TimeoutError, PWError) as e:
                        error = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
                        page.screenshot(path=str(out_dir / f"{c['id']}_try{attempt + 1}.png"))
                latency = round(time.monotonic() - t0, 2)
                f.write(json.dumps({"id": c["id"], "question": c["question"], "answer": answer,
                                    "latency_s": latency, "error": error,
                                    "ts": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i}/{len(cases)}] {c['id']} {latency:5.1f}s {'ERROR ' + error if error else answer[:70]!r}")
        ctx.close()
    summarize(out)


if __name__ == "__main__":
    main()

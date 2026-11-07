"""
Resumable experiment runner.

Runs every (task x condition x think-setting x repeat) cell, scores it, and appends one JSON
line per cell to --out. Re-running the same command skips cells that are already done, so you
can stop and restart freely.

Examples
  # smoke test the pipeline with no model at all
  python run_experiment.py --dry-run --n 2 --out dry.jsonl

  # small pilot: 20 items per (family, k), thinking off
  python run_experiment.py --model qwen3.5:4b --n 20 --think off --out pilot.jsonl

  # both reasoning settings (thinking is a controlled factor, not a default)
  python run_experiment.py --model qwen3.5:4b --n 20 --think off on --out pilot.jsonl

  # sampled run for robustness: 3 repeats with different seeds
  python run_experiment.py --model qwen3.5:4b --n 20 --think off --temperature 0.7 \
      --top-p 0.8 --top-k 20 --repeats 3 --out sampled.jsonl
"""
import argparse
import json
import os
import time

import tasks as T


def dry_llm(messages):
    """Stub model: exercises the pipeline without calling anything."""
    dry_llm.usage["calls"] += 1
    last = messages[-1]["content"]
    if "```python" in last:
        return "```python\ndef f1(x):\n    return x\n```"
    return "Step 1: 0\nStep 2: 0"


dry_llm.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "truncated": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--n", type=int, default=20, help="items per (family, k)")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--families", nargs="+", default=T.FAMILIES, choices=T.FAMILIES)
    ap.add_argument("--conditions", nargs="+", default=T.CONDITIONS, choices=T.CONDITIONS)
    ap.add_argument("--think", nargs="+", default=["off"], choices=["off", "on"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--task-seed", type=int, default=0)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["id"], r["condition"], r["think"], r["repeat"], r.get("task_seed", 0)))
    print(f"{len(done)} cells already done")

    jobs = []
    for fam in a.families:
        for k in a.ks:
            for i in range(a.n):
                for think in a.think:
                    for rep in range(a.repeats):
                        for cond in a.conditions:
                            jobs.append((T.make_task(fam, k, i, a.task_seed), cond, think, rep))
    print(f"{len(jobs)} cells total")

    llms = {}
    t0 = time.time()
    with open(a.out, "a") as f:
        for n, (task, cond, think, rep) in enumerate(jobs, 1):
            if (task["id"], cond, think, rep, a.task_seed) in done:
                continue
            key = (think, rep)
            if key not in llms:
                llms[key] = dry_llm if a.dry_run else T.ollama_llm(
                    a.model, think=(think == "on"), temperature=a.temperature, top_p=a.top_p,
                    top_k=a.top_k, seed=1000 + rep, host=a.host)
            llm = llms[key]
            before = dict(llm.usage)
            start = time.time()
            try:
                responses = T.run_condition(task, cond, llm)
                s = T.score(task, responses, cond)
                err = None
            except Exception as e:  # network error etc.: record and continue
                responses, s, err = [], {"all_correct": False}, repr(e)
            usage = {k_: llm.usage[k_] - before[k_] for k_ in before}
            rec = {**s, "task_seed": a.task_seed, "think": think, "repeat": rep, "model": "dry" if a.dry_run else a.model,
                   "temperature": a.temperature, "seconds": round(time.time() - start, 2),
                   "usage": usage, "responses": responses, "error": err}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s  last={task['id']} {cond} "
                      f"think={think} correct={s['all_correct']}")
    print("done")


if __name__ == "__main__":
    main()

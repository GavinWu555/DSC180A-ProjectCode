"""
Task generator, condition runner and scorer for the
"bulk prompt vs. step-by-step prompts" experiment.

Four task families (all synthetic, seeded, with programmatic ground truth):
  math_independent  - k unrelated computations over one shared list of integers
  math_dependent    - a k-step arithmetic chain; step i consumes the result of step i-1
  code_independent  - write k unrelated small Python functions (unit-tested)
  code_dependent    - write f1..fk where f_i must call f_{i-1} (unit-tested)

Three delivery conditions:
  bulk       - all k steps in one prompt, one call
  multiturn  - one step per turn, full conversation history retained
  chained    - one step per fresh call; only the original input plus the
               previous step's OUTPUT is carried forward (dependent families only)

Usage:
  python tasks.py generate --n 100 --ks 2 4 8 --seed 0 --out tasks.jsonl
  python tasks.py selftest            # oracle check that scorers work
"""
import argparse
import inspect
import json
import os
import random
import re
import subprocess
import sys
import tempfile

FAMILIES = ["math_independent", "math_dependent", "math_dependent_hard", "math_dependent_mid", "code_independent", "code_dependent"]
CONDITIONS = ["bulk", "multiturn", "chained"]


# ----------------------------------------------------------------------------
# Math, independent: k computations over one list
# ----------------------------------------------------------------------------
def _mi_ops(rng):
    T = rng.randint(5, 30)
    D = rng.choice([3, 4, 5, 7])
    return {
        "sum": ("Compute the sum of all the numbers in the list.", lambda L: sum(L)),
        "range": ("Compute the difference between the largest and the smallest number in the list.",
                  lambda L: max(L) - min(L)),
        "even": ("Count how many numbers in the list are even.",
                 lambda L: sum(1 for v in L if v % 2 == 0)),
        "gt": (f"Count how many numbers in the list are greater than {T}.",
               lambda L: sum(1 for v in L if v > T)),
        "odd_sum": ("Compute the sum of all odd numbers in the list.",
                    lambda L: sum(v for v in L if v % 2 != 0)),
        "evenidx": ("Compute the sum of the elements at even positions (positions counted from 0).",
                    lambda L: sum(L[0::2])),
        "prod3": ("Compute the product of the first three numbers in the list.",
                  lambda L: L[0] * L[1] * L[2]),
        "distinct": ("Count the distinct values in the list.", lambda L: len(set(L))),
        "div": (f"Compute the sum of all numbers in the list that are divisible by {D}.",
                lambda L: sum(v for v in L if v % D == 0)),
        "second": ("Find the second largest distinct value in the list.",
                   lambda L: sorted(set(L))[-2]),
    }


def gen_math_independent(rng, k):
    L = [rng.randint(-20, 60) for _ in range(12)]
    ops = _mi_ops(rng)
    keys = rng.sample(sorted(ops), k)
    steps = [{"kind": "int", "instr": ops[key][0], "expected": ops[key][1](L)} for key in keys]
    return {"preamble": f"Here is a list of integers: {L}", "steps": steps}


# ----------------------------------------------------------------------------
# Math, dependent: running-value chain (kept non-negative to avoid floor/mod ambiguity)
# ----------------------------------------------------------------------------
# Pilot finding: "remainder" had a ~41% first-error rate vs <=3% for other ops on qwen3.5:4b,
# so it dominated results. Disabled by default; set MATH_USE_MOD=True to re-enable.
MATH_USE_MOD = False


def _md_step(rng, v):
    choices = ["add", "mul", "fdiv"] + (["mod"] if MATH_USE_MOD else [])
    if v >= 2:
        choices.append("sub")
    if v < 2000:
        choices.append("sq")
    if v > 10 ** 5:
        choices = ["fdiv"]
    c = rng.choice(choices)
    if c == "add":
        a = rng.randint(3, 40)
        return f"add {a}", v + a, c
    if c == "sub":
        a = rng.randint(1, min(40, v))
        return f"subtract {a}", v - a, c
    if c == "mul":
        b = rng.randint(2, 9)
        return f"multiply by {b}", v * b, c
    if c == "fdiv":
        d = rng.randint(2, 6)
        return f"divide by {d} and round down", v // d, c
    if c == "mod":
        m = rng.randint(7, 60)
        return f"take the remainder when divided by {m}", v % m, c
    return "square it", v * v, c


def gen_math_dependent(rng, k):
    x = rng.randint(5, 50)
    v = x
    steps = []
    for i in range(k):
        text, v, op = _md_step(rng, v)
        lead = f"Start with the number {x} and" if i == 0 else "Take the result of the previous step and"
        steps.append({"kind": "int", "op": op, "instr": f"{lead} {text}.", "expected": v})
    return {"preamble": "", "steps": steps}


# ---- Harder variant: bigger numbers, 2-digit multipliers, a branching rule, digit sums, small remainders.
# Difficulty is a guess for a ~4B model; calibrate with `summarize.py --ops` and prune/reweight HARD_POOL.
def _digit_sum(n):
    return sum(int(ch) for ch in str(n))


HARD_POOL = ["add", "add", "sub", "sub", "mul", "mul2d", "mul2d", "fdiv", "digitsum", "branch", "branch", "mod", "sq"]


def _mdh_step(rng, v, base=None):
    pool = list(base or HARD_POOL)
    if v < 30:
        pool = [p for p in pool if p != "sub"]
    if v >= 1000:
        pool = [p for p in pool if p != "sq"]
    if v > 5 * 10 ** 4:
        pool = [p for p in pool if p not in ("mul", "mul2d", "sq")]
    if v > 10 ** 6:
        pool = ["fdiv", "mod"]
    c = rng.choice(pool)
    if c == "add":
        a = rng.randint(25, 400)
        return f"Add {a} to it.", v + a, c
    if c == "sub":
        a = rng.randint(10, min(300, v))
        return f"Subtract {a} from it.", v - a, c
    if c == "mul":
        b = rng.randint(3, 9)
        return f"Multiply it by {b}.", v * b, c
    if c == "mul2d":
        b = rng.randint(11, 29)
        return f"Multiply it by {b}.", v * b, c
    if c == "fdiv":
        d = rng.randint(2, 9)
        return f"Divide it by {d} and round down.", v // d, c
    if c == "mod":
        m = rng.randint(5, 13)
        return f"Take the remainder when it is divided by {m}.", v % m, c
    if c == "digitsum":
        return "Add the sum of its digits to it.", v + _digit_sum(v), c
    if c == "branch":
        return "If it is even, halve it; if it is odd, triple it and add 1.", (v // 2 if v % 2 == 0 else 3 * v + 1), c
    return "Square it.", v * v, c


def _gen_chain(rng, k, base):
    x = rng.randint(20, 400)
    v = x
    steps = []
    for i in range(k):
        text, v, op = _mdh_step(rng, v, base)
        lead = f"Start with the number {x}." if i == 0 else "Take the result of the previous step."
        steps.append({"kind": "int", "op": op, "instr": f"{lead} {text}", "expected": v})
    return {"preamble": "", "steps": steps}


def gen_math_dependent_hard(rng, k):
    """Floor-level on qwen3.5:4b (~20% first-error rate per step): kept for ablations."""
    return _gen_chain(rng, k, None)


# Calibrated from the hard1 pilot: digit-sum (88% error), remainder (48%) and the branching rule (38%)
# removed. Predicted fully-correct rate on qwen3.5:4b is roughly 55% / 40% / 29% at k = 8 / 12 / 16.
MID_POOL = ["add", "add", "sub", "sub", "mul", "mul2d", "mul2d", "fdiv", "sq"]


def gen_math_dependent_mid(rng, k):
    return _gen_chain(rng, k, MID_POOL)


# ----------------------------------------------------------------------------
# Code, independent: pool of small functions with reference implementations
# ----------------------------------------------------------------------------
def sum_even(nums):
    return sum(n for n in nums if n % 2 == 0)


def count_vowels(s):
    return sum(1 for c in s.lower() if c in "aeiou")


def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]


def second_largest(nums):
    u = sorted(set(nums))
    return u[-2]


def reverse_words(s):
    return " ".join(s.split()[::-1])


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def count_primes_below(n):
    return sum(1 for m in range(2, n) if all(m % d for d in range(2, int(m ** 0.5) + 1)))


def longest_run(s):
    best = cur = 0
    prev = None
    for c in s:
        cur = cur + 1 if c == prev else 1
        prev = c
        best = max(best, cur)
    return best


def dedupe_keep_order(nums):
    seen, out = set(), []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def caesar_shift(s, k):
    return "".join(chr((ord(c) - 97 + k) % 26 + 97) if c.isalpha() else c for c in s)


def _rl(rng, lo=-10, hi=30, n=None):
    return [rng.randint(lo, hi) for _ in range(n or rng.randint(3, 9))]


def _rs(rng, alpha="abcdeiouxyz ", n=None):
    return "".join(rng.choice(alpha) for _ in range(n or rng.randint(3, 12)))


CODE_POOL = {
    "sum_even": (sum_even, "takes a list of integers `nums` and returns the sum of the even numbers",
                 lambda r: (_rl(r),)),
    "count_vowels": (count_vowels, "takes a string `s` and returns the number of vowels (a, e, i, o, u; case-insensitive)",
                     lambda r: (_rs(r),)),
    "is_palindrome": (is_palindrome, "takes a string `s` and returns True if it reads the same forwards and backwards, "
                      "ignoring case and ignoring all non-alphanumeric characters, otherwise False",
                      lambda r: (r.choice(["Racecar", "A man, a plan, a canal: Panama", "hello", "No lemon, no melon", _rs(r, "ab ,", 6)]),)),
    "second_largest": (second_largest, "takes a list of integers `nums` (at least two distinct values) and returns the "
                       "second largest distinct value",
                       lambda r: (_rl(r, n=r.randint(4, 9)) + [-50, 99],)),
    "reverse_words": (reverse_words, "takes a string `s` and returns the words in reverse order, joined by single spaces",
                      lambda r: (_rs(r, "abc  ", r.randint(5, 14)),)),
    "gcd": (gcd, "takes two positive integers `a` and `b` and returns their greatest common divisor",
            lambda r: (r.randint(1, 200), r.randint(1, 200))),
    "count_primes_below": (count_primes_below, "takes an integer `n` and returns how many prime numbers are strictly less than n",
                           lambda r: (r.randint(0, 120),)),
    "longest_run": (longest_run, "takes a string `s` and returns the length of the longest run of identical consecutive "
                    "characters (0 for the empty string)",
                    lambda r: (_rs(r, "aab", r.randint(0, 14)),)),
    "dedupe_keep_order": (dedupe_keep_order, "takes a list `nums` and returns a new list with duplicates removed, keeping the "
                          "first occurrence of each value in its original order",
                          lambda r: (_rl(r, 0, 6, r.randint(0, 10)),)),
    "caesar_shift": (caesar_shift, "takes a lowercase string `s` and an integer `k` and shifts every letter forward by k "
                     "places in the alphabet, wrapping from z to a, leaving non-letters unchanged",
                     lambda r: (_rs(r, "abxyz ", r.randint(1, 10)), r.randint(0, 30))),
}


def gen_code_independent(rng, k):
    names = rng.sample(sorted(CODE_POOL), k)
    steps = []
    for name in names:
        fn, desc, argen = CODE_POOL[name]
        tests = []
        for _ in range(8):
            args = argen(rng)
            tests.append([list(args), fn(*args)])
        steps.append({"kind": "code", "func": name,
                      "instr": f"Write a Python function `{name}` that {desc}.",
                      "tests": tests, "ref_code": inspect.getsource(fn)})
    return {"preamble": "", "steps": steps}


# ----------------------------------------------------------------------------
# Code, dependent: f1 base, then f_i = transform(f_{i-1}(x))
# ----------------------------------------------------------------------------
F1_OPTS = [
    ("returns x squared plus 1", "x * x + 1"),
    ("returns x times 3 plus 2", "x * 3 + 2"),
    ("returns the sum of the decimal digits of x", "sum(int(c) for c in str(x))"),
    ("returns x plus the number of decimal digits of x", "x + len(str(x))"),
]
TRANSFORMS = [
    ("adds 7 to that result", "r + 7"),
    ("doubles that result", "r * 2"),
    ("returns the sum of the decimal digits of that result", "sum(int(c) for c in str(r))"),
    ("reverses the decimal digits of that result and returns the resulting integer", "int(str(r)[::-1])"),
    ("takes that result modulo 1000", "r % 1000"),
    ("divides that result by 3, rounding down", "r // 3"),
    ("returns the absolute difference between that result and 100", "abs(r - 100)"),
    ("squares that result and takes it modulo 997", "(r * r) % 997"),
]


def gen_code_dependent(rng, k):
    d1, e1 = rng.choice(F1_OPTS)
    exprs = [("x", e1)]
    steps = [{"kind": "code", "func": "f1",
              "instr": f"Write a Python function `f1(x)` for a non-negative integer x that {d1}.",
              "ref_code": f"def f1(x):\n    return {e1}\n"}]
    for i in range(2, k + 1):
        d, e = rng.choice(TRANSFORMS)
        steps.append({"kind": "code", "func": f"f{i}",
                      "instr": f"Write a Python function `f{i}(x)` that calls `f{i-1}(x)` and then {d}.",
                      "ref_code": f"def f{i}(x):\n    r = f{i-1}(x)\n    return {e}\n"})
    ns = {}
    xs = [rng.randint(0, 500) for _ in range(8)] + [0, 1]
    for s in steps:
        exec(s["ref_code"], ns)
    for s in steps:
        s["tests"] = [[[x], ns[s["func"]](x)] for x in xs]
    return {"preamble": "", "steps": steps}


GENERATORS = {
    "math_independent": gen_math_independent,
    "math_dependent": gen_math_dependent,
    "math_dependent_hard": gen_math_dependent_hard,
    "math_dependent_mid": gen_math_dependent_mid,
    "code_independent": gen_code_independent,
    "code_dependent": gen_code_dependent,
}


def make_task(family, k, idx, seed):
    rng = random.Random(f"{seed}-{family}-{k}-{idx}")
    t = GENERATORS[family](rng, k)
    t.update({"id": f"{family}-k{k}-{idx:04d}", "family": family, "k": k})
    return t


# ----------------------------------------------------------------------------
# Prompt rendering (instruction text is IDENTICAL across conditions; only delivery differs)
# ----------------------------------------------------------------------------
def _fmt_all(task):
    if task["steps"][0]["kind"] == "int":
        return ("Give the answer to every step on its own line, in exactly this form: "
                "`Step N: <integer>`. Do not write anything else.")
    return ("Write each function in its own ```python code block, in order. "
            "Include only the function definitions: no tests, no print statements, no explanations.")


def _fmt_one(task, i):
    if task["steps"][0]["kind"] == "int":
        return f"Reply with exactly one line in this form: `Step {i}: <integer>`. Do not write anything else."
    return ("Write the function in a single ```python code block. "
            "Include only the function definition: no tests, no print statements, no explanations.")


def _step_line(task, i):
    return f"Step {i}: {task['steps'][i-1]['instr']}"


def bulk_prompt(task):
    head = [task["preamble"]] if task["preamble"] else []
    head.append("Complete all of the following steps in order.")
    steps = "\n".join(_step_line(task, i) for i in range(1, task["k"] + 1))
    return "\n\n".join(head + [steps, _fmt_all(task)])


def turn_prompt(task, i, carry=""):
    """Prompt for step i. Step 1 also carries the shared preamble (data)."""
    parts = []
    if i == 1 and task["preamble"]:
        parts.append(task["preamble"])
    if carry:
        parts.append(carry)
    parts.append(_step_line(task, i))
    parts.append(_fmt_one(task, i))
    return "\n\n".join(parts)


def chained_prompt(task, i, prev_responses):
    """Fresh-call prompt for step i. Carries original input, plus the previous
    OUTPUT only for dependent families."""
    carry = ""
    pre = task["preamble"]
    if task["family"].startswith("math_dependent") and i > 1:
        val = parse_int(prev_responses[-1], i - 1)
        carry = f"The result of the previous step was: {val if val is not None else 'unknown'}"
    elif task["family"] == "code_dependent" and i > 1:
        code = "\n\n".join(extract_blocks("\n".join(prev_responses)))
        carry = f"Code written so far:\n```python\n{code}\n```"
    parts = ([pre] if pre else []) + ([carry] if carry else []) + [_step_line(task, i), _fmt_one(task, i)]
    return "\n\n".join(parts)


def run_condition(task, condition, llm):
    """llm(messages: list[{'role','content'}]) -> str. Returns list of raw responses."""
    k = task["k"]
    if condition == "bulk":
        return [llm([{"role": "user", "content": bulk_prompt(task)}])]
    if condition == "multiturn":
        msgs, out = [], []
        for i in range(1, k + 1):
            msgs.append({"role": "user", "content": turn_prompt(task, i)})
            r = llm(msgs)
            msgs.append({"role": "assistant", "content": r})
            out.append(r)
        return out
    if condition == "chained":
        out = []
        for i in range(1, k + 1):
            out.append(llm([{"role": "user", "content": chained_prompt(task, i, out)}]))
        return out
    raise ValueError(condition)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def parse_int(text, i):
    m = re.findall(rf"Step\s*{i}\s*:\s*\**\s*(-?\d+)", text)
    return int(m[-1]) if m else None


def extract_blocks(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return [b for b in blocks if b.strip()]


RUNNER = r'''
import json, sys, signal
spec = json.load(open(sys.argv[1]))
def _h(*a): raise TimeoutError()
signal.signal(signal.SIGALRM, _h)
ns = {}
for blk in spec["blocks"]:
    signal.alarm(5)
    try: exec(blk, ns)
    except BaseException: pass
    signal.alarm(0)
out = {}
for name, tests in spec["tests"].items():
    ok = callable(ns.get(name))
    if ok:
        for args, exp in tests:
            signal.alarm(2)
            try:
                if ns[name](*args) != exp: ok = False
            except BaseException: ok = False
            signal.alarm(0)
            if not ok: break
    out[name] = ok
print("@@RESULT@@" + json.dumps(out))
'''


def run_code_tests(blocks, tests_by_func, timeout=60):
    """NOT a security sandbox. Run the whole harness inside Docker / a VM."""
    with tempfile.TemporaryDirectory() as d:
        spec_p, run_p = os.path.join(d, "spec.json"), os.path.join(d, "run.py")
        json.dump({"blocks": blocks, "tests": tests_by_func}, open(spec_p, "w"))
        open(run_p, "w").write(RUNNER)
        try:
            p = subprocess.run([sys.executable, run_p, spec_p], capture_output=True, text=True, timeout=timeout)
            m = re.search(r"@@RESULT@@(.*)", p.stdout)
            if m:
                return json.loads(m.group(1))
        except subprocess.TimeoutExpired:
            pass
    return {n: False for n in tests_by_func}


def score(task, responses, condition):
    steps, k = task["steps"], task["k"]
    if steps[0]["kind"] == "int":
        if condition == "bulk":
            texts = [responses[0]] * k
        else:
            texts = responses
        got = [parse_int(texts[i], i + 1) for i in range(k)]
        correct = [g == s["expected"] for g, s in zip(got, steps)]
        fmt_ok = [g is not None for g in got]
    else:
        blocks = [b for r in responses for b in extract_blocks(r)]
        res = run_code_tests(blocks, {s["func"]: s["tests"] for s in steps})
        correct = [bool(res.get(s["func"])) for s in steps]
        fmt_ok = [bool(blocks)] * k
    # conditional accuracy: step i correct AND all earlier steps correct (propagation-aware)
    prefix = [all(correct[: i + 1]) for i in range(k)]
    return {"id": task["id"], "family": task["family"], "k": k, "condition": condition,
            "step_correct": correct, "format_ok": fmt_ok, "prefix_correct": prefix,
            "all_correct": all(correct)}


# ----------------------------------------------------------------------------
# Example local-model adapter (Ollama). Swap for llama.cpp / vLLM / transformers as needed.
# ----------------------------------------------------------------------------
def ollama_llm(model, think=False, temperature=0.0, top_p=None, top_k=None, seed=0,
               num_predict=None, num_ctx=None, host="http://localhost:11434"):
    """Ollama adapter. `think` is passed explicitly so reasoning is a controlled variable.
    The returned callable gives back ONLY the visible answer (thinking is stripped, never
    fed back into multi-turn history). Token usage accumulates in llm.usage."""
    import requests

    if num_predict is None:
        num_predict = 8192 if think else 1024
    if num_ctx is None:
        num_ctx = 16384 if think else 8192

    def llm(messages):
        opts = {"temperature": temperature, "seed": seed, "num_predict": num_predict, "num_ctx": num_ctx}
        if top_p is not None:
            opts["top_p"] = top_p
        if top_k is not None:
            opts["top_k"] = top_k
        r = requests.post(f"{host}/api/chat", json={
            "model": model, "messages": messages, "stream": False,
            "think": think, "options": opts}, timeout=1800)
        r.raise_for_status()
        j = r.json()
        content = j["message"].get("content", "") or ""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        u = llm.usage
        u["calls"] += 1
        u["prompt_tokens"] += j.get("prompt_eval_count", 0)
        u["output_tokens"] += j.get("eval_count", 0)
        u["truncated"] += int(j.get("done_reason") == "length")
        return content

    llm.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "truncated": 0}
    return llm


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def oracle_responses(task, condition):
    """Perfect responses, used to validate the scorer."""
    steps = task["steps"]
    if steps[0]["kind"] == "int":
        lines = [f"Step {i+1}: {s['expected']}" for i, s in enumerate(steps)]
    else:
        lines = [f"```python\n{s['ref_code']}\n```" for s in steps]
    return ["\n".join(lines)] if condition == "bulk" else lines


def selftest():
    ok = True
    for fam in FAMILIES:
        for k in (2, 4, 8):
            t = make_task(fam, k, 0, seed=123)
            for cond in CONDITIONS:
                s = score(t, oracle_responses(t, cond), cond)
                ok &= s["all_correct"]
                if not s["all_correct"]:
                    print("ORACLE FAILED", t["id"], cond, s)
            # a deliberately wrong answer must be marked wrong
            bad = oracle_responses(t, "bulk")[0]
            bad = bad.replace("return", "return 0 *", 1) if t["steps"][0]["kind"] == "code" else \
                re.sub(r"Step 1: (-?\d+)", lambda m: f"Step 1: {int(m.group(1)) + 1}", bad)
            s = score(t, [bad], "bulk")
            if s["step_correct"][0]:
                ok = False
                print("NEGATIVE CONTROL FAILED", t["id"])
    # prompt preview
    t = make_task("math_dependent", 3, 0, 123)
    print("--- bulk prompt preview ---\n" + bulk_prompt(t))
    print("--- turn 2 preview ---\n" + turn_prompt(t, 2))
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=100, help="items per (family, k)")
    g.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8])
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="tasks.jsonl")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    with open(a.out, "w") as f:
        for fam in FAMILIES:
            for k in a.ks:
                for i in range(a.n):
                    f.write(json.dumps(make_task(fam, k, i, a.seed)) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

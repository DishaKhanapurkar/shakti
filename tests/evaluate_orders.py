"""Small reproducible order-parsing evaluation set for Shakti.

Run from the project root:
  python tests/evaluate_orders.py                 # regex/template baseline
  SHAKTI_USE_GEMMA=1 python tests/evaluate_orders.py  # optional Gemma run (downloads model if needed)

This reports observed results; it does not assume Gemma is better. Review each failure.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import parse_order  # noqa: E402

CASES = [
    ("2 kg tomato, 1 kg potato", [{"item":"Tomato","qty":2,"unit":"kg"},{"item":"Potato","qty":1,"unit":"kg"}]),
    ("palak 1 bunch", [{"item":"Palak","qty":1,"unit":"bunch"}]),
    ("2 किलो टोमॅटो आणि 1 किलो बटाटे", [{"item":"Tomato","qty":2,"unit":"kg"},{"item":"Potato","qty":1,"unit":"kg"}]),
    ("2 किलो टमाटर और 1 किलो आलू", [{"item":"Tomato","qty":2,"unit":"kg"},{"item":"Potato","qty":1,"unit":"kg"}]),
    ("Tai 2 kg kanda dya", [{"item":"Onion","qty":2,"unit":"kg"}]),
    ("1 bunch dhaniya", [{"item":"Coriander","qty":1,"unit":"bunch"}]),
    ("bhindi 0.5 kg", [{"item":"Bhendi","qty":0.5,"unit":"kg"}]),
    ("tomatoes 3 kg", [{"item":"Tomato","qty":3,"unit":"kg"}]),
    ("आलू 2 किलो", [{"item":"Potato","qty":2,"unit":"kg"}]),
    ("kanda 1 kg, tomato 2 kg", [{"item":"Onion","qty":1,"unit":"kg"},{"item":"Tomato","qty":2,"unit":"kg"}]),
    ("2 kg onion and 2 kg onion", [{"item":"Onion","qty":4,"unit":"kg"}]),
    ("1 जुडी पालक", [{"item":"Palak","qty":1,"unit":"bunch"}]),
    ("Please send 1 kg spinach", [{"item":"Palak","qty":1,"unit":"kg"}]),
    ("3 किलो कांदे", [{"item":"Onion","qty":3,"unit":"kg"}]),
    ("भिंडी 2 किलो", [{"item":"Bhendi","qty":2,"unit":"kg"}]),
    ("2 kg tomato, no onion", [{"item":"Tomato","qty":2,"unit":"kg"}]),
    ("kanda nako, 1 kg tomato", [{"item":"Tomato","qty":1,"unit":"kg"}]),
    ("send 1 kg potato and 1 bunch coriander", [{"item":"Potato","qty":1,"unit":"kg"},{"item":"Coriander","qty":1,"unit":"bunch"}]),
    ("tomato 1.5 kg", [{"item":"Tomato","qty":1.5,"unit":"kg"}]),
    ("please send vegetables", []),
    # Deliberately harder language cases: these should show whether Gemma adds value.
    ("मला दोन किलो टोमॅटो आणि एक जुडी पालक द्या", [{"item":"Tomato","qty":2,"unit":"kg"},{"item":"Palak","qty":1,"unit":"bunch"}]),
    ("Tai, ek kilo batata ani don kilo kanda dya", [{"item":"Potato","qty":1,"unit":"kg"},{"item":"Onion","qty":2,"unit":"kg"}]),
    ("भाऊ, आधी एक किलो बटाटा द्या आणि दोन जुडी कोथिंबीर", [{"item":"Potato","qty":1,"unit":"kg"},{"item":"Coriander","qty":2,"unit":"bunch"}]),
    ("मुझे दो किलो टमाटर और एक किलो आलू चाहिए", [{"item":"Tomato","qty":2,"unit":"kg"},{"item":"Potato","qty":1,"unit":"kg"}]),
    ("Tai 2 kilo tomato pahije, kanda nako", [{"item":"Tomato","qty":2,"unit":"kg"}]),
    ("मला दोन किलो टोमॅटो द्या", [{"item":"Tomato","qty":2,"unit":"kg"}]),
]


def canonical(items):
    return sorted((x["item"], round(float(x["qty"]), 3), x["unit"]) for x in items)


def main():
    matches = 0
    started = time.perf_counter()
    for index, (message, expected) in enumerate(CASES, 1):
        try:
            actual, parser, needs_clarification = parse_order(message)
            ok = canonical(actual) == canonical(expected)
            matches += int(ok)
            print(f"{'PASS' if ok else 'FAIL'} {index:02d} [{parser}] {message}")
            if not ok:
                print("  expected:", json.dumps(expected, ensure_ascii=False))
                print("  actual:  ", json.dumps(actual, ensure_ascii=False), "clarification=", needs_clarification)
        except Exception as exc:
            print(f"ERROR {index:02d} {message}: {exc}")
    elapsed = time.perf_counter() - started
    print(f"\nExact-case accuracy: {matches}/{len(CASES)} = {matches / len(CASES):.1%}")
    print(f"Elapsed: {elapsed:.2f}s | Gemma enabled: {os.getenv('SHAKTI_USE_GEMMA', '0') == '1'}")
    print("This is a small development set, not a general benchmark. Inspect failures before making claims.")


if __name__ == "__main__":
    main()

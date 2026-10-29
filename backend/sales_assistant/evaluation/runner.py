from __future__ import annotations

import argparse
import sys

from sales_assistant.analytics.registry import registry
from sales_assistant.container import Services, build_services
from sales_assistant.data.resolver import EntityNotFound
from sales_assistant.evaluation.cases import EASY_CASES, HARD_CASES, HONESTY_CASES, PlanCase
from sales_assistant.llm.planner import PlanningError


def _normalise(services: Services, args: dict, key: str):
    resolver = services.pipeline.context.resolver
    value = args.get(key)
    try:
        if key == "category":
            return resolver.category(str(value))
        if key == "value":
            return resolver.segment(args["dimension"], str(value))
    except (EntityNotFound, KeyError):
        return str(value)
    return value


def calls_match(services: Services, expected: list[dict], actual: list[dict]) -> bool:
    if [c["tool"] for c in expected] != [c["tool"] for c in actual]:
        return False
    for want, got in zip(expected, actual):
        for key in want["args"]:
            if _normalise(services, got["args"], key) != _normalise(services, want["args"], key):
                return False
    return True


def validate_cases(services: Services, cases: tuple[PlanCase, ...]) -> list[str]:
    failures = []
    context = services.pipeline.context
    for case in cases:
        for expected in case.expected:
            result = registry.execute(context, expected["tool"], expected["args"])
            if "error" in result:
                failures.append(f"{case.question} -> {expected}: {result['error']}")
    return failures


def run_planning(services: Services, cases: tuple[PlanCase, ...]) -> int:
    passed = 0
    for case in cases:
        history = [{"role": "user", "content": q} for q in case.history]
        try:
            planned = [{"tool": c.name, "args": c.args} for c in services.pipeline.planner.plan(case.question, history)]
        except PlanningError as exc:
            planned = [{"tool": "PLANNING_ERROR", "args": {"detail": str(exc)}}]
        ok = calls_match(services, case.expected, planned)
        passed += ok
        print(f"{'OK  ' if ok else 'FAIL'} {case.question}")
        if not ok:
            print(f"     expected: {case.expected}\n     got:      {planned}")
    print(f"\nplanning: {passed}/{len(cases)}")
    return len(cases) - passed


def run_honesty(services: Services) -> int:
    failures = 0
    for case in HONESTY_CASES:
        result = services.pipeline.answer(case.question)
        text = result["answer"]
        problems = []
        if not result["grounded"]:
            problems.append(f"not grounded {result['ungrounded_numbers']} {result['unsupported_terms']}")
        if not any(word.lower() in text.lower() for word in case.required_any):
            problems.append(f"none of {case.required_any} mentioned")
        problems += [f"contains forbidden '{w}'" for w in case.forbidden if w.lower() in text.lower()]
        failures += bool(problems)
        print(f"{'FAIL' if problems else 'OK  '} {case.question}\n     {text}")
        for problem in problems:
            print(f"     ! {problem}")
    print(f"\nhonesty: {len(HONESTY_CASES) - failures}/{len(HONESTY_CASES)}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the planner and the honesty of the answers.")
    parser.add_argument("--validate", action="store_true", help="offline: check that every expected call runs (no Ollama)")
    parser.add_argument("--plan", action="store_true", help="easy cases: does Qwen choose the right tool and arguments?")
    parser.add_argument("--hard", action="store_true", help="harder phrasings, follow-ups and out-of-scope questions")
    parser.add_argument("--honesty", action="store_true", help="full answers: no invented claims, limits explained")
    args = parser.parse_args()
    services = build_services()
    failures = 0
    if args.validate or not (args.plan or args.hard or args.honesty):
        problems = validate_cases(services, EASY_CASES + HARD_CASES)
        print("\n".join(problems) or "all expected calls execute")
        failures += len(problems)
    if args.plan or args.hard or args.honesty:
        if not services.client.is_ready():
            print("Ollama is not reachable. Start it and run: ollama pull qwen3:8b", file=sys.stderr)
            return 2
    if args.plan:
        failures += run_planning(services, EASY_CASES)
    if args.hard:
        failures += run_planning(services, HARD_CASES)
    if args.honesty:
        failures += run_honesty(services)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

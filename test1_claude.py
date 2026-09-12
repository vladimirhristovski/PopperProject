import os
import time

from popper import Popper

from config import (
    ALPHA,
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    DATA_DIR,
    HYPOTHESES,
    MAX_RETRY,
    MAX_TESTS,
    TIME_LIMIT,
)
from popper_common import (
    check_data,
    determine_status,
    generate_report,
    parse_result,
    save_trace,
)


def run(results_file="results/results_claude.csv"):
    print("=" * 70)
    print(f"POPPER — Test 1: Claude ({CLAUDE_MODEL})")
    print("=" * 70)

    os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)

    missing = check_data()
    if missing:
        print(f"ERROR: Missing data files: {missing}")
        print("Run: python prepare_data.py")
        return

    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

    trace_dir = results_file.replace(".csv", "_traces")

    results = []
    total_start = time.time()

    print(f"\nRunning {len(HYPOTHESES)} hypotheses (agent reinitialised per hypothesis)\n")
    print(f"Traces will be saved under: {trace_dir}/")
    print("=" * 70)

    for i, hypothesis in enumerate(HYPOTHESES, 1):
        print(f"\n[{i}/{len(HYPOTHESES)}] {hypothesis[:65]}...")
        print("  Initializing POPPER agent...")

        try:
            agent = Popper(llm=CLAUDE_MODEL, plot_agent_architecture=False, domain="social science")
            agent.register_data(data_path=DATA_DIR, loader_type="custom")
            agent.configure(
                alpha=ALPHA,
                max_num_of_tests=MAX_TESTS,
                max_retry=MAX_RETRY,
                time_limit=TIME_LIMIT,
                aggregate_test="E-value",
                relevance_checker=True,
                use_react_agent=True,
            )
        except Exception as e:
            print(f"  ERROR: Failed to initialize agent — {e}")
            results.append({"model": CLAUDE_MODEL, "hypothesis": hypothesis,
                            "status": "ERROR", "e_value": 0.0,
                            "decision": "init_error", "time_min": 0.0})
            continue

        start = time.time()
        try:
            result = agent.validate(hypothesis=hypothesis)
            elapsed = (time.time() - start) / 60

            try:
                save_trace(result, i, hypothesis, CLAUDE_MODEL, trace_dir)
            except Exception as e:
                print(f"  Warning: failed to save trace — {e}")

            e_value, decision = parse_result(result)
            status = determine_status(e_value, decision)

            print(f"  Status  : {status}")
            print(f"  E-value : {e_value:.4f}")
            print(f"  Decision: {decision}")
            print(f"  Time    : {elapsed:.1f} min")

            results.append({"model": CLAUDE_MODEL, "hypothesis": hypothesis,
                            "status": status, "e_value": e_value,
                            "decision": decision, "time_min": elapsed})

        except Exception as e:
            elapsed = (time.time() - start) / 60
            print(f"  ERROR: {str(e)[:120]}")
            try:
                save_trace({"error": str(e)}, i, hypothesis, CLAUDE_MODEL, trace_dir)
            except Exception as trace_err:
                print(f"  Warning: failed to save error trace — {trace_err}")
            results.append({"model": CLAUDE_MODEL, "hypothesis": hypothesis,
                            "status": "ERROR", "e_value": 0.0,
                            "decision": "error", "time_min": elapsed})

    total_time = (time.time() - total_start) / 60

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    print(f"\nCSV saved to: {results_file}")

    generate_report(results, total_time, results_file, CLAUDE_MODEL)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"{r['status']:15} | E={r['e_value']:7.4f} | {r['time_min']:5.1f}m | {r['hypothesis'][:40]}")
    print("=" * 70)
    print(f"Total time : {total_time:.1f} min ({total_time / 60:.2f} hours)")
    print(f"Supported  : {sum(1 for r in results if r['status'] == 'SUPPORTED')}")
    print(f"Not Supp.  : {sum(1 for r in results if r['status'] == 'NOT SUPPORTED')}")
    print(f"Errors     : {sum(1 for r in results if r['status'] == 'ERROR')}")
    print("=" * 70)


if __name__ == "__main__":
    run()

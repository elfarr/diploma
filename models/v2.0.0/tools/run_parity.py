import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL_DIR = Path("models/v2.0.0")
DEFAULT_TEST_DIR = Path("tests")
DEFAULT_CLI = Path("models/v2.0.0/tools/cli_predict.py")

def run_case(case_path: Path, model_dir: Path, cli_path: Path):
    cmd = [
        sys.executable,
        str(cli_path),
        "--input",
        str(case_path),
        "--model",
        str(model_dir),
    ]
    out = subprocess.check_output(cmd, text=True)  
    start = out.find("{")
    if start == -1:
        raise ValueError()
    end = out.find("}\n", start)
    if end == -1:
        end = out.rfind("}")
    json_text = out[start : end + 1]
    return json.loads(json_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tests",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Папка с case*.json",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Путь к каталогу модели",
    )
    ap.add_argument(
        "--cli",
        type=Path,
        default=DEFAULT_CLI,
        help="Путь к cli_predict.py",
    )
    args = ap.parse_args()

    files = sorted(args.tests.glob("case*.json"))
    if not files:
        print(f"Тестовые кейсы не найдены")
        return

    ok_cnt = 0
    total = 0

    for f in files:
        raw = f.read_text(encoding="utf-8").strip()
        if not raw:
            print(f"{f.name}: пропущен")
            continue
        sample = json.loads(raw)
        expected = sample.get("expected", {})
        exp_p = expected.get("p_unfavorable", None)
        exp_v = expected.get("verdict", None)

        res = run_case(f, args.model, args.cli)
        got_p = res.get("вероятность_неблагоприятного") or res.get("p_unfavorable")
        got_v = res.get("вердикт") or res.get("verdict")

        verdict_map = {
            "благоприятный": "favorable",
            "неблагоприятный": "unfavorable",
            "неопределено": "undetermined",
        }
        if got_v in verdict_map:
            got_v_compare = verdict_map[got_v]
        else:
            got_v_compare = got_v

        total += 1
        status = "OK"
        notes = []

        if got_p is None or got_v_compare is None:
            status = "FAIL"
            notes.append("не удалось распарсить ответ модели")

        if exp_p is not None:
            diff = abs(float(exp_p) - float(got_p))
            if diff > 1e-3:
                status = "FAIL"
                notes.append(f"p diff={diff:.6f} (ожидалось={exp_p}, получили={got_p})")

        if exp_v is not None and exp_v != got_v_compare:
            status = "FAIL"
            notes.append(f"вердикт (ожидалось={exp_v}, получили={got_v_compare})")

        if status == "OK":
            ok_cnt += 1

        print(
            f"{f.name}: {status} | p={got_p} | verdict={got_v}"
            + ("" if not notes else " | " + "; ".join(notes))
        )

    print(f"\nИтог: {ok_cnt}/{total} кейсов прошло")


if __name__ == "__main__":
    main()

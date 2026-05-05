import argparse
import csv
import json
from pathlib import Path


METRIC_COLUMNS = [
    ("index_time", "index_time"),
    ("throughput", "query_throughput"),
    ("recall", "recall"),
]

SAMPLED_COLUMNS = [
    ("sampled_recall", "sampled_recall"),
    ("sampled_tput", "sampled_query_throughput"),
]


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def load_records(result_file):
    records = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line_nr, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                if records:
                    raise ValueError(f"Invalid JSON in {result_file}:{line_nr}") from exc
                return []
            if isinstance(record, dict):
                records.append(record)
    return records


def collect_param_keys(records):
    keys = []
    seen = set()
    for record in records:
        params = record.get("params") or {}
        if not isinstance(params, dict):
            continue
        for key in params:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def get_columns(records, param_keys):
    columns = ["step_id"] + param_keys + [name for name, _ in METRIC_COLUMNS]
    for name, source in SAMPLED_COLUMNS:
        if any(source in record for record in records):
            columns.append(name)
    return columns


def build_row(record, param_keys, columns):
    params = record.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    row = {"step_id": format_value(record.get("step_id"))}
    for key in param_keys:
        row[key] = format_value(params.get(key))
    for name, source in METRIC_COLUMNS + SAMPLED_COLUMNS:
        if name in columns:
            row[name] = format_value(record.get(source))
    return row


def convert_file(result_file):
    records = load_records(result_file)
    if not records:
        return None

    param_keys = collect_param_keys(records)
    columns = get_columns(records, param_keys)
    output_file = result_file.with_suffix(".csv")

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(build_row(record, param_keys, columns))

    return output_file


def collect_result_files(paths):
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.txt")))
        else:
            files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(description="Normalize result JSONL files into CSV files.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[Path("results")],
        help="Result file or directory. Defaults to results.",
    )
    args = parser.parse_args()

    paths = [Path(path) for path in args.paths]
    result_files = collect_result_files(paths)
    if not result_files:
        raise FileNotFoundError("No result files found.")

    converted = 0
    for result_file in result_files:
        if not result_file.exists():
            raise FileNotFoundError(f"File not found: {result_file}")

        output_file = convert_file(result_file)
        if output_file is None:
            print(f"Skipped {result_file}: no JSON records")
            continue

        converted += 1
        print(f"Saved CSV to {output_file}")

    if converted == 0:
        raise ValueError("No JSON result files were converted.")


if __name__ == "__main__":
    main()

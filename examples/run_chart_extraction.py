"""Run the chart reader on real chart images and save data/sample_chart_extractions.json.

Run from the project root, with your vision-model settings in the environment / .env
(see the docstring of src/ingestion/chart_reader.py)::

    python examples/run_chart_extraction.py --images figs/fig1.png figs/fig2.png figs/fig3.png

Optionally record where each chart came from (paper, URL, licence)::

    python examples/run_chart_extraction.py --images figs/*.png --source-info sources.json
    # sources.json: {"fig1.png": {"paper": "...", "url": "...", "license": "..."}}

Every entry in the output comes from a real model call. The file is only written
when at least --min-success charts were read successfully, so it can never
contain hand-written or mock results. Failed images are recorded with their error.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.chart_reader import (  # noqa: E402
    ChartConfigError,
    ChartReaderError,
    HTTPVisionClient,
    extract_chart_information,
    vision_config_from_env,
)


def main(argv=None, client=None, env=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", nargs="+", required=True, help="chart image files (png/jpg/gif/webp)")
    parser.add_argument("--out", default="data/sample_chart_extractions.json")
    parser.add_argument("--source-info", default=None, help="optional JSON: image file name -> paper/url/license")
    parser.add_argument("--min-success", type=int, default=3)
    args = parser.parse_args(argv)

    provider = model = "injected-client"
    if client is None:
        try:
            config = vision_config_from_env(env)
        except ChartConfigError as exc:
            print(f"ERROR: {exc}\nNothing written.")
            return 1
        client, provider, model = HTTPVisionClient(config), config.provider, config.model
    sources = {}
    if args.source_info:
        with open(args.source_info, "r", encoding="utf-8") as handle:
            sources = json.load(handle)

    results = []
    for path in args.images:
        name = os.path.basename(path)
        entry = {"image": name, "source": sources.get(name), "sha256": None,
                 "status": "error", "extraction": None, "error": None}
        try:
            with open(path, "rb") as handle:
                entry["sha256"] = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            pass  # extract_chart_information reports the real problem below
        try:
            entry["extraction"] = extract_chart_information(path, client=client, env=env)
            entry["status"] = "ok"
        except ChartReaderError as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)
        print(f"{name}: {entry['status']}" + (f" ({entry['error']})" if entry["error"] else ""))

    ok = sum(r["status"] == "ok" for r in results)
    if ok < args.min_success:
        print(f"Only {ok} chart(s) read successfully (need {args.min_success}). Output file not written.")
        return 2
    output = {
        "generated_by": {"type": "real_vlm_call", "provider": provider, "model": model,
                         "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "summary": {"images_tested": len(results), "succeeded": ok, "failed": len(results) - ok},
        "note": "Values are model readings of chart images. 'estimated' values are approximate; "
                "verify against the source paper before relying on them.",
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare bounded, sequential downloads without installing any packages.

Uses only the Python standard library, so it also works before Conda setup.
The selected URLs can be written to a shell file for the migration guide.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import platform
import shlex
import statistics
import sys
import time
from urllib.parse import unquote, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen


SOURCES = {
    "pip": {
        "TUNA": "https://pypi.tuna.tsinghua.edu.cn/simple/",
        "Aliyun": "https://mirrors.aliyun.com/pypi/simple/",
        "USTC": "https://mirrors.ustc.edu.cn/pypi/simple/",
        "SJTU": "https://mirror.sjtu.edu.cn/pypi/web/simple/",
        "official": "https://pypi.org/simple/",
    },
    "torch": {
        "SJTU": "https://mirror.sjtu.edu.cn/pytorch-wheels/cu128/",
        "official": "https://download.pytorch.org/whl/cu128/",
    },
    "conda": {
        "TUNA": "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/",
        "USTC": "https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/",
        "BFSU": "https://mirrors.bfsu.edu.cn/anaconda/cloud/conda-forge/",
        "official": "https://conda.anaconda.org/conda-forge/",
    },
    "hf": {
        "HF-Mirror": "https://hf-mirror.com/",
        "official": "https://huggingface.co/",
    },
}
ENV_NAMES = {
    "pip": "MINIWAM_PIP_INDEX",
    "torch": "MINIWAM_TORCH_INDEX",
    "conda": "MINIWAM_CONDA_CHANNEL",
    "hf": "MINIWAM_HF_ENDPOINT",
}
PAYLOADS = {
    "pip": "numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "torch": "torch-2.11.0+cu128-cp310-cp310-*x86_64.whl",
    "conda": "linux-64/repodata.json.zst",
    "hf": "datasets/yuanty/LIBERO-fastwam/resolve/ee018b997c430bb12b5bf3c892d744798c5a2f91/libero_goal_no_noops_lerobot.tar.gz",
}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(value for key, value in attrs if key == "href" and value)


def request(url, timeout, **headers):
    return urlopen(Request(url, headers={
        "User-Agent": "MiniWAM-mirror-benchmark/1.0",
        "Accept-Encoding": "identity", **headers,
    }), timeout=timeout)


def resolve_payload(group, base, timeout):
    if group in ("conda", "hf"):
        return urljoin(base, PAYLOADS[group])
    index = urljoin(base, "numpy/" if group == "pip" else "torch/")
    with request(index, timeout) as response:
        # Bound the index response too; a mirror challenge page is not a wheel.
        html = response.read(8 * 1024 * 1024).decode("utf-8", errors="replace")
    parser = Links()
    parser.feed(html)
    for href in parser.hrefs:
        url = urldefrag(urljoin(index, href))[0]
        filename = unquote(urlparse(url).path.rsplit("/", 1)[-1])
        if fnmatch.fnmatchcase(filename, PAYLOADS[group]):
            return url
    raise ValueError(f"Pinned Linux/Python-3.10 wheel not found at {index}")


def measure(group, base, byte_limit, timeout):
    start = time.monotonic()
    url = resolve_payload(group, base, timeout)
    index_seconds = time.monotonic() - start
    download_start = time.monotonic()
    transferred, prefix = 0, b""
    with request(url, timeout, Range=f"bytes=0-{byte_limit - 1}") as response:
        while transferred < byte_limit:
            if time.monotonic() - download_start > timeout:
                raise TimeoutError(f"Download exceeded {timeout:g} seconds")
            chunk = response.read(min(256 * 1024, byte_limit - transferred))
            if not chunk:
                break
            if not prefix:
                prefix = chunk[:4]
            transferred += len(chunk)
        content_type = response.headers.get("Content-Type", "").lower()
        final_host = urlparse(response.geturl()).netloc
    download_seconds = time.monotonic() - download_start
    expected_magic = {"pip": b"PK\x03\x04", "torch": b"PK\x03\x04",
                      "conda": b"\x28\xb5\x2f\xfd", "hf": b"\x1f\x8b"}[group]
    if not prefix.startswith(expected_magic) or "text/html" in content_type:
        raise ValueError("Response is not the expected binary file (possibly a challenge page)")
    if transferred != byte_limit:
        raise ValueError(f"Incomplete sample: {transferred}/{byte_limit} bytes")
    return {
        "bytes": transferred, "index_seconds": round(index_seconds, 3),
        "download_seconds": round(download_seconds, 3),
        "total_seconds": round(time.monotonic() - start, 3),
        "mib_per_second": round(transferred / 1024**2 / download_seconds, 3),
        "final_host": final_host,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--mib", type=int, default=8, help="Read only this many MiB per file")
    parser.add_argument("--timeout", type=float, default=20, help="Network/download timeout in seconds")
    parser.add_argument("--prefer-domestic", action="store_true",
                        help="Select the fastest successful domestic source when available")
    parser.add_argument("--domestic-only", action="store_true",
                        help="Test domestic mirror endpoints only; never fall back to official sources. "
                             "Mirrors may still redirect downloads to overseas CDNs")
    parser.add_argument("--output", type=Path, default=Path("outputs/migration/mirrors.json"))
    parser.add_argument("--write-env", type=Path, help="Write selected URLs as shell exports")
    args = parser.parse_args()
    if args.rounds < 1 or args.mib < 1 or args.timeout <= 0:
        parser.error("rounds, mib and timeout must be positive")
    report = {
        "tested_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": sys.version.split()[0],
        "proxy_configured": any(os.environ.get(key) for key in
                                ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy")),
        "sample_mib": args.mib, "rounds": args.rounds, "timeout_seconds": args.timeout,
        "domestic_only": args.domestic_only,
        "payloads": {group: PAYLOADS[group] for group in args.groups},
        "results": [], "best": {}, "best_domestic": {}, "selected": {},
    }
    for group in dict.fromkeys(args.groups):
        rows = [{"group": group, "name": name, "url": url, "samples": []}
                for name, url in SOURCES[group].items()
                if not args.domestic_only or name != "official"]
        # Sequential requests avoid competition for the same network bandwidth.
        # Alternate source order to reduce a fixed ordering advantage.
        for iteration in range(args.rounds):
            for row in (rows if iteration % 2 == 0 else reversed(rows)):
                print(f"{group:5} {row['name']:9} round {iteration + 1}/{args.rounds}: ", end="", flush=True)
                try:
                    sample = measure(group, row["url"], args.mib * 1024**2, args.timeout)
                    print(f"{sample['mib_per_second']:.2f} MiB/s, total {sample['total_seconds']:.2f}s", flush=True)
                except Exception as exc:
                    sample = {"error": f"{type(exc).__name__}: {exc}"}
                    print(sample["error"], flush=True)
                row["samples"].append(sample)
        for row in rows:
            successful = [sample for sample in row["samples"] if "error" not in sample]
            row["successful_rounds"] = len(successful)
            row["eligible"] = len(successful) == args.rounds
            if successful:
                row["median_total_seconds"] = statistics.median(sample["total_seconds"] for sample in successful)
                row["median_mib_per_second"] = statistics.median(sample["mib_per_second"] for sample in successful)
        eligible = sorted((row for row in rows if row["eligible"]), key=lambda row: row["median_total_seconds"])
        domestic = [row for row in eligible if row["name"] != "official"]
        if eligible:
            report["best"][group] = eligible[0]["url"]
        if domestic:
            report["best_domestic"][group] = domestic[0]["url"]
        selection = domestic if args.prefer_domestic and domestic else eligible
        if selection:
            report["selected"][group] = selection[0]["url"]
        report["results"].extend(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if args.write_env:
        args.write_env.parent.mkdir(parents=True, exist_ok=True)
        args.write_env.write_text("# Generated by scripts/benchmark_download_mirrors.py\n" + "".join(
            f"export {ENV_NAMES[group]}={shlex.quote(url)}\n" for group, url in report["selected"].items()
        ))
    print(f"\nResults: {args.output}")
    for group in args.groups:
        print(f"{group}: {report['selected'].get(group, 'NO SUCCESSFUL SOURCE; inspect errors and retry')}")
    return 0 if all(group in report["selected"] for group in args.groups) else 1


if __name__ == "__main__":
    raise SystemExit(main())

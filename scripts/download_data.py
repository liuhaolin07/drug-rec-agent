"""一键下载原始数据(仅用标准库, 跨平台, 可重复运行)。

下载内容:
  data/raw/mimic_demo/  MIMIC-III demo 的 6 张核心 CSV(公开镜像, 官方源为 PhysioNet)
  data/raw/ddinter/     DDInter 8 个 ATC 大类的 DDI CSV(官方站公开下载)

用法:
  .venv/Scripts/python.exe scripts/download_data.py          # 跳过已存在文件
  .venv/Scripts/python.exe scripts/download_data.py --force  # 强制重新下载
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIMIC_DIR = ROOT / "data" / "raw" / "mimic_demo"
DDI_DIR = ROOT / "data" / "raw" / "ddinter"

MIMIC_BASE = "https://raw.githubusercontent.com/dalwari/mimic-iii-clinical-database-demo-1.4/main"
MIMIC_FILES = ["PATIENTS.csv", "ADMISSIONS.csv", "PRESCRIPTIONS.csv",
               "DIAGNOSES_ICD.csv", "D_ICD_DIAGNOSES.csv", "ICUSTAYS.csv"]
DDINTER_BASE = "https://ddinter.scbdd.com/static/media/download"
DDINTER_FILES = [f"ddinter_downloads_code_{c}.csv" for c in "ABDHLPRV"]

UA = {"User-Agent": "drug-rec-agent/0.1 (research prototype; data fetcher)"}


def fetch(url: str, dest: Path, force: bool = False) -> bool:
    if dest.exists() and not force and dest.stat().st_size > 1000:
        print(f"  [skip] {dest.relative_to(ROOT)} 已存在 ({dest.stat().st_size:,} B)")
        return True
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"  [ok]   {dest.relative_to(ROOT)}  <- {len(data):,} B")
        return True
    except Exception as e:
        print(f"  [FAIL] {url}\n         {type(e).__name__}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()

    print("[1/2] MIMIC-III demo (真实ICU记录, ODbL; 官方源 PhysioNet)")
    ok1 = all(fetch(f"{MIMIC_BASE}/{f}", MIMIC_DIR / f, args.force) for f in MIMIC_FILES)

    print("[2/2] DDInter (药物相互作用, 学术用途; 官方源 ddinter.scbdd.com)")
    ok2 = all(fetch(f"{DDINTER_BASE}/{f}", DDI_DIR / f, args.force) for f in DDINTER_FILES)

    if ok1 and ok2:
        print("\n[done] 全部数据就绪, 可以开始: python src/prep_data.py")
    else:
        print("\n[warn] 有文件未下载成功(常见原因: 网络/代理), 可重复运行本脚本补齐", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

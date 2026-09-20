"""原始数据体检:DDInter(药物相互作用库) 与 MIMIC-III demo 的字段、规模、可匹配性。

运行:  .venv/Scripts/python.exe src/inspect_data.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def main() -> None:
    print("=" * 72)
    print("[1] DDInter —— 药物-药物相互作用(DDI)真实数据库")
    print("=" * 72)
    frames, total, levels = [], 0, {}
    for f in sorted((RAW / "ddinter").glob("*.csv")):
        df = pd.read_csv(f)
        lvl_col = "Level" if "Level" in df.columns else df.columns[-1]
        total += len(df)
        for k, v in df[lvl_col].value_counts().items():
            levels[k] = levels.get(k, 0) + int(v)
        frames.append(df)
    all_ddi = pd.concat(frames, ignore_index=True)
    print(f"文件 {len(frames)} 个 | DDI 记录总数 {total}")
    print(f"严重程度分布: {levels}")
    print("列名:", list(all_ddi.columns))
    print(all_ddi.head(4).to_string())
    ddi_drugs = set(all_ddi["Drug_A"].str.strip().str.lower()) | set(all_ddi["Drug_B"].str.strip().str.lower())
    print(f"库中不同药名数: {len(ddi_drugs)}")

    print()
    print("=" * 72)
    print("[2] MIMIC-III demo —— 真实 ICU 住院记录")
    print("=" * 72)
    p = pd.read_csv(RAW / "mimic_demo" / "PRESCRIPTIONS.csv")
    diag = pd.read_csv(RAW / "mimic_demo" / "DIAGNOSES_ICD.csv")
    adm = pd.read_csv(RAW / "mimic_demo" / "ADMISSIONS.csv")
    pat = pd.read_csv(RAW / "mimic_demo" / "PATIENTS.csv")
    print(f"患者 {pat.subject_id.nunique()} | 住院 {len(adm)} | 处方 {len(p)} 行 | 诊断 {len(diag)} 行")
    print(f"不同 drug 字段名 {p['drug'].nunique()} | drug_name_generic {p['drug_name_generic'].nunique()}")
    print(f"不同 icd9_code {diag.icd9_code.nunique()}")
    per_hadm = p.groupby("hadm_id")["drug"].nunique()
    cond_per = diag.groupby("hadm_id")["icd9_code"].nunique()
    print(f"每次住院用药数: 中位 {per_hadm.median():.0f} 均值 {per_hadm.mean():.1f} 最大 {per_hadm.max()}")
    print(f"每次住院诊断数: 中位 {cond_per.median():.0f} 均值 {cond_per.mean():.1f} 最大 {cond_per.max()}")

    print()
    print("最高频 drug(前 25):")
    for name, c in p["drug"].str.strip().value_counts().head(25).items():
        print(f"  {c:4d}  {name}")

    print()
    print("--- 两库药名可匹配性预检(小写精确匹配) ---")
    mimic_drugs = set(p["drug"].str.strip().str.lower())
    mimic_drugs |= set(p["drug_name_generic"].dropna().str.strip().str.lower())
    inter = ddi_drugs & mimic_drugs
    print(f"DDInter 与 MIMIC 药名交集: {len(inter)} 个")
    print("样例:", sorted(inter)[:24])


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main()

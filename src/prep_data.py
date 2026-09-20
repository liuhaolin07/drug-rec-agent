"""数据层:把 MIMIC-III demo(真实ICU记录) + DDInter(真实DDI库) 加工成异构图工件。

数据来源与许可:
  - MIMIC-III Clinical Database Demo v1.4 (MIT LCP, ODbL), 100患者/129住院。
  - DDInter 药物相互作用数据库 (scbdd.com), 8个ATC大类共 22 万条 DDI + 严重度。

产出 (data/processed/):
  drugs.csv          药物节点 (drug_idx, name, display, ddi_matched, n_adm, n_presc)
  conditions.csv     诊断节点 (cond_idx, icd9_code, title, n_adm)
  edges_dc.csv       药物-诊断 边 (drug_idx, cond_idx, weight=共现住院数)
  edges_copresc.csv  药物-药物 共处方边 (d1<d2, weight)
  ddi_pairs.csv      药物-药物 真实相互作用边 (d1<d2, level, id_a, id_b)
  admissions.csv     住院摘要 (hadm_id, subject_id, ..., n_drugs, n_conds)
  task_split.json    按患者切分的 train/test 住院
  stats.json         全量统计口径(README与报告引用它)
运行:  .venv/Scripts/python.exe src/prep_data.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

from common import (JUNK_TOKENS, PROC, RAW, SEED, clean_drug_name, display_name,
                    is_fluid, norm_name, save_json, set_seed, strip_salts)

MIMIC = RAW / "mimic_demo"
DDINTER = RAW / "ddinter"

# 人工校对过的别名(两库叫法不同, 临床上等价; 依据: 美国通用名 vs 国际非专利名/数据库命名习惯)
ALIASES = {
    "aspirin": "acetylsalicylic acid",      # 阿司匹林
    "albuterol": "salbutamol",              # 沙丁胺醇(US -> INN)
    "insulin": "insulin human",             # 常规人胰岛素
    "amphotericin b liposomal": "amphotericin b",
}


# ------------------------------------------------------------------ DDInter
def load_ddinter():
    """返回 (names: {lower->display}, pairs: {(lo,hi): (level,id_a,id_b)})"""
    frames = [pd.read_csv(f) for f in sorted(DDINTER.glob("*.csv"))]
    ddi = pd.concat(frames, ignore_index=True)
    names: dict[str, str] = {}
    pairs: dict[tuple[str, str], tuple[str, str, str]] = {}
    for a, b, lvl, ia, ib in zip(ddi.Drug_A, ddi.Drug_B, ddi.Level, ddi.DDInterID_A, ddi.DDInterID_B):
        la, lb = norm_name(a), norm_name(b)
        names.setdefault(la, a.strip())
        names.setdefault(lb, b.strip())
        if la == lb:
            continue
        key = (la, lb) if la < lb else (lb, la)
        pairs[key] = (lvl, ia, ib)  # 重复出现保留后者(行数极少)
    return names, pairs


def build_stripped_index(names: dict[str, str]) -> dict[str, list[str]]:
    """盐型剥离后的倒排索引, 用于二档匹配; 必须唯一命中才采用。"""
    idx: dict[str, list[str]] = defaultdict(list)
    for lo in names:
        s = strip_salts(lo)
        if s and s != lo:
            idx[s].append(lo)
    return idx


def expand_candidates(raw_drug: str, raw_gen: str) -> list[str]:
    """把 MIMIC 的两个药名字段展开成"从严格到宽松"的候选序列。"""
    seq: list[str] = []
    for c in (raw_gen, raw_drug):
        if not c:
            continue
        forms = [norm_name(c), clean_drug_name(c), strip_salts(clean_drug_name(c))]
        # 特殊: 任何 insulin 变体归到常规胰岛素
        if any(x.startswith("insulin") for x in forms if x):
            forms.append("insulin")
        for x in forms:
            if x and x not in seq:
                seq.append(x)
    return seq


def resolve(raw_drug: str, raw_gen: str, ddi_names, stripped_index):
    """级联匹配: 精确 -> 别名 -> 盐型剥离(唯一命中)。返回 (canonical_lower, matched)。"""
    for x in expand_candidates(raw_drug, raw_gen):
        if x in ddi_names:
            return x, True
        if x in ALIASES and ALIASES[x] in ddi_names:
            return ALIASES[x], True
        if len(x) >= 4 and x in stripped_index and len(stripped_index[x]) == 1:
            return stripped_index[x][0], True
    base = clean_drug_name(raw_gen) or clean_drug_name(raw_drug) or norm_name(raw_drug)
    return base, False


# -------------------------------------------------------------------- 主流程
def main() -> None:
    set_seed(SEED)
    stats: dict[str, object] = {}

    # ---- 1. DDInter
    ddi_names, ddi_pairs = load_ddinter()
    stripped_index = build_stripped_index(ddi_names)
    stats["ddinter_records"] = len(ddi_pairs)
    stats["ddinter_drug_names"] = len(ddi_names)
    print(f"[1] DDInter: {len(ddi_pairs)} 条相互作用, {len(ddi_names)} 种药名")

    # ---- 2. MIMIC
    presc = pd.read_csv(MIMIC / "PRESCRIPTIONS.csv")
    diag = pd.read_csv(MIMIC / "DIAGNOSES_ICD.csv", dtype={"icd9_code": str})
    dicd = pd.read_csv(MIMIC / "D_ICD_DIAGNOSES.csv", dtype={"icd9_code": str})
    adm = pd.read_csv(MIMIC / "ADMISSIONS.csv")
    pat = pd.read_csv(MIMIC / "PATIENTS.csv")
    presc["drug"] = presc["drug"].astype(str).str.strip()
    presc["drug_name_generic"] = presc["drug_name_generic"].fillna("").astype(str).str.strip()
    presc = presc[presc["drug"].ne("") & presc["drug"].ne("nan")].copy()

    n_before = len(presc)
    presc = presc[~presc["drug"].map(is_fluid)].copy()
    stats["presc_rows_raw"] = int(n_before)
    stats["presc_rows_after_fluid_filter"] = int(len(presc))
    print(f"[2] 处方行 {n_before} -> 去输液后 {len(presc)}")

    # ---- 3. 药名规范化 + DDInter 匹配
    canons = [resolve(d, g, ddi_names, stripped_index)
              for d, g in zip(presc["drug"], presc["drug_name_generic"])]
    presc["canon"] = [c for c, _ in canons]
    presc["matched"] = [m for _, m in canons]

    # 垃圾药名过滤(如 'Vial' 这种字段污染)
    junk = [c for c in presc["canon"].unique()
            if len(c) < 4 or c in JUNK_TOKENS]
    n_before_junk = len(presc)
    presc = presc[~presc["canon"].isin(junk)].copy()
    stats["junk_drug_names_dropped"] = len(junk)
    stats["junk_examples"] = sorted(junk)[:12]
    stats["presc_rows_after_junk_filter"] = int(len(presc))
    print(f"[3] 药名清洗后 {n_before_junk} -> {len(presc)} 行 (剔除垃圾名 {len(junk)} 个)")

    # 每次住院内药去重(同名药多次开立只算一次)
    per_adm = presc[["hadm_id", "canon", "matched"]].drop_duplicates(["hadm_id", "canon"])
    n_adm_of = per_adm.groupby("canon")["hadm_id"].nunique()

    # 保留出现在 >=2 次住院的药物
    kept = sorted([c for c, n in n_adm_of.items() if n >= 2])
    stats["drugs_before_freq_filter"] = int(len(n_adm_of))
    stats["drugs_kept_min2adm"] = len(kept)
    per_adm = per_adm[per_adm["canon"].isin(kept)]
    matched_set = {c for c in kept if bool(per_adm.loc[per_adm.canon.eq(c), "matched"].any())}
    stats["drugs_matched_to_ddinter"] = len(matched_set)
    stats["drug_match_rate"] = round(len(matched_set) / max(len(kept), 1), 3)
    stats["presc_row_match_rate"] = round(float(per_adm["matched"].mean()), 3)
    print(f"[4] 药物节点 {len(kept)} 个 (匹配 DDInter {len(matched_set)} 个, "
          f"匹配率 {stats['drug_match_rate']:.1%})")

    unmatched = sorted([c for c in kept if c not in matched_set], key=lambda c: -n_adm_of[c])
    stats["unmatched_drug_examples"] = [(c, int(n_adm_of[c])) for c in unmatched[:15]]

    # ---- 4. 节点表
    drug_idx = {c: i for i, c in enumerate(sorted(kept, key=lambda c: (-n_adm_of[c], c)))}
    drugs = pd.DataFrame({
        "drug_idx": [drug_idx[c] for c in drug_idx],
        "name": list(drug_idx.keys()),
        "display": [display_name(c) for c in drug_idx],
        "ddi_matched": [c in matched_set for c in drug_idx],
        "n_adm": [int(n_adm_of[c]) for c in drug_idx],
        "n_presc": [int((presc["canon"] == c).sum()) for c in drug_idx],
    }).sort_values("drug_idx").reset_index(drop=True)
    drugs.to_csv(PROC / "drugs.csv", index=False, encoding="utf-8")

    cond_title = dict(zip(dicd.icd9_code, dicd.long_title.fillna(dicd.short_title).fillna("")))
    diag = diag[diag.icd9_code.notna() & diag.icd9_code.ne("nan")]
    n_adm_cond = diag.groupby("icd9_code")["hadm_id"].nunique()
    cond_order = sorted(n_adm_cond.index, key=lambda c: (-n_adm_cond[c], c))
    cond_idx = {c: i for i, c in enumerate(cond_order)}
    conds = pd.DataFrame({
        "cond_idx": [cond_idx[c] for c in cond_order],
        "icd9_code": cond_order,
        "title": [str(cond_title.get(c, c)) for c in cond_order],
        "n_adm": [int(n_adm_cond[c]) for c in cond_order],
    })
    conds.to_csv(PROC / "conditions.csv", index=False, encoding="utf-8")
    print(f"[5] 诊断节点 {len(conds)} 个")

    # ---- 5. 边: 药物-诊断 / 共处方
    adm_drugs = per_adm.groupby("hadm_id")["canon"].apply(lambda s: sorted({drug_idx[c] for c in s}))
    adm_conds = diag.groupby("hadm_id")["icd9_code"].apply(lambda s: sorted({cond_idx[c] for c in s}))
    common_adm = sorted(set(adm_drugs.index) & set(adm_conds.index))

    dc_cnt: dict[tuple[int, int], int] = defaultdict(int)
    cp_cnt: dict[tuple[int, int], int] = defaultdict(int)
    for h in common_adm:
        ds, cs = adm_drugs[h], adm_conds[h]
        for d in ds:
            for c in cs:
                dc_cnt[(d, c)] += 1
        for d1, d2 in combinations(ds, 2):
            cp_cnt[(d1, d2)] += 1

    pd.DataFrame([(d, c, w) for (d, c), w in sorted(dc_cnt.items())],
                 columns=["drug_idx", "cond_idx", "weight"]).to_csv(
        PROC / "edges_dc.csv", index=False, encoding="utf-8")
    pd.DataFrame([(d1, d2, w) for (d1, d2), w in sorted(cp_cnt.items())],
                 columns=["d1", "d2", "weight"]).to_csv(
        PROC / "edges_copresc.csv", index=False, encoding="utf-8")
    stats["edges_drug_condition"] = len(dc_cnt)
    stats["edges_coprescription"] = len(cp_cnt)
    print(f"[6] 边: 药物-诊断 {len(dc_cnt)}, 共处方 {len(cp_cnt)}")

    # 每次住院的 药物/诊断 id 列表(推荐模型训练与评估直接使用)
    save_json({str(int(h)): [int(x) for x in v] for h, v in adm_drugs.items()},
              PROC / "adm_drug_ids.json")
    save_json({str(int(h)): [int(x) for x in v] for h, v in adm_conds.items()},
              PROC / "adm_cond_ids.json")

    # ---- 6. 真实 DDI 边(限制在我们药物集合内)
    name2idx = {c: drug_idx[c] for c in drug_idx}
    ddi_rows, level_cnt = [], defaultdict(int)
    for (lo, hi), (lvl, ia, ib) in ddi_pairs.items():
        if lo in name2idx and hi in name2idx:
            d1, d2 = sorted((name2idx[lo], name2idx[hi]))
            if d1 == d2:
                continue
            ddi_rows.append((d1, d2, lvl, ia, ib))
            level_cnt[lvl] += 1
    pd.DataFrame(ddi_rows, columns=["d1", "d2", "level", "id_a", "id_b"]).to_csv(
        PROC / "ddi_pairs.csv", index=False, encoding="utf-8")
    stats["ddi_edges_in_graph"] = len(ddi_rows)
    stats["ddi_level_distribution"] = dict(sorted(level_cnt.items()))
    print(f"[7] 图内 DDI 边 {len(ddi_rows)} 条, 严重度分布 {dict(level_cnt)}")

    # ---- 7. 住院摘要 + 划分(按患者切分,避免同患者泄漏)
    adm_small = adm[["hadm_id", "subject_id", "admittime", "dischtime"]].copy()
    adm_small["los_days"] = (
        (pd.to_datetime(adm_small.dischtime) - pd.to_datetime(adm_small.admittime)).dt.total_seconds() / 86400
    ).round(2)
    adm_small["n_drugs"] = adm_small["hadm_id"].map(adm_drugs.apply(len)).fillna(0).astype(int)
    adm_small["n_conds"] = adm_small["hadm_id"].map(adm_conds.apply(len)).fillna(0).astype(int)
    adm_small = adm_small[adm_small["hadm_id"].isin(common_adm)].sort_values("hadm_id")
    adm_small.to_csv(PROC / "admissions.csv", index=False, encoding="utf-8")

    rng = __import__("numpy").random.default_rng(SEED)
    subjects = sorted(adm_small["subject_id"].unique())
    test_subjects = sorted(rng.choice(subjects, size=max(1, int(len(subjects) * 0.2)), replace=False).tolist())
    tr = adm_small[~adm_small.subject_id.isin(test_subjects)]["hadm_id"].tolist()
    te = adm_small[adm_small.subject_id.isin(test_subjects)]["hadm_id"].tolist()
    split = {"train_hadm": [int(x) for x in tr], "test_hadm": [int(x) for x in te],
             "test_subjects": [int(x) for x in test_subjects], "seed": SEED}
    save_json(split, PROC / "task_split.json")
    stats["n_admissions_total"] = int(len(adm_small))
    stats["n_admissions_train"] = len(tr)
    stats["n_admissions_test"] = len(te)
    print(f"[8] 住院 {len(adm_small)} 次 -> train {len(tr)} / test {len(te)} (按患者切分)")

    stats["patients"] = int(pat.subject_id.nunique())
    save_json(stats, PROC / "stats.json")
    print("\n[done] 工件已写入 data/processed/")
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:2400])


if __name__ == "__main__":
    main()

"""Agent 工具集: 把图表示学习模型、DDI 库、病历数据、openFDA 封装成 LLM 可调用的工具。

设计原则:
  - 一切结论可回溯到"证据"(诊断-药物边权重 / DDInter 记录 / openFDA 原文);
  - 安全约束在工具层强制执行(GNN 推荐自动排除与已知用药冲突的候选);
  - 工具错误不抛出, 返回带 error 字段的 JSON, 让模型自行修正。
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common import (PROC, RAW, RES, SEED, clean_drug_name, norm_name,  # noqa: E402
                    strip_salts)
from prep_data import expand_candidates, load_ddinter  # noqa: E402
from rec_model import (HeteroRec, build_adj_dc, build_adj_cp,  # noqa: E402
                       ddi_partner_maps, make_scenarios, scores_for_adm)


class DrugRecEnv:
    """封装一次会话所需的全部数据与模型; 每个工具返回 JSON 字符串。"""

    def __init__(self, scenario: str = "half", seed: int = SEED):
        # ---- 图与实体表
        self.drugs = pd.read_csv(PROC / "drugs.csv")
        self.conds = pd.read_csv(PROC / "conditions.csv")
        self.adm = pd.read_csv(PROC / "admissions.csv", dtype={"hadm_id": int, "subject_id": int})
        dc = pd.read_csv(PROC / "edges_dc.csv")
        cp = pd.read_csv(PROC / "edges_copresc.csv")
        ddi_graph = pd.read_csv(PROC / "ddi_pairs.csv")
        split = json.loads((PROC / "task_split.json").read_text(encoding="utf-8"))
        self.drug_ids = {int(k): v for k, v in json.loads((PROC / "adm_drug_ids.json").read_text(encoding="utf-8")).items()}
        self.cond_ids = {int(k): v for k, v in json.loads((PROC / "adm_cond_ids.json").read_text(encoding="utf-8")).items()}

        self.n_drug, self.n_cond = len(self.drugs), len(self.conds)
        self.name2idx = {str(r.name): int(r.drug_idx) for r in self.drugs.itertuples()}
        self.display2idx = {norm_name(str(r.display)): int(r.drug_idx) for r in self.drugs.itertuples()}
        self.idx2display = {int(r.drug_idx): str(r.display) for r in self.drugs.itertuples()}
        self.idx2n_adm = {int(r.drug_idx): int(r.n_adm) for r in self.drugs.itertuples()}
        cond_rows = {int(r.cond_idx): (str(r.icd9_code), str(r.title)) for r in self.conds.itertuples()}
        self.cond_rows = cond_rows

        # ---- 药物-诊断边(用于"证据链"): drug_idx -> [(cond_idx, weight)]
        self.dc_by_drug: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r in dc.itertuples():
            self.dc_by_drug[int(r.drug_idx)].append((int(r.cond_idx), int(r.weight)))

        # ---- DDI 全库 + 图内安全约束
        self.ddi_names, self.ddi_pairs = load_ddinter()          # 全库 16 万对
        self.hard, self.soft = ddi_partner_maps(ddi_graph)       # 图内 Major / Major+Moderate

        # ---- 推荐模型(实验二 checkpoint)
        ckpt = torch.load(RES / "rec_model.pt", map_location="cpu", weights_only=False)
        self.model = HeteroRec(ckpt["n_drug"], ckpt["n_cond"], ckpt["dim"], ckpt["rounds"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.a_dc = build_adj_dc(dc, self.n_drug, self.n_cond)
        self.a_cp = build_adj_cp(cp, self.n_drug)

        # ---- 场景(已知用药 / 隐藏目标): 供评测控制泄漏
        pos_sets = {int(h): set(v) for h, v in self.drug_ids.items()}
        self.scen = make_scenarios(pos_sets)[scenario]
        self.train_hadm = [int(x) for x in split["train_hadm"]]

        # ---- 患者基本信息(原始表, 仅取性别/年龄)
        pat = pd.read_csv(RAW / "mimic_demo" / "PATIENTS.csv")
        adm_raw = pd.read_csv(RAW / "mimic_demo" / "ADMISSIONS.csv")
        dob = dict(zip(pat.subject_id, pd.to_datetime(pat.dob)))
        gender = dict(zip(pat.subject_id, pat.gender))
        self.subj_of = {int(r.hadm_id): int(r.subject_id) for r in adm_raw.itertuples()}
        self.admittime = {int(r.hadm_id): pd.to_datetime(r.admittime) for r in adm_raw.itertuples()}
        self.dob, self.gender = dob, gender

        self._openfda_cache: dict[str, dict] = {}

    # ================================================================= 工具实现
    def patient_profile(self, hadm_id: int) -> dict:
        """住院概况: 诊断列表 + 已知用药 + 基本信息。"""
        hadm_id = int(hadm_id)
        if hadm_id not in self.cond_ids:
            return {"error": f"hadm_id {hadm_id} 不存在"}
        conds = [{"icd9": self.cond_rows[c][0], "title": self.cond_rows[c][1]}
                 for c in self.cond_ids[hadm_id]]
        known = sorted(self.scen[hadm_id][0])
        subj = self.subj_of.get(hadm_id)
        age = None
        if subj in self.dob and hadm_id in self.admittime:
            age = float(np.round((self.admittime[hadm_id] - self.dob[subj]).days / 365.25, 1))
        return {
            "hadm_id": hadm_id,
            "patient_age_bucket": ">=89" if (age or 0) >= 89 else age,  # MIMIC 90+ 岁统一模糊化
            "gender": self.gender.get(subj, "?"),
            "n_conditions": len(conds),
            "conditions": conds,
            "known_medications": [self.idx2display[d] for d in known],
            "note": "known_medications 为患者当前/已知在用药清单; 任务是在此基础上推荐需要补充的药物",
        }

    def recommend_drugs(self, hadm_id: int, k: int = 10) -> dict:
        """GNN 打分推荐(安全重排): 自动排除与已知用药存在 Major/Moderate 冲突的候选。"""
        hadm_id = int(hadm_id)
        if hadm_id not in self.cond_ids:
            return {"error": f"hadm_id {hadm_id} 不存在"}
        cids = self.cond_ids[hadm_id]
        if not cids:
            return {"error": "该住院无诊断记录"}
        known = self.scen[hadm_id][0]
        scores = scores_for_adm(self.model, self.a_dc, self.a_cp, cids).copy()

        excluded = []
        for d in range(self.n_drug):
            partners = self.soft.get(d, set()) & known
            if partners:
                scores[d] = -1e9
                if len(excluded) < 8:
                    worst = "Major" if self.hard.get(d, set()) & known else "Moderate"
                    excluded.append({
                        "drug": self.idx2display[d],
                        "conflicts_with": [self.idx2display[p] for p in sorted(partners)[:3]],
                        "max_level": worst,
                    })

        order = np.argsort(-scores, kind="stable")[:max(1, int(k))]
        my_conds = set(cids)
        recs = []
        for d in order:
            if scores[d] <= -1e8:
                continue
            support = sorted([(c, w) for c, w in self.dc_by_drug.get(int(d), []) if c in my_conds],
                             key=lambda x: -x[1])[:3]
            recs.append({
                "drug": self.idx2display[int(d)],
                "score": round(float(scores[d]), 3),
                "historical_use_admissions": self.idx2n_adm[int(d)],
                "support_conditions": [
                    {"title": self.cond_rows[c][1], "cooccur_admissions": int(w)} for c, w in support
                ],
            })
        return {
            "method": "GNN 图表示学习打分 + DDI 安全重排(排除与已知用药 Major/Moderate 冲突)",
            "recommendations": recs,
            "excluded_unsafe_candidates": excluded,
            "note": "score 为模型原始打分(越高越匹配), support_conditions 给出证据链; 结果需药师复核",
        }

    def check_ddi(self, drugs: list[str]) -> dict:
        """核对药物清单内的两两相互作用(依据 DDInter 数据库)。"""
        if isinstance(drugs, str):
            drugs = [drugs]
        resolved, unresolved = [], []
        for nm in drugs:
            r = self._resolve_ddi_name(str(nm))
            (resolved if r else unresolved).append((str(nm), r) if r else str(nm))
        found = []
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                na, la = resolved[i]
                nb, lb = resolved[j]
                lo, hi = (la, lb) if la < lb else (lb, la)
                rec = self.ddi_pairs.get((lo, hi))
                if rec:
                    lvl, ia, ib = rec
                    found.append({"drug_a": na, "drug_b": nb, "level": lvl,
                                  "ddinter_ids": [ia, ib]})
        order = {"Major": 0, "Moderate": 1, "Minor": 2, "Unknown": 3}
        found.sort(key=lambda x: order.get(x["level"], 9))
        return {
            "source": "DDInter 数据库(本地 16 万对记录)",
            "n_checked_pairs": len(resolved) * (len(resolved) - 1) // 2,
            "interactions": found,
            "unresolved_names": unresolved,
            "note": "unresolved_names 表示该药不在 DDI 库覆盖范围内, 需人工核对; 无记录不等于绝对安全",
        }

    def drug_info(self, drug: str) -> dict:
        """药物档案: 本地使用统计 + DDI 概况 + openFDA 说明书原文(适应症/警告/相互作用)。"""
        idx = self._resolve_graph_name(drug)
        local = {"in_demo_drug_set": idx is not None}
        if idx is not None:
            local.update({
                "display": self.idx2display[idx],
                "historical_use_admissions": int(self.drugs.iloc[idx]["n_adm"]),
                "frequent_support_conditions": [
                    {"title": self.cond_rows[c][1], "cooccur_admissions": int(w)}
                    for c, w in sorted(self.dc_by_drug.get(idx, []), key=lambda x: -x[1])[:5]
                ],
            })
        ddi_name = self._resolve_ddi_name(drug)
        ddi_profile = {}
        if ddi_name:
            cnt = defaultdict(int)
            for (lo, hi), (lvl, _, _) in self.ddi_pairs.items():
                if lo == ddi_name or hi == ddi_name:
                    cnt[lvl] += 1
            ddi_profile = {"ddinter_name": ddi_name, "partner_count_by_level": dict(cnt)}
        return {
            "drug": drug,
            "local": local,
            "ddi_profile": ddi_profile,
            "fda_label": self._openfda_lookup(drug if ddi_name is None else ddi_name),
            "source": "MIMIC-III demo + DDInter + openFDA drug label API",
        }

    def find_similar_admissions(self, hadm_id: int, k: int = 5) -> dict:
        """检索历史相似病例(仅训练集), 给出共享诊断与既往用药模式(证据检索)。"""
        hadm_id = int(hadm_id)
        if hadm_id not in self.cond_ids:
            return {"error": f"hadm_id {hadm_id} 不存在"}
        mine = set(self.cond_ids[hadm_id])
        subj = self.subj_of.get(hadm_id)
        scored = []
        for h in self.train_hadm:
            if h == hadm_id or self.subj_of.get(h) == subj:
                continue
            hs = set(self.cond_ids.get(h, []))
            if not hs:
                continue
            inter = mine & hs
            if not inter:
                continue
            scored.append((len(inter) / len(mine | hs), h, inter))
        scored.sort(reverse=True)
        top = scored[:max(1, int(k))]
        drug_freq = defaultdict(int)
        for _, h, _ in top:
            for d in set(self.drug_ids.get(h, [])):
                drug_freq[d] += 1
        return {
            "query_hadm": hadm_id,
            "similar_cases": [
                {"hadm_id": h, "shared_conditions": [self.cond_rows[c][1] for c in sorted(inter, key=lambda c: -self.cond_rows[c][1].count(" "))[:3]],
                 "jaccard": round(s, 3)}
                for s, h, inter in top
            ],
            "frequent_drugs_in_similar_cases": [
                {"drug": self.idx2display[d], "in_n_of_top_cases": n}
                for d, n in sorted(drug_freq.items(), key=lambda x: -x[1])[:8]
            ],
            "source": "MIMIC-III demo 训练集住院(不含当前病例本体)",
        }

    # ================================================================= 内部工具
    def _resolve_graph_name(self, name: str):
        n = norm_name(name)
        if n in self.name2idx:
            return self.name2idx[n]
        if n in self.display2idx:
            return self.display2idx[n]
        for cand in expand_candidates(name, ""):
            if cand in self.name2idx:
                return self.name2idx[cand]
        return None

    def _resolve_ddi_name(self, name: str):
        for cand in expand_candidates(name, ""):
            if cand in self.ddi_names:
                return cand
        return None

    def _openfda_lookup(self, name: str) -> dict:
        if name in self._openfda_cache:
            return self._openfda_cache[name]
        q = urllib.parse.quote(f'openfda.generic_name:"{name}"')
        url = f"https://api.fda.gov/drug/label.json?limit=1&search={q}"
        out: dict = {}
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "drug-rec-agent/0.1 (research prototype)"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read())
            res = data.get("results", [])
            if res:
                item = res[0]
                def grab(field, n=420):
                    v = item.get(field)
                    if isinstance(v, list) and v:
                        return " ".join(str(x) for x in v)[:n]
                    return None
                out = {
                    "generic_name": (item.get("openfda", {}).get("generic_name") or [None])[0],
                    "indications_and_usage": grab("indications_and_usage"),
                    "warnings": grab("warnings") or grab("boxed_warning"),
                    "drug_interactions": grab("drug_interactions"),
                    "source_url": url,
                }
            else:
                out = {"note": "openFDA 未收录该药说明书"}
        except Exception as e:
            out = {"error": f"openFDA 请求失败: {type(e).__name__}: {e}"}
        self._openfda_cache[name] = out
        return out

    # ================================================================= 分发
    def call_tool(self, name: str, args: dict) -> str:
        fn = {
            "patient_profile": self.patient_profile,
            "recommend_drugs": self.recommend_drugs,
            "check_ddi": self.check_ddi,
            "drug_info": self.drug_info,
            "find_similar_admissions": self.find_similar_admissions,
        }.get(name)
        if fn is None:
            return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)
        try:
            return json.dumps(fn(**args), ensure_ascii=False)
        except Exception as e:  # 工具错误交给模型修正
            return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "patient_profile",
        "description": "读取某次住院的概况: 诊断列表(ICD-9)、当前已知用药清单、年龄/性别。推荐前必须先调用。",
        "parameters": {"type": "object", "properties": {
            "hadm_id": {"type": "integer", "description": "住院号(hadm_id)"}},
            "required": ["hadm_id"]}}},
    {"type": "function", "function": {
        "name": "recommend_drugs",
        "description": "调用图神经网络(GNN)对候选药物打分并返回 top-k 推荐, 自动排除与已知用药存在 Major/Moderate 相互作用的候选, 附证据链(支持诊断)。",
        "parameters": {"type": "object", "properties": {
            "hadm_id": {"type": "integer", "description": "住院号"},
            "k": {"type": "integer", "description": "返回个数, 默认 10"}},
            "required": ["hadm_id"]}}},
    {"type": "function", "function": {
        "name": "check_ddi",
        "description": "核对一批药物两两之间的相互作用(DDInter 数据库, 含严重程度)。对最终推荐清单必须调用一次。用药名(英文)。",
        "parameters": {"type": "object", "properties": {
            "drugs": {"type": "array", "items": {"type": "string"},
                      "description": "药物英文名列表, 例如 ['Furosemide','Heparin','Warfarin']"}},
            "required": ["drugs"]}}},
    {"type": "function", "function": {
        "name": "drug_info",
        "description": "查询单个药物的档案: 历史使用次数、高频支持诊断、DDI 伙伴数量、FDA 说明书原文(适应症/警告/相互作用)。",
        "parameters": {"type": "object", "properties": {
            "drug": {"type": "string", "description": "药物英文名"}},
            "required": ["drug"]}}},
    {"type": "function", "function": {
        "name": "find_similar_admissions",
        "description": "在历史病例库(训练集)中检索与当前住院相似的既往病例, 返回共享诊断与这些病例的常用药(检索证据)。",
        "parameters": {"type": "object", "properties": {
            "hadm_id": {"type": "integer", "description": "当前住院号"},
            "k": {"type": "integer", "description": "返回病例数, 默认 5"}},
            "required": ["hadm_id"]}}},
]

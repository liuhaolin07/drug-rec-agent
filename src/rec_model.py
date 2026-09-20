"""实验二:药物推荐(GNN 打分 + 安全约束重排)与基线对比。

两个评测场景(测试集 34 次住院, 按患者切分):
  full: 已知=全部真实用药, 目标=全部真实用药(宽松上界, 反映"排序质量")
  half: 已知=真实用药的随机一半, 目标=另一半(贴近真实场景: 有在用药清单, 推荐新增药)
指标: Recall@10/20、Precision@10 对照隐藏目标;
      安全违规率@10 = top-10 药中与"已知用药"存在真实 DDI(Major / Major+Moderate)的比例。
方法: popular / cond_cooccur / gnn(图表示学习) / gnn+ddi_mask(安全重排)
运行:  .venv/Scripts/python.exe src/rec_model.py
"""
from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import PROC, RES, SEED, save_json, set_seed

DIM = 64
ROUNDS = 2
EPOCHS = 150
LR = 0.02
NEG_PER_POS = 2
SEEDS = [42, 43, 44]
TOPK = 10
DDI_HARD = {"Major"}                 # 硬约束
DDI_SOFT = {"Major", "Moderate"}     # 安全重排使用


# ------------------------------------------------------------------ 数据
def load_all():
    drugs = pd.read_csv(PROC / "drugs.csv")
    conds = pd.read_csv(PROC / "conditions.csv")
    dc = pd.read_csv(PROC / "edges_dc.csv")
    cp = pd.read_csv(PROC / "edges_copresc.csv")
    ddi = pd.read_csv(PROC / "ddi_pairs.csv")
    adm = pd.read_csv(PROC / "admissions.csv", dtype={"hadm_id": int, "subject_id": int})
    split = json.loads((PROC / "task_split.json").read_text(encoding="utf-8"))
    drug_ids = {int(k): v for k, v in json.loads((PROC / "adm_drug_ids.json").read_text(encoding="utf-8")).items()}
    cond_ids = {int(k): v for k, v in json.loads((PROC / "adm_cond_ids.json").read_text(encoding="utf-8")).items()}
    return drugs, conds, dc, cp, ddi, adm, split, drug_ids, cond_ids


def build_adj_dc(dc: pd.DataFrame, n_drug: int, n_cond: int) -> torch.Tensor:
    """药物-诊断 二部图对称归一化邻接 [n_drug, n_cond]。"""
    u = torch.from_numpy(dc["drug_idx"].to_numpy()).long()
    v = torch.from_numpy(dc["cond_idx"].to_numpy()).long()
    w = torch.from_numpy(dc["weight"].to_numpy().astype(np.float32))
    dr = torch.zeros(n_drug).scatter_add_(0, u, w).pow(-0.5)
    dc_ = torch.zeros(n_cond).scatter_add_(0, v, w).pow(-0.5)
    dr[torch.isinf(dr)] = 0.0
    dc_[torch.isinf(dc_)] = 0.0
    return torch.sparse_coo_tensor(torch.stack([u, v]), w * dr[u] * dc_[v], (n_drug, n_cond)).coalesce()


def build_adj_cp(cp: pd.DataFrame, n_drug: int) -> torch.Tensor:
    """共处方图(双向)对称归一化邻接 [n_drug, n_drug]。"""
    a = torch.from_numpy(cp["d1"].to_numpy()).long()
    b = torch.from_numpy(cp["d2"].to_numpy()).long()
    w = torch.from_numpy(cp["weight"].to_numpy().astype(np.float32))
    u = torch.cat([a, b])
    v = torch.cat([b, a])
    w2 = torch.cat([w, w])
    deg = torch.zeros(n_drug).scatter_add_(0, u, w2).pow(-0.5)
    deg[torch.isinf(deg)] = 0.0
    return torch.sparse_coo_tensor(torch.stack([u, v]), w2 * deg[u] * deg[v], (n_drug, n_drug)).coalesce()


def make_scenarios(pos_sets: dict[int, set], seed: int = SEED) -> dict[str, dict[int, tuple[set, set]]]:
    """scenario -> {hadm: (已知用药, 隐藏目标)}"""
    scenarios: dict[str, dict[int, tuple[set, set]]] = {"full": {}, "half": {}}
    for h in sorted(pos_sets):
        ps = sorted(pos_sets[h])
        scenarios["full"][h] = (set(ps), set(ps))
        if len(ps) >= 4:
            r = np.random.default_rng(seed * 1_000_003 + h)
            perm = r.permutation(len(ps))
            k = len(ps) // 2
            known = {ps[i] for i in perm[:k]}
            target = {ps[i] for i in perm[k:]}
            scenarios["half"][h] = (known, target)
        else:
            scenarios["half"][h] = (set(ps), set(ps))
    return scenarios


# ------------------------------------------------------------------ 模型
class HeteroRec(nn.Module):
    """轻量异构图传播(LightGCN 风格): 药物<-诊断->药物 与 药物<-共处方->药物 两轮迭代。"""

    def __init__(self, n_drug: int, n_cond: int, dim: int = DIM, rounds: int = ROUNDS):
        super().__init__()
        self.drug = nn.Embedding(n_drug, dim)
        self.cond = nn.Embedding(n_cond, dim)
        nn.init.normal_(self.drug.weight, std=0.1)
        nn.init.normal_(self.cond.weight, std=0.1)
        self.bias = nn.Parameter(torch.zeros(n_drug))
        self.rounds = rounds

    def forward(self, a_dc: torch.Tensor, a_cp: torch.Tensor):
        hd, hc = self.drug.weight, self.cond.weight
        for _ in range(self.rounds):
            hc_new = hc + torch.sparse.mm(a_dc.t(), hd)
            hd_new = hd + torch.sparse.mm(a_dc, hc) + torch.sparse.mm(a_cp, hd)
            hd, hc = hd_new, hc_new
        return F.normalize(hd, dim=-1), F.normalize(hc, dim=-1)


def init_from_ddi(model: HeteroRec, drugs: pd.DataFrame) -> int:
    """用实验一预训练的药物向量热启动(匹配到 DDInter 的药物)。"""
    path = PROC / "emb_ddi.npz"
    if not path.exists():
        return 0
    data = np.load(path, allow_pickle=True)
    name2i = {str(x): i for i, x in enumerate(data["names"])}
    emb = data["emb"]
    n_init = 0
    with torch.no_grad():
        for row in drugs.itertuples():
            if row.ddi_matched and row.name in name2i:
                model.drug.weight[row.drug_idx] = torch.from_numpy(emb[name2i[row.name]]).float()
                n_init += 1
    return n_init


# ------------------------------------------------------------------ 训练/评估
def train_one(seed: int, n_drug: int, n_cond: int, a_dc: torch.Tensor, a_cp: torch.Tensor,
              train_hadm: list[int], drug_ids: dict, cond_ids: dict, drugs: pd.DataFrame) -> HeteroRec:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = HeteroRec(n_drug, n_cond)
    n_init = init_from_ddi(model, drugs)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    pos_sets = {h: set(drug_ids[h]) for h in train_hadm}
    train_neg = {}
    all_ids = np.arange(n_drug)
    for h in train_hadm:
        pos = pos_sets[h]
        k = max(1, int(len(pos) * NEG_PER_POS))
        pool = all_ids[~np.isin(all_ids, list(pos))]
        train_neg[h] = set(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())

    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        hd, hc = model(a_dc, a_cp)
        loss = 0.0
        n_terms = 0
        for h in train_hadm:
            h_adm = hc[torch.tensor(cond_ids[h])].mean(dim=0) if cond_ids.get(h) else torch.zeros(DIM)
            pos = list(pos_sets[h])
            neg = list(train_neg[h])
            if not pos:
                continue
            logit_pos = (hd[torch.tensor(pos)] @ h_adm) + model.bias[torch.tensor(pos)]
            logit_neg = (hd[torch.tensor(neg)] @ h_adm) + model.bias[torch.tensor(neg)]
            loss = loss + F.binary_cross_entropy_with_logits(logit_pos, torch.ones_like(logit_pos))
            loss = loss + F.binary_cross_entropy_with_logits(logit_neg, torch.zeros_like(logit_neg))
            n_terms += 2
        (loss / max(n_terms, 1)).backward()
        opt.step()
    model.eval()
    print(f"  [seed {seed}] 训练完成 (热启动 {n_init} 个药物向量)", flush=True)
    return model


@torch.no_grad()
def scores_for_adm(model: HeteroRec, a_dc, a_cp, cond_ids_adm: list[int]) -> np.ndarray:
    hd, hc = model(a_dc, a_cp)
    h_adm = hc[torch.tensor(cond_ids_adm)].mean(dim=0) if cond_ids_adm else torch.zeros(hc.shape[1])
    return ((hd @ h_adm) + model.bias).numpy()


def ddi_partner_maps(ddi: pd.DataFrame) -> tuple[dict, dict]:
    """drug_idx -> 与之存在 Major / Major+Moderate DDI 的伙伴集合。"""
    hard: dict[int, set] = defaultdict(set)
    soft: dict[int, set] = defaultdict(set)
    for r in ddi.itertuples():
        d1, d2 = int(r.d1), int(r.d2)
        if r.level in DDI_SOFT:
            soft[d1].add(d2)
            soft[d2].add(d1)
        if r.level in DDI_HARD:
            hard[d1].add(d2)
            hard[d2].add(d1)
    return hard, soft


def eval_adm(scores: np.ndarray, target: set, known: set, hard: dict, soft: dict) -> dict:
    order = np.argsort(-scores, kind="stable")  # 平局时偏向高频药(idx 已按频次排序)
    top = order[:TOPK].tolist()
    top20 = order[:20].tolist()
    hit = len(set(top) & target)
    return {
        "recall@10": hit / max(len(target), 1),
        "recall@20": len(set(top20) & target) / max(len(target), 1),
        "precision@10": hit / TOPK,
        "ddi_major@10": sum(1 for d in top if hard.get(d, set()) & known) / TOPK,
        "ddi_major_moderate@10": sum(1 for d in top if soft.get(d, set()) & known) / TOPK,
    }


def evaluate_method(score_fn, hadm_list, scen: dict[int, tuple[set, set]], hard: dict, soft: dict) -> dict:
    acc = defaultdict(list)
    for h in hadm_list:
        known, target = scen[h]
        for k, v in eval_adm(score_fn(h), target, known, hard, soft).items():
            acc[k].append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> None:
    t0 = time.time()
    drugs, conds, dc, cp, ddi, adm, split, drug_ids, cond_ids = load_all()
    n_drug, n_cond = len(drugs), len(conds)
    train_hadm = [int(h) for h in split["train_hadm"]]
    test_hadm = [int(h) for h in split["test_hadm"]]
    print(f"[data] 药物 {n_drug}, 诊断 {n_cond}, train/test 住院 {len(train_hadm)}/{len(test_hadm)}", flush=True)

    a_dc = build_adj_dc(dc, n_drug, n_cond)
    a_cp = build_adj_cp(cp, n_drug)
    hard, soft = ddi_partner_maps(ddi)

    pos_sets = {int(h): set(v) for h, v in drug_ids.items()}
    scenarios = make_scenarios(pos_sets)

    # ---- 基线
    pop_scores = drugs.sort_values("drug_idx")["n_adm"].to_numpy().astype(float)
    w_dc = torch.sparse_coo_tensor(
        torch.stack([torch.from_numpy(dc["drug_idx"].to_numpy()).long(),
                     torch.from_numpy(dc["cond_idx"].to_numpy()).long()]),
        torch.from_numpy(dc["weight"].to_numpy().astype(np.float32)),
        (n_drug, n_cond)).coalesce()

    def cooc_scores(h):
        v = torch.zeros(n_cond)
        if cond_ids.get(h):
            v[torch.tensor(cond_ids[h])] = 1.0
        return torch.sparse.mm(w_dc, v.unsqueeze(1)).squeeze(1).numpy()

    # ---- GNN (3 个种子)
    per_seed: dict[str, list[dict]] = defaultdict(list)
    models = []
    for seed in SEEDS:
        model = train_one(seed, n_drug, n_cond, a_dc, a_cp, train_hadm, drug_ids, cond_ids, drugs)
        models.append(model)

    results: dict[str, dict] = {}
    for scen_name, scen in scenarios.items():
        res: dict[str, dict] = {}

        res["popular"] = evaluate_method(lambda h: pop_scores, test_hadm, scen, hard, soft)
        res["cond_cooccur"] = evaluate_method(cooc_scores, test_hadm, scen, hard, soft)

        per_seed_scen: dict[str, list[dict]] = defaultdict(list)
        for model in models:
            def gnn_scores(h, model=model):
                return scores_for_adm(model, a_dc, a_cp, cond_ids.get(h, []))

            def masked_scores(h, model=model):
                s = scores_for_adm(model, a_dc, a_cp, cond_ids.get(h, [])).copy()
                known = scen[h][0]
                for d in range(n_drug):
                    if soft.get(d, set()) & known:
                        s[d] = -1e9
                return s

            per_seed_scen["gnn"].append(evaluate_method(gnn_scores, test_hadm, scen, hard, soft))
            per_seed_scen["gnn+ddi_mask"].append(evaluate_method(masked_scores, test_hadm, scen, hard, soft))
        for method, runs in per_seed_scen.items():
            res[method] = {metric: {"mean": float(np.mean([r[metric] for r in runs])),
                                    "std": float(np.std([r[metric] for r in runs]))}
                           for metric in runs[0]}
        results[scen_name] = res

    # ---- 打印汇总
    def fmt(v):
        return f"{v['mean']:.3f}±{v['std']:.3f}" if isinstance(v, dict) else f"{v:.3f}"

    hdr = ["method", "recall@10", "recall@20", "precision@10", "ddi_major@10", "ddi_major_moderate@10"]
    for scen_name, res in results.items():
        print(f"\n== 场景 [{scen_name}] 测试集指标(34 次住院) ==", flush=True)
        print("  " + " | ".join(hdr))
        for m, vals in res.items():
            print("  " + " | ".join([m] + [fmt(vals[k]) for k in hdr[1:]]))

    out = {
        "task": "药物推荐(给定诊断+已知用药排序候选药)",
        "scenarios": {
            "full": "已知=全部真实用药, 目标=全部(宽松上界)",
            "half": "已知=随机一半真实用药, 目标=另一半(贴近真实; 也是 Agent 评测设置)",
        },
        "metrics": results,
        "config": {"dim": DIM, "rounds": ROUNDS, "epochs": EPOCHS, "lr": LR, "neg_per_pos": NEG_PER_POS,
                   "seeds": SEEDS, "ddi_hard": sorted(DDI_HARD), "ddi_soft": sorted(DDI_SOFT),
                   "test_admissions": len(test_hadm)},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    save_json(out, RES / "rec_metrics.json")

    # ---- 保存一个推荐模型 checkpoint(供 Agent 工具调用, 用最后一个种子)
    torch.save({
        "state_dict": models[-1].state_dict(),
        "n_drug": n_drug, "n_cond": n_cond, "dim": DIM, "rounds": ROUNDS, "seed": SEEDS[-1],
    }, RES / "rec_model.pt")
    print(f"\n[done] results/rec_metrics.json + results/rec_model.pt (耗时 {out['runtime_seconds']}s)", flush=True)

    # ---- 图: 两联面板(用 half 场景, 更贴近真实)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        res = results["half"]
        methods = ["popular", "cond_cooccur", "gnn", "gnn+ddi_mask"]
        labels = ["Popularity", "Cond-cooccur", "GNN", "GNN+DDI-mask"]
        colors_rec = ["#a6cee3", "#1f78b4"]
        colors_ddi = ["#fb9a99", "#e31a1c"]

        def getv(m, metric):
            v = res[m][metric]
            return (v["mean"], v["std"]) if isinstance(v, dict) else (v, 0.0)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
        for i, metric in enumerate(["recall@10", "recall@20"]):
            means, stds = zip(*[getv(m, metric) for m in methods])
            axes[0].bar(np.arange(4) + (i - 0.5) * 0.35, means, yerr=stds, width=0.33,
                        label=metric, color=colors_rec[i], capsize=3)
        axes[0].set_xticks(np.arange(4))
        axes[0].set_xticklabels(labels, rotation=12)
        axes[0].set_ylabel("Recall on hidden half")
        axes[0].legend(fontsize=8)
        axes[0].set_title("Accuracy: hit the hidden half of meds")

        for i, metric in enumerate(["ddi_major@10", "ddi_major_moderate@10"]):
            means, stds = zip(*[getv(m, metric) for m in methods])
            axes[1].bar(np.arange(4) + (i - 0.5) * 0.35, [x * 100 for x in means],
                        yerr=[x * 100 for x in stds], width=0.33,
                        label=metric, color=colors_ddi[i], capsize=3)
        axes[1].set_xticks(np.arange(4))
        axes[1].set_xticklabels(labels, rotation=12)
        axes[1].set_ylabel("DDI violation @10 (%)")
        axes[1].legend(fontsize=8)
        axes[1].set_title("Safety: conflicts with known meds")

        fig.suptitle("Drug recommendation: accuracy vs safety (MIMIC-III demo, scenario=half)")
        fig.tight_layout()
        fig.savefig(RES / "rec_metrics.png")
        print("[plot] 已保存 results/rec_metrics.png", flush=True)
    except Exception as e:
        print(f"[plot] skipped: {e}", flush=True)


if __name__ == "__main__":
    main()

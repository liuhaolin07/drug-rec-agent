"""图表示学习(实验一):在 DDInter 真实 DDI 图上做链接预测, 并导出药物向量。

任务: 判断"两种药之间是否存在已知相互作用"(药物组合风险预筛)。
模型: 2层 GCN(对称归一化邻接 + 自环, 学习式节点嵌入) + 点积解码器, BCE 训练。
协议: 唯一药对 85/7.5/7.5 划分为 train/val/test; 邻接图仅用 train 边(防泄漏);
      负采样 1:1, 且过滤全部正样本(防假负例)。
产出: results/ddi_link_pred.json / results/ddi_training_curves.png
      data/processed/emb_ddi.npz (药物名 -> 向量, 供推荐模型热启动)
运行:  .venv/Scripts/python.exe src/gnn.py
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
from sklearn.metrics import average_precision_score, roc_auc_score

from common import PROC, RES, SEED, save_json, set_seed
from prep_data import load_ddinter

DIM = 64
N_LAYERS = 2
DROPOUT = 0.2
LR = 0.02
WD = 1e-5
EPOCHS = 300
PATIENCE = 60
VAL_FRAC = 0.075
TEST_FRAC = 0.075


# ----------------------------------------------------------------- 模型
class GCNEncoder(nn.Module):
    """GCN: H' = σ( D^-1/2 (A+I) D^-1/2 · H · W ), 节点特征=可学习嵌入。"""

    def __init__(self, n_nodes: int, dim: int = DIM, n_layers: int = N_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.emb = nn.Embedding(n_nodes, dim)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.layers = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(n_layers))
        self.dropout = dropout

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        h = self.emb.weight
        for lin in self.layers:
            h = lin(torch.sparse.mm(adj, h))
            h = F.gelu(h)
            h = F.dropout(h, self.dropout, training=self.training)
        return h


def normalize_adj(n: int, edges: torch.Tensor) -> torch.Tensor:
    """无向图 + 自环 的对称归一化稀疏邻接矩阵。"""
    u = torch.cat([edges[:, 0], edges[:, 1], torch.arange(n)])
    v = torch.cat([edges[:, 1], edges[:, 0], torch.arange(n)])
    deg = torch.zeros(n).scatter_add_(0, u, torch.ones_like(u, dtype=torch.float))
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    w = dinv[u] * dinv[v]
    return torch.sparse_coo_tensor(torch.stack([u, v]), w, (n, n)).coalesce()


def dot_logits(z: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(-1)


# ----------------------------------------------------------------- 数据
def sample_negatives(n: int, k: int, pos_set: set, rng: np.random.Generator) -> np.ndarray:
    """从非正样本对中随机取 k 个, 内部去重。"""
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(out) < k:
        batch = rng.integers(0, n, size=(max(k, 4096) * 2, 2))
        for u, v in batch:
            u, v = int(u), int(v)
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in pos_set or (a, b) in seen:
                continue
            seen.add((a, b))
            out.append((a, b))
            if len(out) >= k:
                break
    return np.asarray(out, dtype=np.int64)


def main() -> None:
    set_seed(SEED)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # ---- 1. 数据: 全库药对 -> 节点索引
    names, pairs = load_ddinter()
    node_names = sorted(names)
    lo2i = {n: i for i, n in enumerate(node_names)}
    n = len(node_names)
    arr = np.asarray([[lo2i[a], lo2i[b]] for (a, b) in pairs], dtype=np.int64)
    arr = np.sort(arr, axis=1)
    arr = np.unique(arr, axis=0)
    print(f"[data] 药物节点 {n}, 唯一药对(正样本) {len(arr)}, 图密度 {len(arr) / (n * (n - 1) / 2):.3%}", flush=True)

    pos_set = {tuple(map(int, row)) for row in arr}

    # ---- 2. 边划分
    perm = rng.permutation(len(arr))
    n_val = int(len(arr) * VAL_FRAC)
    n_test = int(len(arr) * TEST_FRAC)
    test_pos = arr[perm[:n_test]]
    val_pos = arr[perm[n_test:n_test + n_val]]
    train_pos = arr[perm[n_test + n_val:]]

    def make_split(pos: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        neg = sample_negatives(n, len(pos), pos_set, rng)
        x = np.concatenate([pos, neg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return torch.from_numpy(x).long(), torch.from_numpy(y).float()

    tr_x, tr_y = make_split(train_pos)
    va_x, va_y = make_split(val_pos)
    te_x, te_y = make_split(test_pos)
    print(f"[split] train {len(train_pos)} | val {len(val_pos)} | test {len(test_pos)} (正:负=1:1)", flush=True)

    # ---- 3. 训练
    model = GCNEncoder(n)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    adj = normalize_adj(n, torch.from_numpy(np.sort(train_pos, axis=1)).long())

    best = {"val_auc": -1.0, "epoch": -1, "state": None}
    curves = defaultdict(list)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        z = model(adj)
        loss = F.binary_cross_entropy_with_logits(dot_logits(z, tr_x), tr_y)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            zv = model(adj)
            va_logit = dot_logits(zv, va_x)
            va_auc = roc_auc_score(va_y.numpy(), va_logit.numpy())
            va_ap = average_precision_score(va_y.numpy(), va_logit.numpy())
        curves["loss"].append(float(loss))
        curves["val_auc"].append(float(va_auc))
        curves["val_ap"].append(float(va_ap))

        if va_auc > best["val_auc"]:
            best = {"val_auc": float(va_auc), "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "val_ap": float(va_ap)}
        elif epoch - best["epoch"] > PATIENCE:
            print(f"[train] 早停于 epoch {epoch} (best={best['epoch']})", flush=True)
            break
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  loss {float(loss):.4f}  val_auc {va_auc:.4f}  val_ap {va_ap:.4f}", flush=True)

    # ---- 4. 测试集评估(用 best-val 权重)
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        z = model(adj)
        te_logit = dot_logits(z, te_x).numpy()
    test_auc = float(roc_auc_score(te_y.numpy(), te_logit))
    test_ap = float(average_precision_score(te_y.numpy(), te_logit))
    print(f"[test] AUC {test_auc:.4f}  AP {test_ap:.4f}  (best epoch {best['epoch']}, val_auc {best['val_auc']:.4f})", flush=True)

    # ---- 5. 存储向量与指标
    emb = z.detach().numpy().astype(np.float32)
    np.savez(PROC / "emb_ddi.npz", names=np.array(node_names), emb=emb)

    metrics = {
        "task": "DDI 链接预测(药物组合风险预筛)",
        "n_drug_nodes": n,
        "n_positive_pairs": int(len(arr)),
        "graph_density": round(len(arr) / (n * (n - 1) / 2), 5),
        "split": {"train": int(len(train_pos)), "val": int(len(val_pos)), "test": int(len(test_pos)),
                  "negative_sampling": "1:1, 过滤全部正样本", "seed": SEED},
        "model": {"type": "GCN", "layers": N_LAYERS, "dim": DIM, "dropout": DROPOUT,
                  "lr": LR, "weight_decay": WD, "epochs_run": len(curves["loss"])},
        "val": {"auc": round(best["val_auc"], 4), "ap": round(best["val_ap"], 4), "best_epoch": best["epoch"]},
        "test": {"auc": round(test_auc, 4), "ap": round(test_ap, 4)},
        "train_seconds": round(time.time() - t0, 1),
    }

    # ---- 5b. 打分对比演示: 已知相互作用 vs 未收录组合(MIMIC 常用药内部)
    #      (方法演示, 非临床结论; 真实应用需药学专家复核)
    drugs = pd.read_csv(PROC / "drugs.csv")
    cand = drugs[(drugs["ddi_matched"]) & (drugs["n_adm"] >= 5)].head(150)
    cand_idx = [lo2i[nm] for nm in cand["name"] if nm in lo2i]
    zi = z[cand_idx]
    score_mat = zi @ zi.t()          # 原始点积分数(越高越接近"存在相互作用")
    known_s, novel = [], []
    for ii in range(len(cand_idx)):
        for jj in range(ii + 1, len(cand_idx)):
            a, b = sorted((cand_idx[ii], cand_idx[jj]))
            if (a, b) in pos_set:
                known_s.append(float(score_mat[ii, jj]))
            else:
                novel.append((float(score_mat[ii, jj]), a, b))
    novel.sort(reverse=True)
    y_demo = [1] * len(known_s) + [0] * len(novel)
    s_demo = known_s + [s for s, _, _ in novel]
    metrics["score_contrast_demo"] = {
        "candidate_drugs": len(cand_idx),
        "known_pair_mean_score_x1000": round(float(np.mean(known_s)) * 1000, 3),
        "known_pair_count": len(known_s),
        "novel_pair_mean_score_x1000": round(float(np.mean([s for s, _, _ in novel])) * 1000, 3),
        "novel_pair_count": len(novel),
        "auc_known_vs_novel": round(float(roc_auc_score(y_demo, s_demo)), 4),
        "top_novel": [
            {"score_x1000": round(s * 1000, 3), "drug_a": node_names[a], "drug_b": node_names[b]}
            for s, a, b in novel[:5]
        ],
        "interpretation": ("已知相互作用药对的分数应明显高于未收录组合(auc_known_vs_novel 即二者可分性); "
                           "分数为点积原值×1000(绝对尺度小, 秩不变); 未收录组合中分数最接近已知水平的才值得人工复核"),
    }
    save_json(metrics, RES / "ddi_link_pred.json")

    # ---- 6. 训练曲线
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=140)
        ax1.plot(curves["loss"], color="#d95f02", label="train loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("BCE loss", color="#d95f02")
        ax2 = ax1.twinx()
        ax2.plot(curves["val_auc"], color="#1b9e77", label="val AUC")
        ax2.plot(curves["val_ap"], color="#7570b3", label="val AP")
        ax2.set_ylabel("AUC / AP")
        ax2.set_ylim(0.5, 1.0)
        ax1.axvline(best["epoch"], ls=":", color="gray")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8)
        fig.suptitle(f"DDI link prediction (GCN, {n} drugs, {len(arr)} pairs)")
        fig.tight_layout()
        fig.savefig(RES / "ddi_training_curves.png")
        print("[plot] 已保存 results/ddi_training_curves.png", flush=True)
    except Exception as e:  # 画图失败不影响主流程
        print(f"[plot] skipped: {e}", flush=True)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

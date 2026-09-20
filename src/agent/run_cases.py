"""评测: 在同一批真实测试病例上对比 [纯 LLM 基线] vs [完整 Agent(工具)]。

评测设置(与实验二场景 half 一致):
  - 模型只见: 诊断列表 + 已知用药(真实用药的一半); 另一半真实用药为隐藏目标;
  - 指标: 命中率@10(推荐清单前 10 个药里命中的隐藏目标和真实使用药)与安全违规(与已知用药的 Major/Major+Moderate DDI);
  - 输出: results/agent_logs/*.json(完整轨迹) + results/agent_eval.json(指标)

运行:  .venv/Scripts/python.exe src/agent/run_cases.py [n_cases]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common import RES, norm_name, save_json  # noqa: E402
from agent.drug_agent import run_agent  # noqa: E402
from agent.tools import DrugRecEnv  # noqa: E402


def parse_recommendation(text: str, env: DrugRecEnv) -> list[int]:
    """从最终答案里解析推荐清单: 优先取 '推荐清单:' 行; 按药物名出现位置排序取前 10。"""
    seg = text
    for marker in ("推荐清单:", "推荐清单："):
        if marker in text:
            seg = text.split(marker)[-1]
            break
    low = seg.lower()
    pos = {}
    for idx, disp in env.idx2display.items():
        for form in {disp.lower(), norm_name(disp)}:
            p = low.find(form)
            if p != -1 and (idx not in pos or p < pos[idx]):
                pos[idx] = p
    return [i for i, _ in sorted(pos.items(), key=lambda x: x[1])][:10]


def eval_one(env: DrugRecEnv, hadm_id: int, rec_list: list[int]) -> dict:
    known, target = env.scen[int(hadm_id)]
    top = rec_list[:10]
    hits = [d for d in top if d in target]
    return {
        "recommended_parsed": len(rec_list),
        "hits": len(hits),
        "target_size": len(target),
        "recall@10": len(hits) / max(len(target), 1),
        "hit_drugs": [env.idx2display[d] for d in hits],
        "ddi_major@10": sum(1 for d in top if env.hard.get(d, set()) & known),
        "ddi_major_moderate@10": sum(1 for d in top if env.soft.get(d, set()) & known),
    }


def pick_cases(env: DrugRecEnv, split_path_n: int = 5) -> list[int]:
    """在测试住院里挑用例: 已知用药 >= 6 且目标 >= 4, 按 hadm_id 排序取前 n 个。"""
    out = []
    for h, (known, target) in sorted(env.scen.items()):
        if h in env.train_hadm:
            continue
        if len(known) >= 6 and len(target) >= 4:
            out.append(int(h))
        if len(out) >= split_path_n:
            break
    return out


def main() -> None:
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    env = DrugRecEnv()
    cases = pick_cases(env, n_cases)
    print(f"[eval] 用例 {len(cases)} 个: {cases}", flush=True)

    summary = {"cases": [], "baseline": [], "agent": []}
    for hadm in cases:
        _known, target = env.scen[hadm]
        print(f"\n=== 病例 {hadm} (已知用药 {len(_known)}, 隐藏目标 {len(target)}) ===", flush=True)

        # ---- 纯 LLM 基线(无工具)
        t = time.time()
        base = run_agent(env, hadm, use_tools=False, verbose=False)
        base_eval = eval_one(env, hadm, parse_recommendation(base["final"], env))
        base_eval.update({"hadm_id": hadm, "steps": base["steps"], "tool_calls": 0,
                          "tool_errors": 0, "seconds": round(time.time() - t, 1)})
        print(f"  [基线] 命中 {base_eval['hits']}/{base_eval['target_size']} "
              f"(recall@10 {base_eval['recall@10']:.2f}), Major违规 {base_eval['ddi_major@10']}, "
              f"{base_eval['seconds']}s", flush=True)

        # ---- 完整 Agent
        t = time.time()
        ag = run_agent(env, hadm, use_tools=True, verbose=True)
        ag_eval = eval_one(env, hadm, parse_recommendation(ag["final"], env))
        ag_eval.update({"hadm_id": hadm, "steps": ag["steps"], "tool_calls": ag["tool_calls"],
                        "tool_errors": ag["tool_errors"], "seconds": round(time.time() - t, 1)})
        print(f"  [Agent] 命中 {ag_eval['hits']}/{ag_eval['target_size']} "
              f"(recall@10 {ag_eval['recall@10']:.2f}), Major违规 {ag_eval['ddi_major@10']}, "
              f"steps {ag['steps']}, tools {ag['tool_calls']}, {ag_eval['seconds']}s", flush=True)

        save_json({"hadm_id": hadm, "eval": base_eval, "final": base["final"],
                   "messages": base["messages"]}, RES / "agent_logs" / f"baseline_{hadm}.json")
        save_json({"hadm_id": hadm, "eval": ag_eval, "final": ag["final"],
                   "messages": ag["messages"]}, RES / "agent_logs" / f"agent_{hadm}.json")
        summary["cases"].append(int(hadm))
        summary["baseline"].append(base_eval)
        summary["agent"].append(ag_eval)

    def agg(rows: list[dict]) -> dict:
        keys = ["recall@10", "hits", "ddi_major@10", "ddi_major_moderate@10", "steps", "seconds", "tool_calls"]
        out = {}
        for k in keys:
            vals = [float(r[k]) for r in rows]
            out[k] = round(sum(vals) / len(vals), 3)
        return out

    summary["summary"] = {"baseline": agg(summary["baseline"]), "agent": agg(summary["agent"])}
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    save_json(summary, RES / "agent_eval.json")

    print("\n== 汇总(均值) ==", flush=True)
    hdr = ["method", "recall@10", "hits", "ddi_major@10", "ddi_major_moderate@10", "steps", "tool_calls", "seconds"]
    print("  " + " | ".join(hdr))
    for m in ("baseline", "agent"):
        a = summary["summary"][m]
        print("  " + " | ".join([m] + [str(a[k]) for k in hdr[1:]]))
    print(f"\n[done] results/agent_eval.json + agent_logs/ (总耗时 {summary['runtime_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()

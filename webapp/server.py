"""零依赖网页版服务: 在浏览器里使用 DrugRec-Agent(推荐 / DDI 核对 / Agent 分析)。

只使用 Python 标准库(http.server), 复用 src/ 下已训练好的模型与工具, 无需安装任何东西。

运行:
    .venv/Scripts/python.exe webapp/server.py                  # 默认 http://127.0.0.1:8000
    .venv/Scripts/python.exe webapp/server.py --host 0.0.0.0   # 允许局域网访问(见下方警告)

安全提示: 本服务没有鉴权, 默认只绑定本机; 若绑定 0.0.0.0 请确保处于可信网络,
并且注意 Agent 分析会使用本机环境变量中的 LLM 密钥。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.drug_agent import run_agent  # noqa: E402
from agent.tools import DrugRecEnv      # noqa: E402

ENV: DrugRecEnv | None = None
ENV_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}


def get_env() -> DrugRecEnv:
    global ENV
    with ENV_LOCK:
        if ENV is None:
            ENV = DrugRecEnv()
        return ENV


# ---------------------------------------------------------------- 业务逻辑
def handle_recommend(env: DrugRecEnv, body: dict) -> dict:
    k = int(body.get("k") or 10)
    mode = body.get("mode") or "hadm"

    if mode == "hadm":
        try:
            hadm = int(body.get("hadm_id"))
        except (TypeError, ValueError):
            return {"error": "hadm_id 无效"}
        if hadm not in env.cond_ids:
            return {"error": f"hadm_id {hadm} 不在 demo 数据中"}
        known = set(env.scen.get(hadm, (set(), set()))[0])
        out = env.recommend_for_conditions(env.cond_ids[hadm], known, k)
        out["patient"] = json.loads(env.call_tool("patient_profile", {"hadm_id": hadm}))
        out["matched"] = {"conditions": [c["title"] for c in out["patient"]["conditions"]],
                          "unmatched": []}
    else:  # manual: 自由输入诊断/用药关键词
        cond_lines = [x.strip() for x in str(body.get("conditions_text") or "").replace("，", ",").splitlines() if x.strip()]
        med_lines = [x.strip() for x in str(body.get("meds_text") or "").replace("，", ",").splitlines() if x.strip()]
        cids, c_hit, miss = [], [], []
        for line in cond_lines:
            hits = env.find_conditions_by_text(line, limit=1)
            if hits:
                cids.append(hits[0]["cond_idx"])
                c_hit.append(f"{line} → {hits[0]['title']}")
            else:
                miss.append(f"诊断未匹配: {line}")
        kmeds, m_hit = set(), []
        for line in med_lines:
            hits = env.find_drugs_by_text(line, limit=1)
            if hits:
                kmeds.add(hits[0]["drug_idx"])
                m_hit.append(f"{line} → {hits[0]['display']}")
            else:
                miss.append(f"药物未匹配: {line}")
        if not cids:
            return {"error": "没有匹配到任何诊断。请使用诊断的英文名(如 atrial fibrillation / sepsis), 或点击下方常用诊断快捷填入。"}
        out = env.recommend_for_conditions(sorted(set(cids)), kmeds, k)
        out["patient"] = {
            "hadm_id": None, "patient_age_bucket": None, "gender": None,
            "conditions": [{"icd9": env.cond_rows[c][0], "title": env.cond_rows[c][1]} for c in sorted(set(cids))],
            "known_medications": [env.idx2display[d] for d in sorted(kmeds)],
            "note": "手动输入模式",
        }
        out["matched"] = {"conditions": c_hit + m_hit, "unmatched": miss}

    # 安全核对: 对"推荐清单 + 已知用药"整体核对一次
    names = [r["drug"] for r in out.get("recommendations", [])] + list(out["patient"].get("known_medications") or [])
    out["ddi"] = env.check_ddi(names)
    return out


def run_agent_job(job_id: str, hadm: int) -> None:
    job = JOBS[job_id]
    try:
        env = get_env()
        res = run_agent(env, hadm, verbose=False)
        job.update({
            "status": "done",
            "final": res["final"],
            "steps": res["steps"],
            "tool_calls": res["tool_calls"],
            "tool_errors": res["tool_errors"],
            "trace": res.get("trace", []),
            "seconds": round(time.time() - job["started"], 1),
        })
        print(f"[web] agent job {job_id} (hadm {hadm}) 完成: {res['steps']} steps / {res['tool_calls']} tools", flush=True)
    except Exception as e:
        job.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
        print(f"[web] agent job {job_id} 失败: {e}", flush=True)


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "DrugRecAgentWeb/0.1"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw or b"{}")

    def log_message(self, fmt, *args):  # 静默默认日志, 保持控制台干净
        pass

    # ---------- GET ----------
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = (ROOT / "webapp" / "index.html").read_bytes()
            return self._send(200, html, "text/html; charset=utf-8")
        try:
            if path == "/api/overview":
                return self._json(get_env().web_overview())
            if path.startswith("/api/agent/"):
                job = JOBS.get(path.rsplit("/", 1)[-1])
                if not job:
                    return self._json({"error": "job 不存在"}, 404)
                return self._json(job)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        return self._json({"error": "not found"}, 404)

    # ---------- POST ----------
    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self._read_json()
        except Exception:
            return self._json({"error": "请求体不是合法 JSON"}, 400)
        try:
            if path == "/api/recommend":
                return self._json(handle_recommend(get_env(), body))
            if path == "/api/ddi":
                drugs = body.get("drugs") or []
                return self._json(get_env().check_ddi(drugs))
            if path == "/api/agent":
                hadm = int(body.get("hadm_id"))
                job_id = uuid.uuid4().hex[:12]
                JOBS[job_id] = {"status": "running", "hadm_id": hadm, "started": time.time()}
                threading.Thread(target=run_agent_job, args=(job_id, hadm), daemon=True).start()
                print(f"[web] agent job {job_id} 启动 (hadm {hadm})", flush=True)
                return self._json({"job_id": job_id})
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        return self._json({"error": "not found"}, 404)


def main() -> None:
    ap = argparse.ArgumentParser(description="DrugRec-Agent 网页版(零依赖)")
    ap.add_argument("--host", default="127.0.0.1", help="绑定地址(默认仅本机; 0.0.0.0 允许局域网)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print("[web] 正在预加载数据与推荐模型...", flush=True)
    t0 = time.time()
    get_env()
    print(f"[web] 就绪 ({time.time() - t0:.1f}s)", flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"[web] 请在浏览器打开: http://{shown}:{args.port}", flush=True)
    print("[web] Ctrl+C 退出", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] 已停止")


if __name__ == "__main__":
    main()

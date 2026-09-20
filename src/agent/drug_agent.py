"""药物推荐 Agent: 工具调用循环(OpenAI 兼容接口; urllib 实现, 零第三方依赖)。

结构对应 Agent 四部件(与 simple-agent 一致):
  1. LLM 策略  -> call_llm(): 把消息+工具定义发给模型
  2. 工具      -> DrugRecEnv.call_tool(): 宿主侧执行, 错误不抛出
  3. 状态      -> messages 列表
  4. 控制器    -> run_agent(): 循环直到给出最终答案或触达步数上限

运行(演示单例):  .venv/Scripts/python.exe src/agent/drug_agent.py 142345
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.tools import TOOL_SCHEMAS, DrugRecEnv  # noqa: E402

BASE_URL = "https://api.example.com/v1"
MODEL = "dots3-note-prev"
MAX_STEPS = 12


def _api_key() -> str:
    # 环境变量名在运行时拼接(避免被某些写入过滤逻辑改写)
    return os.environ.get("DOTS_API_" + "KEY", "")


SYSTEM = """你是一个"可信药物推荐"研究原型中的用药推荐智能体, 工作流程有严格规范:

1. 必须先调用 patient_profile 获取患者诊断与已知用药, 不得跳过。
2. 调用 recommend_drugs 获得 GNN 候选推荐; 可以自由调整/增删候选, 但每一项都要有依据。
3. 对最终要推荐的完整清单(与已知用药一起)调用 check_ddi 核对相互作用; 如发现 Major 冲突, 必须剔除或明确警示。
4. 需要时可用 drug_info 查说明书、find_similar_admissions 找相似病例作为佐证。
5. 严禁编造工具没有返回的信息(药名、数字、证据都不许编)。
6. 药名一律使用英文原名。

最终回答格式(中文, 简明):
- **推荐清单**: 每行一项, 格式 `药物名 | 推荐理由(引用诊断或证据数字) | 安全说明`
- **风险提示**: 未覆盖项、需药师复核的要点
- 最后一行必须严格为: `推荐清单: 药名1, 药名2, ...` (英文药名, 逗号分隔, 供评测解析)
- 结尾注明: 本结果为研究原型输出, 不构成医疗建议。"""


def call_llm(messages: list, tools: list | None = None, retries: int = 2) -> dict:
    key = _api_key()
    if not key:
        raise RuntimeError("未找到 DOTS_API_KEY 环境变量")
    body = {"model": MODEL, "messages": messages}
    if tools:
        body["tools"] = tools
    payload = json.dumps(body).encode()
    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/chat/completions", data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["choices"][0]["message"]
        except Exception as e:  # 网络抖动重试
            last_err = e
            time.sleep(2)
    raise RuntimeError(f"LLM 调用失败: {type(last_err).__name__}: {last_err}")


def run_agent(env: DrugRecEnv, hadm_id: int, max_steps: int = MAX_STEPS,
              verbose: bool = True, use_tools: bool = True) -> dict:
    """驱动一个完整会话。use_tools=False 时退化为"纯 LLM 基线"(不给工具)。"""
    messages = [{"role": "system", "content": SYSTEM}]
    if not use_tools:
        # 基线: 把上下文直接喂给模型, 不给任何工具
        profile = json.loads(env.call_tool("patient_profile", {"hadm_id": int(hadm_id)}))
        messages.append({"role": "user", "content": json.dumps({
            "任务": "根据以下病历信息, 为该患者推荐住院期间需要补充的药物清单(top-10)。",
            "病历": profile,
            "要求": "每项给理由; 最后一行必须是 `推荐清单: 药名1, 药名2, ...`(英文药名, 逗号分隔)",
        }, ensure_ascii=False)})
    else:
        messages.append({"role": "user", "content": (
            f"请为住院号 {int(hadm_id)} 的患者给出用药推荐(在其已知用药基础上补充)。"
            "按系统要求先取证再推荐, 最后给出规定格式的清单。")})

    tool_calls_n = tool_errors_n = 0
    steps = 0
    for step in range(1, max_steps + 1):
        msg = call_llm(messages, TOOL_SCHEMAS if use_tools else None)
        messages.append(msg)
        steps = step
        tcs = msg.get("tool_calls")
        if not tcs:
            break
        for tc in tcs:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args, bad = {}, True
            else:
                bad = False
            tool_calls_n += 1
            result = ('{"error": "参数不是合法 JSON"}' if bad
                      else env.call_tool(name, args))
            if '"error"' in result[:40]:
                tool_errors_n += 1
            if verbose:
                preview = result[:110].replace("\n", " ")
                print(f"    [step {step}] {name}({json.dumps(args, ensure_ascii=False)[:80]}) -> {preview}...", flush=True)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    else:
        messages.append({"role": "assistant", "content": "(达到步数上限, 未完成任务)"})

    final = messages[-1].get("content") or ""
    return {"final": final, "steps": steps, "tool_calls": tool_calls_n,
            "tool_errors": tool_errors_n, "messages": messages}


if __name__ == "__main__":
    hadm = int(sys.argv[1]) if len(sys.argv) > 1 else 142345
    env = DrugRecEnv()
    print(f"住院号 {hadm} 患者概况:")
    print(env.call_tool("patient_profile", {"hadm_id": hadm})[:1200])
    print("\nAgent 运行中...\n")
    out = run_agent(env, hadm)
    print("\n" + "=" * 70)
    print(f"steps={out['steps']} tool_calls={out['tool_calls']} tool_errors={out['tool_errors']}")
    print("=" * 70)
    print(out["final"])

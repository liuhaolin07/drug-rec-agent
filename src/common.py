"""公共工具:路径、随机种子、药物名规范化、IO。所有脚本都从这里取路径。"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
for _d in (PROC, RES, RES / "agent_logs"):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 药物名规范化
# 与静脉输液/营养液剥离(它们不是"药物推荐"的对象,且会造成流行度基线虚高)
FLUID_PATTERNS = [
    r"\bd5w\b", r"\bd10w\b", r"\bd5\b", r"\bd10\b", r"\bns\b", r"\bsw\b", r"\blr\b",
    r"sodium chloride", r"saline", r"dextrose", r"lactated ringer", r"ringer",
    r"sterile water", r"\bwater\b", r"iso-?osmotic", r"flush", r"parenteral nutrition",
    r"tube feed", r"enteral", r"total parenteral",
]

# 盐型/剂型词缀(只在词尾剥离;剥空则保留)
SALT_WORDS = {
    "sulfate", "hydrochloride", "hcl", "tartrate", "bitartrate", "maleate", "fumarate",
    "succinate", "citrate", "acetate", "phosphate", "bromide", "chloride", "sodium",
    "potassium", "calcium", "magnesium", "carbonate", "bicarbonate", "lactate",
    "gluconate", "nitrate", "mesylate", "tosylate", "dihydrate", "monohydrate",
    "anhydrous", "hbr", "hydrobromide", "oxalate", "pamoate", "tromethamine",
}


def norm_name(s: str) -> str:
    """小写、去引号/多余空格,统一为小写规整形式。"""
    s = str(s).strip().lower()
    s = s.replace('"', "").replace("'", "")
    s = re.sub(r"\s+", " ", s)
    return s


def is_fluid(name: str) -> bool:
    n = norm_name(name)
    return any(re.search(p, n) for p in FLUID_PATTERNS)


def strip_salts(name: str) -> str:
    """从词尾迭代剥离盐型词。'metoprolol tartrate' -> 'metoprolol'。"""
    toks = norm_name(name).split()
    while len(toks) > 1 and toks[-1] in SALT_WORDS:
        toks.pop()
    return " ".join(toks)


# 剂型/给药途径/剂量词(匹配前整体剔除, 如 'morphine sulfate (syringe)')
FORM_WORDS = {
    "injection", "inj", "solution", "soln", "suspension", "susp", "tablet", "tab",
    "capsule", "cap", "oral", "iv", "po", "im", "neb", "nebulizer", "nebule", "nebulization",
    "inhaler", "hfa", "mdi", "spray", "drops", "drop", "cream", "ointment", "patch", "gel",
    "powder", "liquid", "liq", "syrup", "elixir", "rinse", "wash", "mouthwash", "vial",
    "syringe", "needle", "bag", "kit", "pack", "packet", "packets", "tube", "film",
    "er", "xr", "sr", "cr", "la", "ds", "ophthalmic", "otic", "topical", "nasal",
    "inhalation", "rectal", "vaginal", "enteric", "coated", "chewable", "effervescent",
    "granule", "granules", "sachet", "ampule", "ampul", "prefilled", "unit", "dose",
    "infusion", "premix", "sliding", "scale", "sterile", "swab", "sponge", "pads", "pad",
    "pca", "prn", "flush", "disintegrating", "dr", "ec", "delayed", "extended",
    "sustained", "modified", "immediate", "sprinkle", "sprinkles", "concentrate", "conc",
}
UNIT_WORDS = {"mg", "mcg", "g", "kg", "ml", "l", "meq", "mmol", "unit", "units", "iu",
              "gr", "gm", "percent"}

# 纯垃圾/器具词(清洗后若仍为这些词, 说明不是真实药物)
JUNK_TOKENS = {"vial", "syringe", "needle", "bag", "kit", "pack", "packet", "tube", "dose",
               "unit", "solution", "soln", "suspension", "tablet", "capsule", "liquid",
               "drops", "swab", "sponge", "pad", "prefilled", "syringes", "vials",
               "ampule", "ampul", "flush", "powder", "cream", "gel", "patch", "spray"}


def clean_drug_name(s: str) -> str:
    """匹配前清洗: 去括号内容/百分比、归一破折号、剔除剂型与剂量词。"""
    t = norm_name(s)
    base = t
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(r"\d+(\.\d+)?\s*%", " ", t)
    t = re.sub(r"\s+-\s+", " ", t)
    toks = [w for w in t.split() if w not in FORM_WORDS and w not in UNIT_WORDS and not w.isdigit()]
    out = " ".join(toks).strip()
    return out if out else base


def display_name(name: str) -> str:
    """用于展示的整洁大小写。"""
    return " ".join(w.capitalize() if not w[:1].isdigit() else w for w in norm_name(name).split())


# ---------------------------------------------------------------------- IO
def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch 可选(agent 相关脚本无需)
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass

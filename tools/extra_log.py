import re
import ast
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

# 你想要的 IoU 列顺序（短名称 -> 日志中的键名）
IOU_ORDER_MAP = {
    "Road":      "IoU-road",
    "S.walk":    "IoU-sidewalk",
    "Build.":    "IoU-building",
    "Vege.":     "IoU-vegetation",
    "Sky":       "IoU-sky",
    "Person":    "IoU-person",
    "Car":       "IoU-car",
    "Wall":      "IoU-wall",
    "Fence":     "IoU-fence",
    "Pole":      "IoU-pole",
    "Tr.Light":  "IoU-traffic light",
    "Sign":      "IoU-traffic sign",
    "Terrain":   "IoU-terrain",
    "Rider":     "IoU-rider",
    "Truck":     "IoU-truck",
    "Bus":       "IoU-bus",
    "Train":     "IoU-train",
    "M.bike":    "IoU-motorcycle",
    "Bike":      "IoU-bicycle",
}

def _safe_eval_dict(dct_str: str) -> Dict[str, Any]:
    """
    尝试安全地把形如 {'mIoU': 50.1, 'IoU-road': 95.3, ...} 的字符串解析成 dict。
    先 literal_eval；失败则启用 eval + 受限内置，并允许 float('nan')/('inf').
    """
    # 先走 literal_eval（最安全）
    try:
        obj = ast.literal_eval(dct_str)
        if isinstance(obj, dict):
            return obj
        raise ValueError("parsed object is not a dict")
    except Exception:
        pass

    # 允许的“安全”名字
    safe_env = {
        "__builtins__": {},  # 一定要是空 dict，不要用 None
        "float": float,
    }

    # 把裸的 NaN/Inf 变成可被 eval 识别的形式（依赖于上面的 float）
    cleaned = (
        dct_str.replace("NaN", "float('nan')")
               .replace("nan", "float('nan')")
               .replace("INF", "float('inf')")
               .replace("Inf", "float('inf')")
               .replace("inf", "float('inf')")
               .replace("-float('inf')", "-1.0*float('inf')")  # 防止被误替换为非法表达式
               .replace("-inf", "-1.0*float('inf')")
    )

    try:
        obj = eval(cleaned, safe_env, {})
        if isinstance(obj, dict):
            return obj
        raise ValueError("parsed object is not a dict")
    except Exception as e:
        # 把前 200 字打印在异常里，便于你在日志中快速定位问题区段
        snippet = dct_str[:200].replace("\n", "\\n")
        raise ValueError(f"无法解析 sem_seg 字典，开头片段: `{snippet}...`；原因: {e!r}")


def parse_log_multi_to_excel(
    log_path: str,
    excel_path: Optional[str] = None,
    sheet_name: str = "metrics"
) -> pd.DataFrame:
    """
    解析包含多次 Inference 与多段 OrderedDict(sem_seg=...) 的日志，
    仅保留 IoU 相关（mIoU、fwIoU、IoU-*），并按指定顺序写入 Excel。
    """
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1) 所有 "Inference done A/B"
    inf_pat = re.compile(r"Inference done\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    inf_matches: List[Tuple[int, int, int]] = [(m.start(), int(m.group(1)), int(m.group(2)))
                                               for m in inf_pat.finditer(text)]

    # 2) 所有 OrderedDict([('sem_seg', {...})])
    od_pat = re.compile(
        r"OrderedDict\(\s*\[\s*\('sem_seg'\s*,\s*({.*?})\)\s*\]\s*\)",
        re.DOTALL
    )
    od_matches: List[Tuple[int, int, str]] = [(m.start(), m.end(), m.group(1))
                                              for m in od_pat.finditer(text)]
    if not od_matches:
        raise RuntimeError("未在日志中找到任何 sem_seg 的 OrderedDict。")

    def pick_prev_inference(pos: int) -> Tuple[Optional[int], Optional[int]]:
        prev = [(p, d, t) for (p, d, t) in inf_matches if p <= pos]
        if not prev:
            return (None, None)
        _, done, total = prev[-1]
        return (done, total)

    rows: List[Dict[str, Any]] = []
    for (start, _, dct_str) in od_matches:
        sem_seg = _safe_eval_dict(dct_str)
        done, total = pick_prev_inference(start)

        row: Dict[str, Any] = {
            "Inference_done": done,
            "Inference_total": total,
            "mIoU": sem_seg.get("mIoU", None),
            "fwIoU": sem_seg.get("fwIoU", None),
        }

        # 记录该段 IoU-* 的原始出现顺序
        iou_keys_in_order = [k for k in sem_seg.keys() if k.startswith("IoU-")]

        # 先放指定顺序（列名使用短名称）
        for short_name, full_key in IOU_ORDER_MAP.items():
            row[short_name] = sem_seg.get(full_key, None)

        # 追加未在清单中的 IoU-*（保持原始键名和出现顺序）
        remaining_iou_keys = [k for k in iou_keys_in_order if k not in IOU_ORDER_MAP.values()]
        for k in remaining_iou_keys:
            row[k] = sem_seg.get(k, None)

        # 不再添加任何 ACC 相关或其它指标（mACC、pACC、ACC-* 全部舍弃）
        rows.append(row)

    df = pd.DataFrame(rows)
    df.index.name = "eval_round"

    # 组织最终列顺序：Meta -> mIoU, fwIoU -> 指定 IoU -> 其它 IoU
    meta_cols = ["Inference_done", "Inference_total", "mIoU", "fwIoU"]
    desired_short_iou_cols = list(IOU_ORDER_MAP.keys())

    # 跨多轮收集“其它 IoU-*”并保序
    other_iou_cols: List[str] = []
    for row in rows:
        for k in row.keys():
            if k.startswith("IoU-") and k not in IOU_ORDER_MAP.values() and k not in other_iou_cols:
                other_iou_cols.append(k)

    final_cols = meta_cols + desired_short_iou_cols + other_iou_cols
    for c in final_cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[final_cols]

    if excel_path is None:
        excel_path = log_path.rsplit(".", 1)[0] + "_metrics_iou.xlsx"
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=True)

    print(f"✅ 已写入仅含 IoU 指标的 Excel：{excel_path}（工作表：{sheet_name}）")
    return df


if __name__ == "__main__":
    # 修改为你的日志路径
    log_file = '/data/zd/TTA_VLM/DeCLIP_CATSeg_Mamba_v2/logs/alb_cs7_eva_b16_r512_MambaV3_lay1_catseg_chunk_8.log'
    parse_log_multi_to_excel(log_file)
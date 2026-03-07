#!/usr/bin/env python3
"""
run_eda_benchmark.py — Run official TG-RAG pipeline on our EDA version-evolution benchmark.

Pipeline:
  1. Build: Read v1.md + v2.md → TemporalGraphRAG.insert() → entity extraction → temporal hierarchy
  2. Query: Iterate qa_pairs.json → graph_rag.query(q, QueryParam(mode="local"))
  3. Output: JSON file compatible with our eval pipeline (calculate_dhr / calculate_ers)

Usage:
  # Set API keys for TG-RAG's LLM provider (e.g., OpenAI)
  export OPENAI_API_KEY="sk-xxx"

  python run_eda_benchmark.py \
    --v1-text ../../benchmark/test_markdown/v1.md \
    --v2-text ../../benchmark/test_markdown/v2.md \
    --gt-dir ../../benchmark/gts \
    --config tgrag/configs/eda_config.yaml \
    --query-mode local \
    --output-dir ../../runs/tgrag
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — add TG-RAG project root to sys.path
# ---------------------------------------------------------------------------
TGRAG_ROOT = os.path.dirname(os.path.abspath(__file__))
if TGRAG_ROOT not in sys.path:
    sys.path.insert(0, TGRAG_ROOT)

# Also add our project's src/ for evaluation metrics
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "../../src"))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)
try:
    from config.prompts import get_baseline_generation_append
except ImportError:
    pass

BASELINES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if BASELINES_DIR not in sys.path:
    sys.path.insert(0, BASELINES_DIR) # Corrected from SRC_DIR to BASELINES_DIR

PROJECT_ROOT = os.path.dirname(os.path.dirname(TGRAG_ROOT))  # codes/
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.sys.path.insert(0, SRC_DIR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="TG-RAG EDA Benchmark Runner: Run official Temporal-GraphRAG on our LLM4EDA dataset"
    )

    # Input files
    parser.add_argument("--v1-text", required=True, help="V1 版本 Markdown/文本路径")
    parser.add_argument("--v2-text", required=True, help="V2 版本 Markdown/文本路径")
    parser.add_argument("--gt-dir", required=True, help="Ground Truth 目录路径 (含 qa_pairs.json 等)")

    # TG-RAG config
    parser.add_argument("--config", default=os.path.join(TGRAG_ROOT, "tgrag/configs/eda_config.yaml"),
                        help="TG-RAG 的 YAML 配置文件路径")
    parser.add_argument("--working-dir", default=None,
                        help="TG-RAG 图缓存目录 (覆盖配置文件中的 working_dir)")
    parser.add_argument("--query-mode", default="local", choices=["local", "global", "naive"],
                        help="TG-RAG 查询模式 (local/global/naive)")
    parser.add_argument("--domain", default="tcl",
                        help="Target script domain/language for code generation (e.g., tcl)")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重新构建图 (忽略缓存)")
    parser.add_argument("--query-only", action="store_true", help="Skip index building, only query existing DB.")

    # Output
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "runs/tgrag"),
                        help="输出目录")

    return parser.parse_args()


def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"tgrag_run_{timestamp}.log")

    logger = logging.getLogger("tgrag_benchmark")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, log_path, timestamp


def main():
    args = parse_args()
    logger, log_path, timestamp = setup_logger(args.output_dir)
    log = logger.info

    log("=" * 70)
    log(f"TG-RAG EDA Benchmark Runner | {datetime.now().isoformat()}")
    log("=" * 70)

    # Log args (mask no sensitive info since API keys come from env)
    log("\n[0] 运行参数:")
    for k, v in vars(args).items():
        log(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 1. Load input texts
    # ------------------------------------------------------------------
    log("\n[1] 加载输入文本...")
    with open(args.v1_text, "r", encoding="utf-8") as f:
        v1_text = f.read()
    with open(args.v2_text, "r", encoding="utf-8") as f:
        v2_text = f.read()
    log(f"  V1: {len(v1_text)} chars ({args.v1_text})")
    log(f"  V2: {len(v2_text)} chars ({args.v2_text})")

    # ------------------------------------------------------------------
    # 2. Load Ground Truth
    # ------------------------------------------------------------------
    log("\n[2] 加载 Ground Truth...")
    with open(os.path.join(args.gt_dir, "qa_pairs.json"), "r", encoding="utf-8") as f:
        gt_qa_data = json.load(f) # Renamed to gt_qa_data
    with open(os.path.join(args.gt_dir, "evolution_rationale.json"), "r", encoding="utf-8") as f:
        gt_rationale = json.load(f)
    log(f"  QA pairs: {len(gt_qa_data.get('qa_pairs', []))}") # Updated to use gt_qa_data
    log(f"  Rationales: {len(gt_rationale['rationales'])}")

    # ------------------------------------------------------------------
    # 3. Build TG-RAG graph
    # ------------------------------------------------------------------
    log("\n[3] 初始化 TG-RAG (Temporal-GraphRAG)...")

    from tgrag import create_temporal_graphrag_from_config, QueryParam

    override_config = {}
    if args.working_dir:
        override_config["working_dir"] = args.working_dir

    graph_rag = create_temporal_graphrag_from_config(
        config_path=args.config,
        config_type="building",
        override_config=override_config if override_config else None,
    )
    log(f"  Working dir: {graph_rag.working_dir}")
    log(f"  Enable local: {graph_rag.enable_local}")
    log(f"  Enable naive RAG: {graph_rag.enable_naive_rag}")

    # Check if graph already built (skip rebuild unless --rebuild)
    graph_marker = os.path.join(graph_rag.working_dir, "graph_chunk_entity_relation.graphml")
    if os.path.exists(graph_marker) and not args.rebuild and not args.query_only: # Added args.query_only
        log("  -> 检测到已有图缓存, 跳过构建 (使用 --rebuild 强制重建)")
    elif args.query_only: # Added query_only logic
        log("  -> --query-only 模式, 跳过图构建.")
        if not os.path.exists(graph_marker):
            log("  -> 警告: 在 --query-only 模式下未找到图缓存. 请确保图已预先构建.")
    else:
        log("\n[3.1] 构建时序知识图谱...")
        log("  -> 插入 V1 文档...")
        documents = [
            {"title": "EDA Clock Timing Constraints Guide v1.0", "doc": v1_text},
            {"title": "EDA Clock Timing Constraints Guide v2.0", "doc": v2_text},
        ]
        graph_rag.insert(documents)
        log("  -> 图构建完成!")

    # ------------------------------------------------------------------
    # 4. Query phase — re-initialize for querying mode
    # ------------------------------------------------------------------
    log("\n[4] 初始化查询模式...")

    # Reload for querying config (may have different model/params)
    query_override = {}
    if args.working_dir:
        query_override["working_dir"] = args.working_dir
    else:
        query_override["working_dir"] = graph_rag.working_dir

    graph_rag_query = create_temporal_graphrag_from_config(
        config_path=args.config,
        config_type="querying",
        override_config=query_override if query_override else None,
    )

    query_param = QueryParam(mode=args.query_mode)
    log(f"  Query mode: {args.query_mode}")

    # ------------------------------------------------------------------
    # 5. Run QA queries
    # ------------------------------------------------------------------
    log(f"\n[5] 运行 QA 查询 ({len(gt_qa_data.get('qa_pairs', []))} 个问题)...")

    qa_results = []
    appender = get_baseline_generation_append(args.domain) if 'get_baseline_generation_append' in globals() else ""

    for idx, qa in enumerate(gt_qa_data.get("qa_pairs", [])):
        base_query = qa.get("query", "")
        expected_rationale = qa.get("expected_rationale", "")
        expected_command = qa.get("expected_command", "")
        deprecated = qa.get("deprecated_terms_in_context", [])
        query_id = qa.get("query_id", f"query_{idx}")
        
        # Actionable QA prompt injection
        formatted_query = base_query + appender

        log(f"\n  [Q{idx+1}]: {base_query}")

        try:
            result = graph_rag_query.query(formatted_query, param=query_param)

            # local mode returns (response, retrieval_detail)
            if isinstance(result, tuple):
                response_text = str(result[0])
                retrieval_detail = result[1] if len(result) > 1 else None
            else:
                response_text = str(result)
                retrieval_detail = None

            log(f"  [A{idx+1}]: {response_text[:500]}")

        except Exception as e:
            log(f"  [Error Q{idx+1}]: {e}")
            response_text = f"[TG-RAG Error]: {e}"
            retrieval_detail = None

        qa_results.append({
            "query_id": query_id,
            "query": base_query,
            "expected_rationale": expected_rationale,
            "expected_command": expected_command,
            "response": response_text.strip(),
            "deprecated_terms": deprecated,
            "retrieval_detail": str(retrieval_detail)[:500] if retrieval_detail else None,
        })

    # ------------------------------------------------------------------
    # 6. Generate evolution rationale via TG-RAG
    # ------------------------------------------------------------------
    log("\n[6] 生成演化推理 (ERS evaluation)...")
    evo_query = (
        "Summarize all changes and evolution between version 1.0 and version 2.0 "
        "of the Clock Timing Constraints Guide. List each change as a bullet point."
    )
    # Ensure we also force the structural representation of domain code in ERS response just in case.
    appender = get_baseline_generation_append(args.domain) if 'get_baseline_generation_append' in globals() else ""
    try:
        evo_result = graph_rag_query.query(evo_query + appender, param=query_param)
        if isinstance(evo_result, tuple):
            evo_rationale = str(evo_result[0])
        else:
            evo_rationale = str(evo_result)
        log(f"  [ERS Rationale]: {evo_rationale[:600]}")
    except Exception as e:
        log(f"  [ERS Error]: {e}")
        evo_rationale = f"[TG-RAG Error]: {e}"

    # ------------------------------------------------------------------
    # 7. Save output JSON (compatible with eval pipeline)
    # ------------------------------------------------------------------
    log("\n[7] 保存结果...")

    output = {
        "baseline": "TG-RAG (Temporal-GraphRAG)",
        "timestamp": timestamp,
        "config": {
            "config_path": args.config,
            "query_mode": args.query_mode,
            "working_dir": graph_rag.working_dir,
        },
        "qa_results": qa_results,
        "ers_rationale": evo_rationale.strip(),
        "gt_rationale_text": "\n".join(gt_rationale["rationales"]),
    }

    result_path = os.path.join(args.output_dir, f"tgrag_result_{timestamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"\n日志已保存: {log_path}")
    log(f"结果已保存: {result_path}")
    log("\n" + "=" * 70)
    log("TG-RAG 运行完成! 请使用 eval_baselines.py 进行 DHR/ERS 打分。")
    log("=" * 70)


if __name__ == "__main__":
    main()

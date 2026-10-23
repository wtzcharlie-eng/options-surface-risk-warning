"""Export a dated risk snapshot; optional evidence-grounded conversational query."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from vol_surface.predictive_service import get_snapshot, build_evidence
    from vol_surface.risk_assistant import LLMConfig, answer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol', default='ag')
    parser.add_argument('--as-of', help='ISO timestamp, Shanghai local time; omitted means current time')
    parser.add_argument('--days', type=int, choices=[1, 2, 3, 4, 5], default=3)
    parser.add_argument('--anchor-dir', type=Path, default=ROOT / 'data_out' / 'anchor')
    parser.add_argument('--model-dir', type=Path, help='Explicit artifact directory; otherwise select the requested version')
    parser.add_argument('--version', choices=['auto', 'v1', 'v2', 'v3'], default='auto')
    parser.add_argument('--question', help='Optional risk question; remote provider requires environment settings')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env', override=False)
        snapshot, report = get_snapshot(args.symbol, args.as_of, args.anchor_dir, args.model_dir, args.version)
        result = {'snapshot': snapshot}
        if args.question:
            response = answer(args.question, build_evidence(snapshot, args.days, report), config=LLMConfig.from_env())
            result['assistant'] = response.to_dict()
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered + '\n', encoding='utf-8')
        else:
            print(rendered)
    except (OSError, ValueError, KeyError) as exc:
        print(f'无法生成风险快照：{type(exc).__name__}；请检查数据、模型和时间参数。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Evaluate the brief's one-off half-position rule using supplied real daily data."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from vol_surface.submission_metrics import first_alert_cvar


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prices', type=Path, required=True)
    ap.add_argument('--alerts', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a=ap.parse_args()
    result=first_alert_cvar(pd.read_csv(a.prices), pd.read_csv(a.alerts))
    import hashlib
    result['input_sha256']={key:hashlib.sha256(path.read_bytes()).hexdigest()
                            for key,path in [('prices',a.prices),('alerts',a.alerts)]}
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('计算完成；真实开盘、交易日完整性及合约处理来源仍须另行核验。')


if __name__=='__main__':main()

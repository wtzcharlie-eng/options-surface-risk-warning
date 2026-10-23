from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
from scripts.build_submission import payload, ROOT, check_links
from scripts.make_package import _excluded, files_to_include
from vol_surface.portable_rule_ml import PortableRuleML


def test_current_delivery_is_generated_and_links_resolve():
    files=payload()
    for name,expected in files.items():
        assert (ROOT/name).read_text()==expected,name
    for name in ['index.html','data_out/doc_paper.html','data_out/doc_package.html','data_out/doc_requirements.html','data_out/doc_delivery_review.html']:
        assert not check_links(ROOT/name,files[name]),name


def test_original_delivery_copies_are_byte_identical_to_recorded_sources():
    req=json.loads((ROOT/'docs/competition_requirements.json').read_text())
    for name,record in req['original_delivery_files'].items():
        assert hashlib.sha256((ROOT/'docs/legacy_delivery'/name).read_bytes()).hexdigest()==record['sha256']


def test_portable_tree_probabilities_match_original_172_golden_values():
    model=PortableRuleML(ROOT/'data_out/rule_ml_portable.json')
    x=np.array([[v]*32 for v in [0.,-1.,1.,3.,float('nan')]])
    # Independently evaluated by frozen joblib in original sklearn 1.7.2.
    expected=[0.009720674182903722,0.10020518861407614,0.648340388953372,0.7621346478931452,0.11157903617817298]
    assert np.max(np.abs(model.predict_proba(x)[:,1]-expected)) < 1e-12
    with pytest.raises(ValueError):model.predict_proba(np.zeros((1,31)))
    proof=json.loads((ROOT/'data_out/rule_ml_portable.validation.json').read_text())
    assert proof['n_compared_rows']==138794 and proof['max_probability_error']==0
    for name,key in [('rule_ml_model.joblib','source_joblib_sha256'),('rule_ml_portable.json','portable_sha256')]:
        assert hashlib.sha256((ROOT/'data_out'/name).read_bytes()).hexdigest()==proof[key]


def test_replayed_closed_scores_match_frozen_results_without_retuning():
    a=json.loads((ROOT/'data_out/submission/audit.json').read_text())
    original=json.loads((ROOT/'data_out/rule_ml.json').read_text())['test']['fused']
    for key in ('precision','recall','avg_lead_min','n_alert'):
        assert a['metrics']['rule_ml']['legacy_closed'][key]==pytest.approx(original[key])
    wp=json.loads((ROOT/'data_out/drl/workpoint.json').read_text())
    for key in ('precision','recall','avg_lead_min','n_alert'):
        assert a['metrics']['drl']['legacy_closed'][key]==pytest.approx(wp[key])
    assert all(c['passed'] for c in a['checks'])
    assert a['business_cvar']['status']=='unverified'
    assert a['human_review']['status']=='pending'


def test_package_includes_new_runtime_and_design_but_excludes_local_credentials():
    files=set(files_to_include())
    assert {'docs/V3实验设计.md','docs/competition_requirements.json','.env.example',
            'requirements-predictive.lock.txt','启动系统.command','data_out/submission/audit.json'} <= files
    for name in ['.git/config','.env','.env.local','dashboard/.env','dashboard/token.key',
                 '.streamlit/secrets.toml','data_out/.cache/model.json','scripts/__pycache__/x.pyc']:
        assert _excluded(name),name
    assert not _excluded('.env.example')

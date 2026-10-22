"""Task semantics must survive the forecast-to-conversation boundary."""
import pytest

from vol_surface.predictive_service import build_evidence
from vol_surface.risk_assistant import LLMConfig, answer


def snapshot(task='onset', mode='hazard_baseline'):
    return {
        'model_version': 'v2', 'symbol': 'ag', 'as_of': '2026-01-12T15:00:00',
        'data_timestamp': '2026-01-12T15:00:00', 'prediction_task': task,
        'prediction_task_definition': ('当前正常，未来窗口首次超阈值' if task == 'onset' else '当前已异常，未来窗口至少一次再次超阈值，不代表全程异常'),
        'horizons': [{'days': 3, 'probability': .4, 'threshold': .6, 'alert': False,
                      'model_status': mode, 'baseline_probability': .3}],
        'signals': {'currently_exceeding': task == 'persistence'},
        'evaluation': {'3': {'brier': .2, 'baseline_brier': .19, 'brier_skill': -.0526}},
        'quality': {'horizon_probabilities_consistent': True}, 'warnings': [], 'status':'ready',
    }


def test_onset_task_is_cited_and_baseline_is_not_called_complex_model_gain():
    evidence = build_evidence(snapshot(), 3)
    response = answer('解读当前风险', evidence, config=LLMConfig())
    assert '正常状态下的新异常出现' in response.text
    assert '[prediction_task_definition]' in response.text
    assert '条件风险率统计基准' in response.text
    assert '不代表特征模型已获得增量' in response.text
    assert evidence['forecast']['probability'] == .4


def test_abnormal_task_does_not_imply_uninterrupted_persistence_or_recovery():
    evidence = build_evidence(snapshot(task='persistence'), 3)
    response = answer('解释这个预测', evidence, config=LLMConfig())
    assert '不代表全程异常' in response.text
    assert '[prediction_task]' in response.text
    assert '恢复时间为' not in response.text


def test_evaluation_is_selected_horizon_and_uses_the_strong_baseline():
    evidence = build_evidence(snapshot(), 3)
    response = answer('预测效果可靠吗？', evidence, config=LLMConfig())
    assert '[evaluation.brier]' in response.text
    assert '不能据此宣称有可用增量预测能力' in response.text
    assert '同任务' in evidence['evaluation_baseline']


def test_v2_contribution_is_conditional_hazard_not_cumulative_probability():
    source = snapshot(mode='joint_logistic')
    source['horizons'][0]['top_contributions'] = [{
        'feature':'atm_iv', 'value':.2, 'log_odds_contribution':.3,
        'hazard_step':3, 'contribution_scale':'conditional_hazard_log_odds'}]
    response = answer('为什么会有这个特征贡献？', build_evidence(source, 3), config=LLMConfig())
    assert '指定交易日条件风险率' in response.text
    assert '不是累计风险概率的直接加总' in response.text


def evidence_with_operational_metrics():
    source = snapshot(mode='joint_logistic')
    source['evaluation']['3'].update(n=100, positives=20, precision=.8, recall=.75)
    source['evaluation_details'] = {'3': {'operational': {
        'n_total': 100, 'n_available': 50, 'probability_coverage': .5,
        'predictive_signal_coverage': .3, 'recall_all_origins': .25,
        'precision': .4, 'false_alarms': 6, 'abstentions': 50,
        'abstentions_count_as_non_alerts': True,
    }}}
    return build_evidence(source, 3)


@pytest.mark.parametrize('question', [
    '概率可用率和增量模型覆盖率是多少？',
    '运行召回率含不含弃权？',
    '有多少误报和弃权原点？',
])
def test_local_operational_questions_use_protected_metrics(question):
    response = answer(question, evidence_with_operational_metrics(), config=LLMConfig())
    assert '50.00% [operational_evaluation.probability_coverage]' in response.text
    assert '30.00% [operational_evaluation.predictive_signal_coverage]' in response.text
    assert '25.00% [operational_evaluation.recall_all_origins]' in response.text
    assert '弃权原点仍计入事件分母' in response.text
    assert '[operational_evaluation.abstentions_count_as_non_alerts]' in response.text
    assert '[evaluation.recall]' not in response.text
    assert '[evaluation_context]' in response.text


def test_reliability_answer_keeps_research_skill_and_operational_coverage_distinct():
    response = answer('预测效果可靠吗？', evidence_with_operational_metrics(), config=LLMConfig())
    assert '保护前研究 · Brier 分数' in response.text
    assert '[evaluation.brier]' in response.text
    assert '[evaluation.baseline_brier]' in response.text
    assert '[evaluation.brier_skill]' in response.text
    assert '运行保护后 · 概率可用率（含统计基准）' in response.text
    assert '[operational_evaluation.recall_all_origins]' in response.text
    assert '[evaluation.recall]' not in response.text
    assert '[evaluation_context]' in response.text
    assert '完整测试期的事后统计' in response.text


def test_unavailable_operational_statistics_are_not_replaced_with_raw_recall():
    response = answer('运行召回率与弃权如何？', build_evidence(snapshot(), 3), config=LLMConfig())
    assert '未提供运行保护后的覆盖率' in response.text
    assert '不能用保护前研究指标代替' in response.text


def test_calibrated_v2_baseline_and_disabled_alerts_hide_sentinel_threshold():
    source = snapshot()
    source['horizons'][0]['threshold'] = 1.000001
    response = answer('解读当前风险', build_evidence(source, 3), config=LLMConfig())
    assert '同任务已校准条件基准概率' in response.text
    assert '训练期历史基础发生率' not in response.text
    assert '统计参考，预测预警关闭' in response.text
    assert '[forecast.threshold]' not in response.text
    assert '未达到预警阈值' not in response.text

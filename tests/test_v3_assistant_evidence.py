from tests.test_v2_assistant_evidence import snapshot
from vol_surface.predictive_service import build_evidence
from vol_surface.risk_assistant import LLMConfig,answer


def evidence():
    s=snapshot(mode='joint_logistic_shrunk')
    s['model_version']='v3'
    s['model_context']={'feature_hazard_weight':.5,'statistical_hazard_weight':.5,
                        'explanation':'风险率层融合后合成累计概率，不是可相加的特征贡献'}
    s['evaluation']['3'].update(n=100,prevalence=.3,mean_probability=.2,calibration_bias=-.1,
                               log_loss=.6,ece_5_fixed_bins=.1)
    s['evaluation_details']={'3':{'operational':{'n_total':100,'probability_coverage':.9,
        'predictive_signal_coverage':.8,'recall_all_origins':.4,'precision':.6,
        'false_alarms':4,'abstentions':10,'abstentions_count_as_non_alerts':True}}}
    return build_evidence(s,3)


def test_v3_mixture_is_not_explained_as_additive_linear_features():
    response=answer('为什么会有这些特征贡献？',evidence(),config=LLMConfig())
    assert '[model_context.feature_hazard_weight]' in response.text
    assert '不是累计概率或单项特征贡献' in response.text
    assert '向统计基准收缩' in response.text


def test_v3_reliability_uses_operational_recall_and_honest_baseline():
    response=answer('预测效果可靠吗？',evidence(),config=LLMConfig())
    assert '[evaluation.brier]' in response.text
    assert '[operational_evaluation.recall_all_origins]' in response.text
    assert '[evaluation_context]' in response.text
    assert '不能据此宣称有可用增量预测能力' in response.text


def test_calibration_question_uses_the_provided_diagnostics():
    response=answer('概率校准有偏差吗？',evidence(),config=LLMConfig())
    assert '[evaluation.mean_probability]' in response.text
    assert '[evaluation.prevalence]' in response.text
    assert '[evaluation.calibration_bias]' in response.text

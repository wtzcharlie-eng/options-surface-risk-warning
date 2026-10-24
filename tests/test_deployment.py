from pathlib import Path
from dataclasses import replace
import io,json,os
from unittest.mock import MagicMock,patch
import pytest
from dotenv import dotenv_values
from deployment.runtime import RuntimeSettings,DeploymentConfigError
from scripts.make_package import submission_env,ROOT
from vol_surface.risk_assistant import LLMConfig,_request_completion,AssistantConfigError


def test_platform_template_lists_all_guide_fields_without_credentials():
    env=dotenv_values(stream=io.StringIO(submission_env()))
    assert {'MODEL_API_KEY','MODEL_BASE_URL','MODEL_NAME','EMBEDDING_API_KEY','SERVICE_PORT'}<=set(env)
    assert not env['MODEL_API_KEY'] and not env['EMBEDDING_API_KEY']
    assert env['MODEL_NAME']=='qwen3.8-27b-vllm'
    assert RuntimeSettings.from_env(env).public_status()['missing_model_fields']==['MODEL_API_KEY']


@pytest.mark.parametrize('field,value',[('SERVICE_PORT','0'),('SERVICE_PORT','65536'),('SERVICE_PORT','abc'),('SERVICE_BASE_PATH','../escape'),('MODEL_PROVIDER','openai'),('MODEL_ENABLE_THINKING','maybe'),('MODEL_MAX_TOKENS','0'),('MODEL_TIMEOUT_SECONDS','nan')])
def test_invalid_deployment_settings_fail_closed(field,value):
    with pytest.raises((ValueError,DeploymentConfigError)):
        RuntimeSettings.from_env({field:value})


def test_platform_settings_map_to_existing_assistant_without_openai_key_fallback(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','unrelated-global-key')
    s=RuntimeSettings.from_env({})
    with patch.dict(os.environ,{},clear=False):
        s.apply();c=LLMConfig.from_env()
        assert c.provider=='local' and c.api_key==''
        assert c.enable_thinking is False and c.max_tokens==1400
    env={'MODEL_NAME':'test-qwen','MODEL_BASE_URL':'https://example.invalid/v1','MODEL_API_KEY':'test-only-key','SERVICE_PORT':'8123','SERVICE_BASE_PATH':'/risk/demo/'}
    s=RuntimeSettings.from_env(env)
    assert s.port==8123 and s.base_path=='risk/demo'
    assert 'test-only-key' not in json.dumps(s.public_status()) and 'test-only-key' not in repr(s)
    with patch.dict(os.environ,{},clear=False):
        s.apply();c=LLMConfig.from_env();c.validate()
        assert c.model=='test-qwen' and c.base_url==env['MODEL_BASE_URL']


@pytest.mark.parametrize('url',['https://api.openai.com/v1','https://api.anthropic.com/v1','https://generativelanguage.googleapis.com/v1','http://remote.invalid/v1','https://bad.invalid/v1?key=x'])
def test_competition_endpoint_rejects_unsupported_providers_and_unsafe_urls(url):
    with pytest.raises(ValueError):RuntimeSettings.from_env({'MODEL_BASE_URL':url})


def test_qwen_transport_uses_flat_chat_template_option_and_output_limit():
    cfg=LLMConfig(provider='openai-compatible',model='test-qwen',base_url='https://example.invalid/v1',api_key='test-only-key',enable_thinking=False,max_tokens=1400)
    response=MagicMock();response.__enter__.return_value=response
    response.read.return_value=b'{"choices":[{"message":{"content":"{}"}}]}'
    opener=MagicMock();opener.open.return_value=response
    with patch('vol_surface.risk_assistant.build_opener',return_value=opener):
        _request_completion(cfg,[{'role':'user','content':'test'}])
    req=opener.open.call_args.args[0];body=json.loads(req.data)
    assert body['chat_template_kwargs']=={'enable_thinking':False}
    assert body['max_tokens']==1400 and 'test-only-key' not in req.data.decode()
    assert req.full_url=='https://example.invalid/v1/chat/completions'
    with pytest.raises(AssistantConfigError):replace(cfg,max_tokens=True).validate()


def test_private_environment_is_an_explicit_package_option_with_allowlisted_variables(tmp_path,monkeypatch):
    import scripts.make_package as package
    template=(ROOT/'deployment/competition.env').read_text()
    (tmp_path/'deployment').mkdir()
    (tmp_path/'deployment/competition.env').write_text(template)
    (tmp_path/'.env').write_text(template.replace('MODEL_API_KEY=\n','MODEL_API_KEY=test-private-token\n')+'UNRELATED_SECRET=do-not-copy\n')
    monkeypatch.setattr(package,'ROOT',tmp_path)
    public=dotenv_values(stream=io.StringIO(package.submission_env(False)))
    private=dotenv_values(stream=io.StringIO(package.submission_env(True)))
    assert public['MODEL_API_KEY']=='' and private['MODEL_API_KEY']=='test-private-token'
    assert 'UNRELATED_SECRET' not in private


def test_numeric_grounding_allows_honest_rounding_without_mixing_fraction_and_percent():
    from vol_surface.risk_assistant import _validate_completion
    fields={'forecast.probability':.0832699,'forecast.threshold':.20425,'evaluation.n':74}
    raw=json.dumps({'answer':'风险概率约 8.33%，阈值约 20.43%。[forecast.probability] [forecast.threshold]','cited_fields':['forecast.probability','forecast.threshold']},ensure_ascii=False)
    assert '8.33%' in _validate_completion(raw,fields,'')[0]
    for text,cited in [('概率 0%。[forecast.probability]',['forecast.probability']),
                       ('概率 99%。[forecast.probability]',['forecast.probability']),
                       ('概率为0。[forecast.probability]',['forecast.probability']),
                       ('样本率74%。[evaluation.n]',['evaluation.n'])]:
        with pytest.raises(ValueError):
            _validate_completion(json.dumps({'answer':text,'cited_fields':cited}),fields,'')


def test_compact_remote_evidence_keeps_task_quality_and_operational_limits():
    from vol_surface.risk_assistant import _compact_fields
    fields={'prediction_task':'onset','target_definition':'proxy only','mode':'history',
            'forecast.probability':.1,'forecast.evaluation.n':999,'forecast.threshold':.2,
            'evaluation.brier':.2,'operational_evaluation.recall_all_origins':.3,
            'data_quality.warnings.0':'stale','model_context.explanation':'not causal'}
    compact=_compact_fields(fields)
    assert 'forecast.evaluation.n' not in compact
    assert set(compact)==set(fields)-{'forecast.evaluation.n'}


def test_platform_retries_one_transient_503_but_never_auth_failures():
    from urllib.error import HTTPError
    cfg=LLMConfig(provider='openai-compatible',model='test',base_url='https://example.invalid/v1',api_key='test-only-key',compact_evidence=True)
    response=MagicMock();response.__enter__.return_value=response
    response.read.return_value=b'{"choices":[{"message":{"content":"ok"}}]}'
    opener=MagicMock();opener.open.side_effect=[HTTPError('https://example.invalid',503,'busy',{},io.BytesIO()),response]
    with patch('vol_surface.risk_assistant.build_opener',return_value=opener),patch('vol_surface.risk_assistant.time.sleep'):
        assert _request_completion(cfg,[])=='ok'
    assert opener.open.call_count==2
    assert 0<opener.open.call_args.kwargs['timeout']<=30
    opener.open.side_effect=HTTPError('https://example.invalid',401,'no',{},io.BytesIO());opener.reset_mock()
    with patch('vol_surface.risk_assistant.build_opener',return_value=opener),pytest.raises(HTTPError):
        _request_completion(cfg,[])
    assert opener.open.call_count==1


def test_platform_reference_array_is_explicitly_labelled_and_still_grounded():
    from vol_surface.risk_assistant import answer
    evidence={'forecast':{'probability':.0832699}}
    cfg=LLMConfig(provider='openai-compatible',model='test',base_url='https://example.invalid/v1',api_key='test-only-key',compact_evidence=True)
    good=json.dumps({'answer':'概率约 8.33%。','cited_fields':['forecast.probability']},ensure_ascii=False)
    with patch('vol_surface.risk_assistant._request_completion',return_value=good):
        r=answer('风险？',evidence,config=cfg)
    assert r.mode=='llm' and '参考字段（模型声明）' in r.text and '[forecast.probability]' in r.text
    for bad in [good.replace('8.33','99'),good.replace('forecast.probability','invented.probability')]:
        with patch('vol_surface.risk_assistant._request_completion',return_value=bad):
            assert answer('风险？',evidence,config=cfg).mode=='local'

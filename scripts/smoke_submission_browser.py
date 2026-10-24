"""Browser acceptance for the deployed single-port UI (Playwright is QA-only)."""
from pathlib import Path
import argparse,json,re,time
from playwright.sync_api import sync_playwright,expect


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--url',required=True);ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--prefix-url');args=ap.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    checks=[];errors=[];bad=[]
    expect.set_options(timeout=120000)
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        page.set_default_timeout(120000)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('response',lambda r:bad.append({'path':r.url.split('?')[0],'status':r.status}) if r.status>=400 and '/static/' in r.url else None)
        page.goto(args.url,wait_until='domcontentloaded')
        page.get_by_role('heading',name='看见风险，再追问原因',exact=True).wait_for()
        expect(page.get_by_test_id('stMetric')).to_have_count(3)
        page.screenshot(path=str(args.out_dir/'predictive-desktop.png'),full_page=True)
        checks.append('homepage_and_prediction_cards')
        page.get_by_role('button',name='解读当前风险',exact=True).click()
        page.get_by_text('本地证据解读（未调用 LLM）',exact=True).wait_for()
        box=page.locator('textarea[placeholder^="例如："]')
        box.fill('预测效果可靠吗？');box.press('Enter')
        page.wait_for_function("document.querySelectorAll('[data-testid=stChatMessage]').length >= 4")
        assert 'Brier' in page.locator('body').inner_text()
        checks.append('offline_chat_and_followup_explicitly_labelled')
        page.get_by_role('tab',name='数据与快照',exact=True).click()
        with page.expect_download() as dl:
            page.get_by_role('button',name='下载当前预测快照',exact=True).click()
        file=args.out_dir/'downloaded-snapshot.json';dl.value.save_as(str(file))
        snapshot=json.loads(file.read_text())
        assert snapshot['model_version']=='v3' and len(snapshot['horizons'])==5
        assert not any(k in snapshot for k in ['api_key','MODEL_API_KEY'])
        checks.append('valid_snapshot_download')
        for title,heading,name,minimum in [
            ('曲面预警','📊 期权波动率曲面风控预警系统','surface',4),
            ('图谱解释','风险传导路径推理','graph',0),
            ('DRL 推演','DRL 自适应预警 · 推理接口','drl',2),
            ('比赛对照','期权曲面风控 · 比赛交付总览','review',3)]:
            page.get_by_role('link',name=title,exact=True).click()
            if heading:page.get_by_role('heading',name=heading,exact=True).wait_for()
            else:page.get_by_role('heading',level=1).wait_for()
            if name=='surface':expect(page.get_by_test_id('stPlotlyChart').nth(1)).to_be_visible()
            if name=='graph':expect(page.get_by_test_id('stSelectbox')).to_have_count(2)
            page.wait_for_timeout(500)
            assert page.get_by_test_id('stException').count()==0
            if minimum:expect(page.get_by_test_id('stMetric')).to_have_count(minimum)
            page.screenshot(path=str(args.out_dir/(name+'-desktop.png')),full_page=True)
            checks.append('navigation_'+name)
        page.get_by_role('link',name='预测与问答',exact=True).click()
        page.get_by_role('heading',name='看见风险，再追问原因',exact=True).wait_for()
        page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(500)
        page.screenshot(path=str(args.out_dir/'predictive-mobile.png'),full_page=True)
        width=page.evaluate('({viewport:innerWidth,content:document.documentElement.scrollWidth})')
        assert width['content']<=width['viewport']+1,width
        checks.append('mobile_no_document_overflow')
        if args.prefix_url:
            page.set_viewport_size({'width':1440,'height':1100})
            page.goto(args.prefix_url,wait_until='domcontentloaded')
            page.get_by_role('heading',name='看见风险，再追问原因',exact=True).wait_for()
            expect(page.get_by_test_id('stMetric')).to_have_count(3)
            page.get_by_role('link',name='图谱解释',exact=True).click()
            page.get_by_role('heading',name='风险传导路径推理',exact=True).wait_for()
            expect(page.get_by_test_id('stSelectbox')).to_have_count(2)
            assert page.url.startswith(args.prefix_url) and page.url.endswith('/graph')
            page.get_by_role('link',name='预测与问答',exact=True).click()
            expect(page.get_by_test_id('stMetric')).to_have_count(3)
            page.screenshot(path=str(args.out_dir/'prefixed-desktop.png'),full_page=True)
            checks.append('proxy_prefix_and_custom_port')
        assert not errors,errors
        assert not bad,bad
        browser.close()
    result={'passed':True,'checks':checks,'javascript_errors':errors,'failed_static_resources':bad,
            'llm_mode':'offline-local-evidence; real platform test recorded separately',
            'target':'linux/amd64 container','resource_limit':'2 CPU, 4 GiB'}
    (args.out_dir/'browser-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

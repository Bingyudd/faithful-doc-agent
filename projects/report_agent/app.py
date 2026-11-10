"""Local-only review surface. Start from repository root with python -m streamlit."""
import difflib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import streamlit as st
from projects.paper_common.io import ROOT,digest,read_json,write_json
from projects.paper_common.prepare_data import extract_pdf
from projects.paper_common.snapshot_provider import SnapshotProvider as Provider
from projects.report_agent.cli import local_key
from projects.report_agent.engine import ReportEngine

st.set_page_config(page_title='技术文档审校台',page_icon='📄',layout='wide')
st.title('技术文档审校台')
st.caption('从资料生成有来源的技术初稿，逐项核对后导出。')
workspace=st.sidebar.radio('工作区',['生成与审校','抽样复核'])
if workspace=='抽样复核':
    from projects.report_agent.review import show_review
    show_review()
    st.stop()
manifest=read_json(ROOT/'data/paper/sources.json')
corpus=read_json(ROOT/'data/paper/corpus.json')['elements']
names={f"{d['id']} · {d['device_name'][:65]}":d for d in manifest['sources']}

with st.sidebar:
    st.subheader('新建生成任务')
    source_mode=st.radio('资料来源',['公开样例','上传资料'])
    document=None
    if source_mode=='公开样例':
        selected=st.selectbox('选择资料',list(names))
        document=names[selected]
        st.link_button('查看原始公开文件',document['url'])
        if document['sparse_pages']:st.warning('部分页面文字较少，需要人工检查扫描内容。')
    else:
        uploaded=st.file_uploader('PDF、TXT 或 Markdown',type=['pdf','txt','md'])
        if uploaded:
            payload=uploaded.getvalue()
            doc_id='upload-'+digest(payload)[:12]
            document={'id':doc_id,'device_name':uploaded.name,'sha256':digest(payload),'split':'interactive'}
            if uploaded.name.lower().endswith('.pdf'):
                path=ROOT/'artifacts/paper/ui'/f'{doc_id}.pdf'
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
                corpus,sparse=extract_pdf(path,doc_id)
                if sparse:st.warning('部分页面可能需要 OCR；本工具不会猜测扫描内容。')
            else:
                text=payload.decode('utf-8-sig')
                corpus=[{'id':f'{doc_id}:p1:c{i//1100}','document_id':doc_id,'page':1,'text':text[i:i+1100]}
                        for i in range(0,len(text),1100) if text[i:i+1100].strip()]
    task=st.selectbox('目标文件',['产品技术概述','测试证据摘要'])
    template='overview' if task=='产品技术概述' else 'testing'
    mode=st.selectbox('生成方式',['证据审校','章节拆解','自主规划','直接生成'])
    method={'证据审校':'reviewed','章节拆解':'decomposed','自主规划':'agentic','直接生成':'direct'}[mode]
    model=st.selectbox('模型',['deepseek-v4.1-flash','hy4-preview-f'])
    if st.button('生成初稿',type='primary',disabled=document is None):
        with st.spinner('正在检索资料并生成初稿…'):
            try:
                provider=Provider(model,api_key=local_key())
                result=ReportEngine(provider,corpus).run(document,template,method,ROOT/'artifacts/paper/ui/runs')
                st.session_state['result']=result
            except Exception as exc:st.error(type(exc).__name__+': '+str(exc))

result=st.session_state.get('result')
if not result:
    st.info('在左侧选择资料与目标文件，或加载已完成的本地实验。')
    saved=sorted((ROOT/'artifacts/paper/raw/reports_v3').glob('*/*/result.json'))
    if not saved:saved=sorted((ROOT/'artifacts/paper/raw/reports_v2').glob('*/*/result.json'))
    if saved:
        options={}
        for path in saved:
            identity=read_json(path)['identity']
            label=f"{identity['document']['id']} · {identity['document']['device_name'][:45]} · {identity['template']} · {identity['method']}"
            options[label]=path
        chosen=st.selectbox('已完成报告',list(options))
        if st.button('打开报告'):
            st.session_state['result']=read_json(options[chosen]);st.rerun()
elif result['status']!='complete':
    st.error('生成未完成：'+result.get('error','未知错误'))
    st.info('运行记录已保存，可在服务恢复后重试。')
else:
    left,right=st.columns([1.4,1])
    with left:
        st.subheader('初稿')
        current_document=result['identity']['document']
        st.caption('当前审校资料：'+current_document['id']+'；左侧设置用于下一次生成。')
        if current_document.get('url'):st.link_button('当前报告的原始资料',current_document['url'])
        displayed=result['markdown'].replace('# ','### ',1)
        for code,label in [('review_required','需人工复核'),('missing','待补资料'),('conflict','存在冲突')]:
            displayed=displayed.replace('状态：'+code,'状态：'+label)
        st.markdown(displayed)
        problems=sum(len(v) for v in result['audits'].values())
        if problems:st.warning(f'有 {problems} 项自动检查提示，请逐条核对。')
        st.caption('自动检查通过也需要人工核对语义、产品归属和遗漏。')
        with st.expander('编辑与导出',expanded=True):
            edited=st.text_area('审校后的 Markdown',value=result['markdown'],height=280,key='edit-'+result['run_id'])
            notes=st.text_input('审校备注',key='notes-'+result['run_id'])
            st.download_button('导出 Markdown',edited,'技术文档初稿.md','text/markdown')
            export={**result,'review':{'edited_markdown':edited,'notes':notes,'status':'user_edit_not_automatically_verified'}}
            st.download_button('导出含证据的 JSON',json.dumps(export,ensure_ascii=False,indent=2),'技术文档证据.json','application/json')
            if st.button('保存审校版本'):
                write_json(ROOT/'artifacts/paper/ui'/f"review_{result['run_id'][:20]}.json",export)
                st.success('审校版本已保存。原始生成结果仍保留。')
    with right:
        st.subheader('原文证据')
        evidence=result['evidence']
        if evidence:
            anchor=st.selectbox('定位引用',list(evidence),key='anchor-'+result['run_id'])
            item=evidence[anchor]
            st.caption(f"{item['document_id']} · 第 {item['page']} 页")
            st.text(item['text'])
        with st.expander('检查提示'):
            st.json(result['audits'])
        with st.expander('自动修订差异'):
            for revision in result['revisions']:
                st.caption(f"{revision['section']} · 第 {revision['round']} 轮")
                before=json.dumps(revision['before'],ensure_ascii=False,indent=2).splitlines()
                after=json.dumps(revision['after'],ensure_ascii=False,indent=2).splitlines()
                st.code('\n'.join(difflib.unified_diff(before,after,fromfile='修订前',tofile='修订后')),language='diff')
        with st.expander('运行统计'):
            st.json(result['usage'])

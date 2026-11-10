"""Manual annotation UI. No automatic or assistant labels are marked as human."""
import streamlit as st
from filelock import FileLock
from projects.paper_common.io import ROOT,read_json,write_json


def show_review():
    path=ROOT/'artifacts/paper/human_review_queue.json'
    st.header('固定抽样复核')
    if not path.exists():
        st.info('实验汇总后将生成固定抽样队列。');return
    queue=read_json(path)
    st.caption(f"已复核 {queue['completed']} 项；当前已生成 {len(queue['items'])} 项，完整队列 {queue['required']} 项。")
    if not queue['items']:return
    options={f"样本 {i+1:02d} · {'文档初稿' if x['kind']=='report' else '摘要'} · {'待核对' if x['status']=='pending' else '已核对'}":x
             for i,x in enumerate(queue['items'])}
    entry=options[st.selectbox('选择样本',list(options))]
    raw=read_json(ROOT/entry['path'])
    if raw['status']!='complete':
        st.warning('该运行未生成完整输出，已计入实验失败。')
        st.text(raw.get('error',''))
    else:
        left,right=st.columns(2)
        with left:
            st.subheader('生成内容')
            st.markdown(raw['markdown'] if entry['kind']=='report' else raw['text'])
        with right:
            st.subheader('输入证据')
            if entry['kind']=='report':
                for item in raw['evidence'].values():
                    with st.expander(f"{item['document_id']} · 第 {item['page']} 页 · {item['id']}"):
                        st.text(item['text'])
            else:st.text(raw['used_source'])
    with st.form('annotation-'+entry['run_id']):
        support=st.radio('生成内容是否得到来源支持？',['未判断','全部支持','部分不支持','无法判断'])
        issue=st.multiselect('发现的问题',['错误产品归属','数值或单位错误','遗漏重要条件','重复生成','引文不匹配','其他'])
        comment=st.text_area('核对依据／备注',value=entry.get('comment',''))
        reviewer=st.text_input('核对人')
        submitted=st.form_submit_button('保存人工复核')
    if submitted:
        if not reviewer.strip() or support=='未判断':
            st.error('请填写核对人并完成来源支持判断。');return
        with FileLock(str(path)+'.lock'):
            current=read_json(path)
            target=next(x for x in current['items'] if x['run_id']==entry['run_id'])
            target.update(status='reviewed',reviewer=reviewer.strip(),support=support,issues=issue,comment=comment)
            current['completed']=sum(x['status']=='reviewed' for x in current['items'])
            write_json(path,current)
        st.success('人工复核已保存。不会修改原始模型输出或自动指标。')

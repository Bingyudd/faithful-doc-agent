# 上下文忠实解码

按 CFB（ACL Findings 2026）公式独立实现解码算法，直接控制 token logits。代码不依赖报告 Agent、训练数据或远程生成接口。

## 运行

在 worktree 根目录的 WSL 项目环境运行：

```bash
export LD_LIBRARY_PATH=/home/bingyu/miniconda3/envs/physical_ex/lib
/home/bingyu/.venvs/ai2resu-gpu/bin/python -m projects.faithful_decoding.cli evaluate --model main --dataset data/paper/xsum_dev.jsonl
/home/bingyu/.venvs/ai2resu-gpu/bin/python -m projects.faithful_decoding.cli generate --model main --source-file example.txt --methods token
```

主模型 Qwen2.5-1.5B-Instruct，辅模型 Qwen2.5-0.5B-Instruct；revision 固定在 `data/paper/provenance.json`。原论文使用 7B–13B 模型，本项目属于小模型方法复现及场景迁移，不是原表分数的严格复现。

## 方法与实现

- `greedy`：原始 argmax。
- `cad`：有／无来源条件的 log-prob 对比，两套 KV cache 同步消费同一已生成 token。
- `static`：仅给来源 span 中的 token 加固定偏置。
- `context`：用首次预测的有／无来源分布计算 base-2 JSD，一次决定样本级偏置。
- `token`：按末层平均头注意力，聚合来源内重复 token；结合 token embedding 与来源 embedding 的平均余弦相似度，按平均相关度归一化。
- `no_attention`／`no_semantic`：分别移除一种相关度信号。

语义相关度利用归一化 embedding 的均值计算，避免构造来源长度平方的余弦矩阵。预填充至倒数第二 token，再读取最后 query 的注意力，避免为所有层保留完整的来源平方注意力张量。数值单测检查它与完整前向一致。

来源 token 通过 fast tokenizer 的字符 offset 定位，排除提示词、特殊 token 和生成历史。注意力缺失、非有限值或非正归一化均值会显式报错，不切换到另一种隐式算法。

## 固定实验

默认 `variant=author_reference` 对齐固定作者提交：余弦相似度先映射 `(cos+1)/2`，来源注意力再次 softmax。`paper_equations` 保留直接按公式实现的原始版本。开发集的重复退化触发了这次源码核对；两版独立保存，详见 `docs/12_论文驱动项目实验报告.md`。

XSum 50 开发／200 测试，公开技术文本 40 开发／60 测试。输入上限 1024 个来源 token、输出上限 64 token；截断和是否正常 EOS 均记录。所有方法同样预算，解码为贪心而非采样。

每条输出保存来源实际窗口、生成文本、逐 token 诊断、代码／输入哈希、耗时、峰值显存。中断后按运行身份跳过已完成项；错误结果保留，不静默丢弃。

`python -m projects.paper_common.evaluate decoding --input artifacts/paper/raw/decoding_reference --output artifacts/paper/decoding_report.json` 重算指标。ROUGE-L 为项目的无词干化 LCS F1；数字对应不是语义事实准确率；FactKB 是独立模型预测且记录 512-token 裁剪。技术文本无人工摘要参考，不能报告该子集的摘要 ROUGE 分数。

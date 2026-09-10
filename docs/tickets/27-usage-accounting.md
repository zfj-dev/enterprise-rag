# 27: per-query 用量记账（usage 优先 + 本地兜底 + 口径标注）

**What to build:** 每次生成记一条用量记录：输入 / 输出 token、模型、**口径来源**；provider 返回的 usage 优先，拿不到回退本地分词器，两者皆无则标记**不可用**。

**Blocked by:** 17 上下文预算装配 + token 计数缝

**Status:** ready-for-agent

- [ ] 有 provider usage 时**用 usage** 并标注口径来源
- [ ] 无 usage 时回退**本地分词器**（复用 0003 的分词器缝）并标注
- [ ] 两者皆无时标记**不可用**，**不得记成 0**
- [ ] 每条记录都能看出"这是账单口径还是估算口径"

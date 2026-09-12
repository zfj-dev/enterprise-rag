# 05: RAGAS 四项

**What to build:** faithfulness / answer-relevancy / context-precision / recall 四项指标数字，裁判口径固定，使数字**可跨时间、跨项目比较**。

**Blocked by:** 01 评测核心注入缝

**Status:** ready-for-agent

- [ ] 四项指标均出现在报告中
- [ ] 裁判固定为 `qwen-turbo`、温度 0，报告**显式写明该口径**
- [ ] 裁判从外部注入，可用 stub 裁判做确定性单测（无网络）
- [ ] 缺 API Key 或裁判不可用时**明确报错**，不得静默给出错误数字

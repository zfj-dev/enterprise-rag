# 10: MCP 往返打通 + Calculator 工具

**What to build:** 用一个**算术工具**把"工具定义 → MCP server → client 发现并调用"整条路打通，并证明不可信的表达式进不来。传输默认本地、不占端口。

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] 工具带 MCP 规范元数据（名称 / 描述 / 输入 schema），可被 MCP client **发现**
- [ ] 经 MCP 调用返回正确数值
- [ ] 非算术表达式（名字绑定 / 属性访问 / 下标 / 导入等）被**拒绝**；实现**不使用 eval/exec**
- [ ] 传输层在测试中被 stub，单测**不起真实 MCP 进程、不联网**

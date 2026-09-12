# 16: MCP server 对外可挂载

**What to build:** **任意 MCP 客户端**都能挂载这个 server，看到并调用三个工具——这就是 ROADMAP 的 P2b 交付物，也是"互操作"能力的证明。

**Blocked by:** 12 KbRetrieve 工具；13 SqlQuery 工具

**Status:** ready-for-agent

- [ ] 三个工具（KbRetrieve / SqlQuery / Calculator）均可被发现，含名称、描述与**输入 schema**
- [ ] 由**外部 MCP 客户端**挂载并成功调用至少一个工具
- [ ] 默认**仅本地**可访问；对外暴露需显式配置 + 鉴权
- [ ] 工具可发现性有**契约测试**（工具集与 schema 形状）

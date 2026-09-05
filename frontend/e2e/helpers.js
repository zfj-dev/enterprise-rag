// 共享测试工具：拦截 /api/v1/** 以确定性模拟后端；让 app 在登录/带 token 后能正常"启动"。
const { expect } = require('@playwright/test');

function json(route, body, status = 200) {
  return route.fulfill({
    status,
    body: JSON.stringify(body),
    headers: { 'content-type': 'application/json' },
  });
}

// 未知路径的默认响应：GET 空数组、其它 ok，避免真实后端 401 把登录态踢掉
const DEFAULT_HANDLER = (method) => (route) => {
  if (method === 'GET') return json(route, []);
  return json(route, { ok: true }, 200);
};

// 安装单一 /api/v1/** 拦截，返回 routes 对象；调用方向 routes 填 handler: (route, req) => {...}
async function installApiMock(page) {
  const routes = {};
  await page.route('**/api/v1/**', (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname.replace(/^\/api\/v1/, '');
    const h = routes[`${req.method().toUpperCase()} ${path}`] || routes[path];
    if (h) return h(route, req);
    return DEFAULT_HANDLER(req.method().toUpperCase())(route, req);
  });
  return routes;
}

const KB = { id: 'kb1', name: '默认知识库', description: '', embedding_model: 'BAAI/bge-large-zh-v1.5', doc_count: 0 };

// 让 app 在登录/带 token 后能加载默认库、文档、会话、历史，而不请求真实后端
function fillAppBoot(routes, opt = {}) {
  const kb = opt.kb || KB;
  routes['GET /knowledge'] = (r) => json(r, opt.kbs !== undefined ? opt.kbs : [kb]);
  routes['POST /knowledge'] = (r) => json(r, kb, 200);
  routes['GET /documents'] = (r) => json(r, opt.docs !== undefined ? opt.docs : []);
  routes['GET /chat/sessions'] = (r) => json(r, []);
  routes['GET /chat/history'] = (r) => json(r, { session_id: null, messages: [] });
}

// 一条 SSE 发言的伪响应（前端 fetch('/chat/stream') 后按行解析）
const SSE_ANSWER = [
  'data: {"type":"sources","session_id":"e2esess","data":[]}',
  '',
  'data: {"type":"delta","text":"这是模拟回答。"}',
  '',
  'data: {"type":"done","session_id":"e2esess","message_id":"m1","sources":[],"answer":"这是模拟回答。","cache_hit":false}',
  '',
  'data: [DONE]',
  '',
].join('\n');

module.exports = {
  json,
  installApiMock,
  fillAppBoot,
  KB,
  SSE_ANSWER,
  expect,
};

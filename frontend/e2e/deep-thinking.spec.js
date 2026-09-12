// 「深度思考」接到代理链路（票 37 / #45）：
// 按钮只在服务端允许时出现；勾了才发 deep；回答回来如实标注走没走代理。
// 注意 /health 不在 /api/v1 下，要单独 route。
const { test } = require('@playwright/test');
const { installApiMock, fillAppBoot, json, SSE_ANSWER, expect } = require('./helpers');

const BTN = '#deepBtn';

function sse(doneExtra) {
  const done = Object.assign({
    type: 'done', session_id: 's1', message_id: 'm1', sources: [],
    answer: '答案。', cache_hit: false,
  }, doneExtra);
  return [
    'data: {"type":"sources","session_id":"s1","data":[]}',
    '',
    'data: {"type":"delta","text":"答案。"}',
    '',
    'data: ' + JSON.stringify(done),
    '',
    'data: [DONE]',
    '',
  ].join('\n');
}

// 带登录态的启动：mock /health（决定按钮显隐）+ /api/v1 全套
async function boot(page, { agentEnabled = false, stream = SSE_ANSWER } = {}) {
  await page.route('**/health', (r) => json(r, {
    status: 'ok', app: 'test', use_real: false, agent_enabled: agentEnabled,
  }));
  const routes = await installApiMock(page);
  fillAppBoot(routes);
  routes['POST /chat/stream'] = (r) => r.fulfill({
    status: 200, body: stream, headers: { 'content-type': 'text/event-stream' },
  });
  return routes;
}

async function ask(page, question = '营收多少？') {
  await page.locator('#question').fill(question);
  await page.locator('#sendBtn').click();
}

test('用例54 服务端没开代理时，「深度思考」按钮不出现（不做点了没反应的入口）', async ({ page }) => {
  await boot(page, { agentEnabled: false });
  await page.goto('/');
  await expect(page.locator('#appCard')).toBeVisible();

  await expect(page.locator(BTN)).toBeHidden();
});

test('用例55 服务端开了代理时，按钮出现', async ({ page }) => {
  await boot(page, { agentEnabled: true });
  await page.goto('/');
  await expect(page.locator('#appCard')).toBeVisible();

  await expect(page.locator(BTN)).toBeVisible();
  await expect(page.locator(BTN)).toContainText('深度思考');
});

test('用例56 勾选后请求体带 deep:true；不勾选为 false', async ({ page }) => {
  const seen = [];
  const routes = await boot(page, { agentEnabled: true });
  routes['POST /chat/stream'] = (r) => {
    seen.push(r.request().postDataJSON().deep);
    return r.fulfill({ status: 200, body: SSE_ANSWER, headers: { 'content-type': 'text/event-stream' } });
  };
  await page.goto('/');
  await expect(page.locator('#appCard')).toBeVisible();

  await ask(page);
  await expect.poll(() => seen.length).toBe(1);
  expect(seen[0]).toBe(false);                       // 默认不发代理

  await page.locator(BTN).click();                   // 打开「深度思考」
  await ask(page);
  await expect.poll(() => seen.length).toBe(2);
  expect(seen[1]).toBe(true);
});

test('用例57 走了代理时，回答上明确标出「本次由代理链路作答」', async ({ page }) => {
  await boot(page, { agentEnabled: true, stream: sse({ agent: true }) });
  await page.goto('/');
  await expect(page.locator('#appCard')).toBeVisible();

  await page.locator(BTN).click();
  await ask(page);

  await expect(page.locator('.agent-ok')).toContainText('代理链路作答');
});

test('用例58 代理被降级时，标出原因而不是假装走了代理', async ({ page }) => {
  await boot(page, {
    agentEnabled: true,
    stream: sse({ agent: false, agent_skipped: '自带模型 m 不支持工具调用，本次降级为单步回答' }),
  });
  await page.goto('/');
  await expect(page.locator('#appCard')).toBeVisible();

  await page.locator(BTN).click();
  await ask(page);

  await expect(page.locator('.agent-note')).toContainText('降级');
  await expect(page.locator('.agent-ok')).toHaveCount(0);   // 没走成就不许标成走了
});

test('用例59 确定性链路不出现代理标注', async ({ page }) => {
  await boot(page, { agentEnabled: true, stream: sse({ agent: false }) });
  await page.goto('/');
  await expect(page.locator('#appCard')).toBeVisible();

  await ask(page);                                    // 没勾「深度思考」

  await expect(page.locator('.agent-ok')).toHaveCount(0);
  await expect(page.locator('.agent-note')).toHaveCount(0);
});

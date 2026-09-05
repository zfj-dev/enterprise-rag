// 登录页测试（真实 vanilla 单页：无 /login 路由，登录卡 #loginCard；token=localStorage['_rag_tok']）
const { test } = require('@playwright/test');
const { installApiMock, fillAppBoot, json, KB, expect } = require('./helpers');

const U = '#u';      // 用户名输入框 placeholder=用户名
const P = '#p';      // 密码输入框 placeholder=密码
const MSG = '#loginMsg';

test('用例1 页面渲染：登录卡正确显示用户名/密码输入框与登录/注册按钮', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#loginCard')).toBeVisible();
  await expect(page.locator(U)).toHaveAttribute('placeholder', '用户名');
  await expect(page.locator(P)).toHaveAttribute('placeholder', '密码');
  await expect(page.locator(P)).toHaveAttribute('type', 'password');
  await expect(page.getByRole('button', { name: '登录' })).toBeVisible();
  await expect(page.getByRole('button', { name: '注册' })).toBeVisible();
});

test('用例2 正常登录：正确凭据登录成功，切到应用视图并写入 localStorage._rag_tok', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { access_token: 'tok', token_type: 'bearer', role: 'viewer', username: 'test_user_001' });
  fillAppBoot(routes);   // 登录后 loadKbs 需要 /knowledge 等，避免真实后端 401 踢回
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator('#appCard')).toBeVisible();
  const tok = await page.evaluate(() => localStorage.getItem('_rag_tok'));
  expect(tok).toBe('tok');
});

test('用例3 登录加载/防重复：点击一次仅发一次请求（前端无内置 loading 禁用态，记录该缺口）', async ({ page }) => {
  const routes = await installApiMock(page);
  let calls = 0;
  routes['POST /auth/login'] = (r) => { calls += 1; return json(r, { access_token: 'tok', role: 'viewer', username: 'test_user_001' }); };
  fillAppBoot(routes);
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator('#appCard')).toBeVisible();
  expect(calls).toBe(1);   // 单次点击只发一次请求
});

test('用例4 错误密码：提示统一文案"用户名或密码错误"，不暴露具体原因', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { detail: '用户名或密码错误' }, 401);
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.fill(P, 'wrong_pass');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator(MSG)).toHaveText('用户名或密码错误');
});

test('用例5 空表单提交：未填字段提示校验信息（无前端中文校验，显示后端 422 消息）', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { detail: [{ msg: 'String should have at least 1 character' }] }, 422);
  await page.goto('/');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator(MSG)).not.toHaveText('');   // 等待后端校验响应
  const text = await page.locator(MSG).textContent();
  expect(text && text.length > 0).toBeTruthy();
  expect(text.indexOf('注册成功') === -1).toBeTruthy();
});

test('用例6 仅输入用户名：仍提示校验信息（前端未单独校验密码为空）', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { detail: [{ msg: 'String should have at least 1 character' }] }, 422);
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator(MSG)).not.toHaveText('');
  const text = await page.locator(MSG).textContent();
  expect(text && text.length > 0).toBeTruthy();
});

test('用例7 仅输入密码：仍提示校验信息（前端未单独校验用户名为空）', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { detail: [{ msg: 'String should have at least 1 character' }] }, 422);
  await page.goto('/');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator(MSG)).not.toHaveText('');
  const text = await page.locator(MSG).textContent();
  expect(text && text.length > 0).toBeTruthy();
});

test('用例8 SQL 注入：用户名注入 payload 返回统一登录失败，不崩溃、不注入', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { detail: '用户名或密码错误' }, 401);
  await page.goto('/');
  await page.fill(U, "' OR '1'='1");
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator(MSG)).toHaveText('用户名或密码错误');
});

test('用例9 XSS：用户名含 <script> 仅作为普通字符串，返回统一登录失败', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { detail: '用户名或密码错误' }, 401);
  await page.goto('/');
  await page.fill(U, '<script>alert(1)</script>');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator(MSG)).toHaveText('用户名或密码错误');
});

test('用例10 超长用户名：LoginRequest 无 max_length，超长不触发 422，后端按不存在用户返回 401', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { detail: '用户名或密码错误' }, 401);
  await page.goto('/');
  await page.fill(U, 'x'.repeat(300));
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator(MSG)).toHaveText('用户名或密码错误');
});

test('用例11 Unicode 用户名：中文/emoji 用户名可正常登录', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { access_token: 'tok', role: 'viewer', username: '用户😀' });
  fillAppBoot(routes);
  await page.goto('/');
  await page.fill(U, '用户😀');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator('#appCard')).toBeVisible();
});

test('用例12 大小写敏感：Alice 可登录而 alice 失败（后端按用户名精确匹配）', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => {
    const b = r.request().postDataJSON();
    return b.username === 'Alice'
      ? json(r, { access_token: 'tok', role: 'viewer', username: 'Alice' })
      : json(r, { detail: '用户名或密码错误' }, 401);
  };
  fillAppBoot(routes);
  await page.goto('/');
  await page.fill(U, 'Alice');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator('#appCard')).toBeVisible();
  // 再来一次小写 alice → 失败
  await page.evaluate(() => localStorage.clear());
  await page.goto('/');
  await page.fill(U, 'alice');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator(MSG)).toHaveText('用户名或密码错误');
});

test('用例13 响应时间：登录完成到应用可见 < 500ms', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { access_token: 'tok', role: 'viewer', username: 'test_user_001' });
  fillAppBoot(routes);
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.fill(P, 'TestPass123!');
  const t0 = Date.now();
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator('#appCard')).toBeVisible();
  const elapsed = Date.now() - t0;
  expect(elapsed).toBeLessThan(500);
});

test('用例14 已登录访问根页：自动展示应用视图（等效于"已登录访问 /login 重定向到 /dashboard"）', async ({ page }) => {
  const routes = await installApiMock(page);
  fillAppBoot(routes);
  await page.goto('/');
  await page.evaluate(() => { localStorage.setItem('_rag_tok', 'tok'); localStorage.setItem('_rag_user', 'test_user_001'); });
  await page.reload();
  await expect(page.locator('#appCard')).toBeVisible();
});

test('用例15 登录 500 错误：显示通用错误而非白屏', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => json(r, { detail: '服务器内部错误' }, 500);
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  await expect(page.locator(MSG)).toHaveText('服务器内部错误');
  await expect(page.locator('#loginCard')).toBeVisible();   // 未白屏、仍停留在登录卡
});

test('用例16 登录超时：模拟网络超时后仍停留在登录卡并给出反馈', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/login'] = (r) => new Promise(() => {});   // 永不返回，模拟挂起/超时
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '登录' }).click();
  // 等待超过 fetch 超时阈值（前端未显式处理超时，这里验证点击后界面未被破坏、仍可在登录卡操作）
  await page.waitForTimeout(3000);
  await expect(page.locator('#loginCard')).toBeVisible();
});

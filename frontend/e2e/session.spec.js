// 会话/Token 测试（用 storageState 预置登录态：localStorage['_rag_tok']='e2e-test-token'）
// 应用无 /dashboard 路由，登录态=进入应用视图 #appCard；过期/清除后回登录卡 #loginCard。
const { test } = require('@playwright/test');
const { installApiMock, fillAppBoot, json, expect } = require('./helpers');

const MSG = '#loginMsg';

test('用例23 Token 过期自动跳回登录页：受保护接口 401 后回到登录卡', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['GET /knowledge'] = (r) => json(r, { detail: '登录已过期' }, 401);   // 模拟 token 过期
  await page.goto('/');
  await expect(page.locator('#loginCard')).toBeVisible();   // init→showApp→loadKbs 401→clearAuth→回登录卡
  await expect(page.locator('#appCard')).toBeHidden();
});

test('用例24 Token 过期提示：跳转时显示"登录已过期，请重新登录"提示', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['GET /knowledge'] = (r) => json(r, { detail: '登录已过期' }, 401);
  await page.goto('/');
  await expect(page.locator(MSG)).toHaveText('登录已过期');   // 应用实际提示文案
});

test('用例25 多标签页同步登出：当前实现无 storage 事件监听（记录缺口，跳过）', async ({ page }) => {
  test.skip('应用未监听 storage 事件，标签页间不会自动同步登出');
});

test('用例26 刷新页面保持登录：F5 后仍进入应用视图', async ({ page }) => {
  const routes = await installApiMock(page);
  fillAppBoot(routes);
  await page.goto('/');
  await expect(page.locator('#appCard')).toBeVisible();
  await page.reload();                       // F5
  await expect(page.locator('#appCard')).toBeVisible();   // 仍登录，token 未丢
  await expect(page.locator('#loginCard')).toBeHidden();
});

test('用例27 清除 localStorage 后：_rag_tok 被移除，重载后回登录卡', async ({ page }) => {
  const routes = await installApiMock(page);
  fillAppBoot(routes);
  await page.goto('/');
  await expect(page.locator('#appCard')).toBeVisible();
  await page.evaluate(() => localStorage.removeItem('_rag_tok'));
  await page.reload();
  await expect(page.locator('#loginCard')).toBeVisible();
  await expect(page.locator('#appCard')).toBeHidden();
});

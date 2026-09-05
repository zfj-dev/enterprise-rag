// 登录态预置：把 token/用户名写进 localStorage，导出 storageState，供 "已登录" 场景（session.spec）复用。
// 避免每个测试都重新登录；token 本身是占位字符串，后续测试用 page.route 模拟 API，不依赖真实后端校验。
const { test } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

test('预置登录态 storageState', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => {
    localStorage.setItem('_rag_tok', 'e2e-test-token');   // 应用实际用的 token key 是 _rag_tok
    localStorage.setItem('_rag_user', 'test_user_001');
  });
  const dir = path.join('e2e', '.auth');
  fs.mkdirSync(dir, { recursive: true });                  // 目录可能不存在，先建
  await page.context().storageState({ path: path.join(dir, 'state.json') });
});

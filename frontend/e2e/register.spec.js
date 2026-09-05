// 注册页测试（真实 vanilla 单页：注册与登录共用同一张卡 #loginCard，只有 用户名/密码两个输入框；
// 注册只调 /auth/register，成功显示"注册成功，请登录"，不自动登录；无确认密码/角色/强度提示。）
const { test } = require('@playwright/test');
const { installApiMock, json, expect } = require('./helpers');

const U = '#u';
const P = '#p';
const MSG = '#loginMsg';

test('用例16 页面渲染：登录卡正确显示表单元素（用户名/密码/注册按钮）', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#loginCard')).toBeVisible();
  await expect(page.locator(U)).toHaveAttribute('placeholder', '用户名');
  await expect(page.locator(P)).toHaveAttribute('placeholder', '密码');
  await expect(page.getByRole('button', { name: '注册' })).toBeVisible();
});

test('用例17 正常注册：成功提示"注册成功，请登录"，且不自动登录', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/register'] = (r) => json(r, { access_token: 'tok', token_type: 'bearer', role: 'viewer' });
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '注册' }).click();
  await expect(page.locator(MSG)).toHaveText('注册成功，请登录');
  await expect(page.locator('#appCard')).toBeHidden();   // 注册后不自动登录，仍停留登录卡
});

test('用例18 密码强度/过短：弱密码（<6位）由后端 422 拒绝并提示（前端无强度提示控件）', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/register'] = (r) => json(r, { detail: [{ msg: 'String should have at least 6 characters' }] }, 422);
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.fill(P, '12345');   // 5 位 < 6
  await page.getByRole('button', { name: '注册' }).click();
  await expect(page.locator(MSG)).not.toHaveText('');
  const text = await page.locator(MSG).textContent();
  expect(text && text.length > 0).toBeTruthy();
});

test('用例19 密码不匹配：注册页无确认密码字段（记录缺口，跳过）', async ({ page }) => {
  test.skip('注册页只有 用户名/密码 两个输入框，无确认密码字段，无法测"两次密码不一致"');
});

test('用例20 重复用户名：提示"用户名已存在"', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/register'] = (r) => json(r, { detail: '用户名已存在' }, 400);
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '注册' }).click();
  await expect(page.locator(MSG)).toHaveText('用户名已存在');
});

test('用例21 角色选择：登录卡无角色选择控件，注册恒为 viewer', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/register'] = (r) => json(r, { access_token: 'tok', token_type: 'bearer', role: 'viewer' });
  await page.goto('/');
  // 登录卡上确认没有角色选择（select / radio）
  const roleControls = page.locator('#loginCard select, #loginCard input[type=radio]');
  await expect(roleControls).toHaveCount(0);
  await page.fill(U, 'test_user_001');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '注册' }).click();
  await expect(page.locator(MSG)).toHaveText('注册成功，请登录');
});

test('用例22 注册后不自动登录：需单独登录（应用视图未展示）', async ({ page }) => {
  const routes = await installApiMock(page);
  routes['POST /auth/register'] = (r) => json(r, { access_token: 'tok', role: 'viewer' });
  await page.goto('/');
  await page.fill(U, 'test_user_001');
  await page.fill(P, 'TestPass123!');
  await page.getByRole('button', { name: '注册' }).click();
  await expect(page.locator(MSG)).toHaveText('注册成功，请登录');
  await expect(page.locator('#appCard')).toBeHidden();   // 未见应用视图
});

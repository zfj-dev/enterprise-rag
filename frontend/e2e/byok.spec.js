// BYOK（自带 Key）界面测试：填 / 轮换 / 删除入口 + 状态回显 + 余额两维度。
// 全程 mock /llm/** 与启动接口，后端仍由 webServer 真实托管 index.html。
const { test } = require('@playwright/test');
const { installApiMock, fillAppBoot, json, expect } = require('./helpers');

const NAV = '.side-nav .nav-item:has-text("自带模型")';
const MODAL = '#byokModal';

// 启动所需的最小接口 + BYOK 三件套；返回 routes 供各用例改写
async function boot(page, opt = {}) {
  const cfg = opt.config || unconfigured();
  const routes = await installApiMock(page);
  fillAppBoot(routes);
  routes['GET /llm/config'] = (r) => json(r, cfg);
  routes['GET /llm/balance'] = (r) => json(r, opt.balance || BALANCE);
  routes['GET /llm/capability'] = (r) => json(r, opt.cap || (cfg.configured ? capUnknown() : capUnconfigured()));
  return routes;
}

// 后端 CapabilityOut 的三种典型形态
const capUnconfigured = () => ({ configured: false, checked: false, reachable: null,
  supports_tools: null, source: '', note: '未配置自带模型 —— 走服务端全局模型，无需探测' });
const capUnknown = () => ({ configured: true, checked: false, reachable: null,
  supports_tools: null, source: '', note: '暂无结论 —— 点「测试连通性」探一次' });
const capProbedOk = () => ({ configured: true, checked: true, reachable: true,
  supports_tools: true, source: 'probed', note: '探到 provider 接受了带 tools 的请求' });

// 后端 LLMConfigOut 的两种典型形态
const unconfigured = () => ({
  configured: false, base_url: '', model: '', key_tail: '', updated_at: '', persistent: false,
});
const configured = (over = {}) => ({
  configured: true, base_url: 'https://api.deepseek.com/v1', model: 'deepseek-chat',
  key_tail: 'a1b2', updated_at: '2026-09-12T10:00:00', persistent: true, ...over,
});

const BALANCE = {
  vendor: { available: true, amount: 42.5, currency: 'CNY', note: '厂商接口返回', alert: '' },
  ours: { total_cost: 0.36, note: '我方统计用量', total_tokens: 1200 },
};

// 打开面板：点侧栏入口，等输入框出现
async function openPanel(page) {
  await page.locator(NAV).click();
  await expect(page.locator(MODAL)).toBeVisible();
  await expect(page.locator('#byokBaseUrl')).toBeVisible();
}

test('用例29 侧栏有「自带模型」入口，点击打开面板并显示三个输入框', async ({ page }) => {
  await boot(page);
  await page.goto('/');
  await expect(page.locator('#appCard')).toBeVisible();

  await expect(page.locator(NAV)).toBeVisible();
  await openPanel(page);

  await expect(page.locator('#byokBaseUrl')).toBeVisible();
  await expect(page.locator('#byokKey')).toBeVisible();
  await expect(page.locator('#byokModel')).toBeVisible();
  await expect(page.locator('#byokSave')).toBeVisible();
});

test('用例30 Key 输入框是 password 类型（明文不回显在屏幕上）', async ({ page }) => {
  await boot(page);
  await page.goto('/');
  await openPanel(page);
  await expect(page.locator('#byokKey')).toHaveAttribute('type', 'password');
});

test('用例31 未配置状态：提示未配置、隐藏删除按钮、key 输入框留空', async ({ page }) => {
  await boot(page, { config: unconfigured() });
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokStatus')).toContainText('未配置');
  await expect(page.locator('#byokDelete')).toBeHidden();
  await expect(page.locator('#byokBaseUrl')).toHaveValue('');
  await expect(page.locator('#byokModel')).toHaveValue('');
});

test('用例32 已配置状态：回显 base_url / model / key 尾号，且**不回显明文 key**', async ({ page }) => {
  await boot(page, { config: configured() });
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokStatus')).toContainText('https://api.deepseek.com/v1');
  await expect(page.locator('#byokStatus')).toContainText('deepseek-chat');
  await expect(page.locator('#byokStatus')).toContainText('a1b2');       // 尾号
  await expect(page.locator('#byokKey')).toHaveValue('');                // 明文永不回填
  await expect(page.locator('#byokDelete')).toBeVisible();
});

test('用例33 persistent=false 时如实提示「仅存内存、重启失效」', async ({ page }) => {
  await boot(page, { config: configured({ persistent: false }) });
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokStatus')).toContainText('重启');
});

test('用例34 persistent=true 时不出现「重启失效」警告', async ({ page }) => {
  await boot(page, { config: configured({ persistent: true }) });
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokStatus')).not.toContainText('重启');
});

test('用例35 保存：PUT /llm/config，body 是 {base_url, key, model}', async ({ page }) => {
  const routes = await boot(page, { config: unconfigured() });
  let sent = null;
  routes['PUT /llm/config'] = (r) => {
    sent = r.request().postDataJSON();
    return json(r, configured({ model: sent.model }));
  };
  await page.goto('/');
  await openPanel(page);

  await page.locator('#byokBaseUrl').fill('https://api.deepseek.com/v1');
  await page.locator('#byokKey').fill('sk-secret-xyz');
  await page.locator('#byokModel').fill('deepseek-chat');
  await page.locator('#byokSave').click();

  await expect.poll(() => sent && sent.model).toBe('deepseek-chat');
  expect(sent.base_url).toBe('https://api.deepseek.com/v1');
  expect(sent.key).toBe('sk-secret-xyz');

  // 保存后清空 key 输入框，并把明文 key 留在页面/localStorage 之外
  await expect(page.locator('#byokKey')).toHaveValue('');
  const dumped = await page.evaluate(() => JSON.stringify(localStorage));
  expect(dumped).not.toContain('sk-secret-xyz');
});

test('用例36 保存被 SSRF 防护拒绝（400）时，原样显示后端给出的原因', async ({ page }) => {
  const routes = await boot(page);
  routes['PUT /llm/config'] = (r) => json(r, { detail: '地址指向私网，已拒绝' }, 400);
  await page.goto('/');
  await openPanel(page);

  await page.locator('#byokBaseUrl').fill('http://192.168.1.10/v1');
  await page.locator('#byokKey').fill('sk-x');
  await page.locator('#byokModel').fill('m');
  await page.locator('#byokSave').click();

  await expect(page.locator('#byokMsg')).toContainText('地址指向私网，已拒绝');
});

test('用例37 删除：DELETE /llm/config，删完回到未配置状态', async ({ page }) => {
  const routes = await boot(page, { config: configured() });
  let deleted = false;
  routes['DELETE /llm/config'] = (r) => { deleted = true; return json(r, { ok: true }); };
  routes['GET /llm/config'] = (r) => json(r, deleted ? unconfigured() : configured());
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokDelete')).toBeVisible();
  await page.locator('#byokDelete').click();

  await expect.poll(() => deleted).toBe(true);
  await expect(page.locator('#byokStatus')).toContainText('未配置');
  await expect(page.locator('#byokDelete')).toBeHidden();
});

test('用例38 面板内显示余额两个维度：厂商余额 + 我方统计用量（不拿用量冒充余额）', async ({ page }) => {
  await boot(page, { config: configured(), balance: BALANCE });
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokBal')).toContainText('42.5');
  await expect(page.locator('#byokBal')).toContainText('0.36');
  await expect(page.locator('#byokBal')).toContainText('我方统计用量');   // 两个维度分开写清楚
});

test('用例39 余额查不到时照实写原因，不留空、不用 0 顶替', async ({ page }) => {
  await boot(page, {
    config: configured(),
    balance: {
      vendor: { available: false, amount: null, currency: '', note: '该厂商不支持余额查询', alert: '' },
      ours: { total_cost: null, note: '单价未知' },
    },
  });
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokBal')).toContainText('该厂商不支持余额查询');
});

test('用例40 点关闭后遮罩消失（面板可退出）', async ({ page }) => {
  await boot(page);
  await page.goto('/');
  await openPanel(page);

  await page.locator('#byokClose').click();
  await expect(page.locator(MODAL)).toHaveCount(0);
});

test('用例41 打开面板时重新拉取配置与余额（不缓存上次的旧值）', async ({ page }) => {
  let cfg = unconfigured();
  const routes = await boot(page, { config: unconfigured() });
  routes['GET /llm/config'] = (r) => json(r, cfg);
  await page.goto('/');
  await openPanel(page);
  await expect(page.locator('#byokStatus')).toContainText('未配置');

  cfg = configured();                       // 面板已开时外部改了配置
  await page.locator('#byokClose').click();
  await openPanel(page);
  await expect(page.locator('#byokStatus')).toContainText('https://api.deepseek.com/v1');
});

test('用例42 一个知识库都没有时也能打开面板（BYOK 与知识库无关）', async ({ page }) => {
  const routes = await boot(page);
  routes['GET /knowledge'] = (r) => json(r, []);      // 后端一个库都没有
  await page.goto('/');
  await expect(page.locator('#appCard')).toBeVisible();
  await openPanel(page);
});

test('用例44 拉配置失败时不谎报「未配置」，而是明说取不到', async ({ page }) => {
  const routes = await boot(page);
  routes['GET /llm/config'] = (r) => json(r, { detail: '登录已过期' }, 401);
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokStatus')).toContainText('取配置失败');
  await expect(page.locator('#byokStatus')).toContainText('登录已过期');
  // 关键是**不能**把「不知道」渲染成「已确认未配置」
  await expect(page.locator('#byokStatus')).not.toContainText('● 未配置');
});

test('用例45 余额取不到时明说没取到（不留空、也不写 0）', async ({ page }) => {
  const routes = await boot(page);
  routes['GET /llm/balance'] = (r) => json(r, { detail: '上游超时' }, 502);
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokBal')).toContainText('余额没取到');
});

test('用例46 双击保存只发一次 PUT（in-flight 守卫）', async ({ page }) => {
  const routes = await boot(page);
  let puts = 0;
  routes['PUT /llm/config'] = (r) => { puts += 1; return json(r, configured()); };
  await page.goto('/');
  await openPanel(page);

  await page.locator('#byokBaseUrl').fill('https://api.deepseek.com/v1');
  await page.locator('#byokKey').fill('sk-secret-xyz');
  await page.locator('#byokModel').fill('deepseek-chat');
  await page.locator('#byokSave').click();
  await page.locator('#byokSave').click({ force: true });      // 第二次点击必须被挡住
  await expect.poll(() => puts).toBe(1);
  await page.waitForTimeout(400);
  expect(puts).toBe(1);
});

test('用例43 保存后，面板内的余额块也要刷新（不再显示旧的「未配置」口径）', async ({ page }) => {
  const routes = await boot(page);
  // 保存前：没配自带模型，厂商余额无从谈起
  let saved = false;
  routes['GET /llm/balance'] = (r) => json(r, saved
    ? BALANCE
    : { vendor: { available: false, amount: null, currency: '', note: '未配置自带模型', alert: '' },
        ours: { total_cost: 0.36, note: '我方统计用量' } });
  routes['PUT /llm/config'] = (r) => { saved = true; return json(r, configured()); };

  await page.goto('/');
  await openPanel(page);
  await expect(page.locator('#byokBal')).toContainText('未配置自带模型');

  await page.locator('#byokBaseUrl').fill('https://api.deepseek.com/v1');
  await page.locator('#byokKey').fill('sk-secret-xyz');
  await page.locator('#byokModel').fill('deepseek-chat');
  await page.locator('#byokSave').click();

  await expect(page.locator('#byokBal')).toContainText('42.5');           // 已切换到真实厂商余额
  await expect(page.locator('#byokBal')).not.toContainText('未配置自带模型');
});

test('用例47 未配置时能力块说明「未配置」，且不给测试按钮', async ({ page }) => {
  await boot(page, { config: unconfigured() });
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokCap')).toContainText('未配置自带模型');
  await expect(page.locator('#byokTest')).toHaveCount(0);
});

test('用例48 没有结论时显示「? 未探到 / ? 未知」—— 不谎报「不通 / 不支持」', async ({ page }) => {
  await boot(page, { config: configured(), cap: capUnknown() });
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokCap')).toContainText('未探到');
  await expect(page.locator('#byokCap')).toContainText('未知');
  await expect(page.locator('#byokCap')).not.toContainText('不支持');
  await expect(page.locator('#byokTest')).toBeVisible();
});

test('用例49 点「测试连通性」触发 POST 探测，并把结论显示出来', async ({ page }) => {
  const routes = await boot(page, { config: configured(), cap: capUnknown() });
  let probed = 0;
  routes['POST /llm/capability/probe'] = (r) => { probed += 1; return json(r, capProbedOk()); };
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokCap')).toContainText('未探到');
  await page.locator('#byokTest').click();

  await expect.poll(() => probed).toBe(1);
  await expect(page.locator('#byokCap')).toContainText('✓ 支持');
  await expect(page.locator('#byokCap')).toContainText('✓ 通');
});

test('用例50 探测失败时如实说原因，仍**不**显示「不支持」', async ({ page }) => {
  const routes = await boot(page, { config: configured(), cap: capUnknown() });
  routes['POST /llm/capability/probe'] = (r) => json(r, {
    configured: true, checked: true, reachable: null, supports_tools: null, source: '',
    note: '连不上（ConnectError）—— 请检查地址与网络',
  });
  await page.goto('/');
  await openPanel(page);

  await page.locator('#byokTest').click();

  await expect(page.locator('#byokCap')).toContainText('ConnectError');   // 原因要透出来
  await expect(page.locator('#byokCap')).toContainText('未知');
  await expect(page.locator('#byokCap')).not.toContainText('不支持');
});

test('用例51 保存后旧的能力结论作废，重新拉一次', async ({ page }) => {
  const routes = await boot(page, { config: unconfigured(), cap: capUnconfigured() });
  let capCalls = 0;
  routes['GET /llm/capability'] = (r) => {
    capCalls += 1;
    return json(r, capCalls > 1 ? capUnknown() : capUnconfigured());
  };
  routes['PUT /llm/config'] = (r) => json(r, configured());
  await page.goto('/');
  await openPanel(page);
  await expect(page.locator('#byokCap')).toContainText('未配置自带模型');

  await page.locator('#byokBaseUrl').fill('https://api.deepseek.com/v1');
  await page.locator('#byokKey').fill('sk-secret-xyz');
  await page.locator('#byokModel').fill('deepseek-chat');
  await page.locator('#byokSave').click();

  await expect.poll(() => capCalls).toBeGreaterThan(1);       // 配置变了 → 能力重拉
  await expect(page.locator('#byokCap')).toContainText('未探到');
});

test('用例52 打开面板不发探测请求（只看缓存）', async ({ page }) => {
  const routes = await boot(page, { config: configured(), cap: capUnknown() });
  let probed = 0;
  routes['POST /llm/capability/probe'] = (r) => { probed += 1; return json(r, capProbedOk()); };
  await page.goto('/');
  await openPanel(page);

  await expect(page.locator('#byokCap')).toBeVisible();
  await page.waitForTimeout(300);
  expect(probed).toBe(0);                                     // 打开面板不该打一次外网
});

test('用例53 「无法据此确认」时显示「? 未知（保守默认）」，**不**显示「✗ 不支持」', async ({ page }) => {
  // 后端被 4xx 拒时的形态：supports_tools=false 是**内部保守默认**，不是探测结论。
  // 照它写「✗ 不支持」会和同一块里的 note 自相矛盾 —— 那正是本仓库禁止的「拿不知道当结论」。
  const routes = await boot(page, { config: configured(), cap: capUnknown() });
  routes['POST /llm/capability/probe'] = (r) => json(r, {
    configured: true, checked: true, reachable: true, supports_tools: false,
    source: 'conservative',
    note: '探测被拒（HTTP 400）—— 无法据此确认工具支持，按保守默认处理',
  });
  await page.goto('/');
  await openPanel(page);
  await page.locator('#byokTest').click();

  await expect(page.locator('#byokCap')).toContainText('未知（保守默认）');
  await expect(page.locator('#byokCap')).toContainText('✓ 通');       // 对端答话了，连通性为真
  await expect(page.locator('#byokCap')).not.toContainText('✗ 不支持');
});

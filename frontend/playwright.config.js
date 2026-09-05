// Playwright E2E 配置 —— 针对真实前端（vanilla 单文件 index.html，FastAPI 在 8000 托管）
// 说明：应用无 /login /register /dashboard 路由，认证靠 #loginCard / #appCard 显隐；
// token 存 localStorage['_rag_tok']。API 用 page.route 拦截 /api/v1/** 模拟，保证确定性。
const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,       // 依赖同一个后端服务，串行更稳
  retries: 0,
  reporter: [
    ['list'],
    ['html', { open: 'never', outputFolder: 'playwright-report' }],
  ],
  use: {
    baseURL: 'http://127.0.0.1:8000',
    headless: true,
    viewport: { width: 1280, height: 800 },
    locale: 'zh-CN',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    // 先预置登录 storageState（auth.setup.js 写入 e2e/.auth/state.json）
    { name: 'setup', testMatch: /auth\.setup\.js/ },
    // 未登录场景：登录页 / 注册页 / 完整流程
    {
      name: 'app',
      testMatch: /login\.spec\.js|register\.spec\.js|auth-flow\.spec\.js/,
      dependencies: ['setup'],
      use: { ...devices['Desktop Chrome'] },
    },
    // 已登录场景：会话 / token（用 setup 产出的 storageState 作为起始登录态）
    {
      name: 'auth',
      testMatch: /session\.spec\.js/,
      dependencies: ['setup'],
      use: {
        ...devices['Desktop Chrome'],
        storageState: 'e2e/.auth/state.json',
      },
    },
  ],
  // FastAPI（demo 模式）托管前端 index.html + /api/v1；用真实后端 serve 页面，测试中用 route 模拟 API
  webServer: {
    command: 'cd ..\\backend && .venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000',
    url: 'http://127.0.0.1:8000/health',
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});

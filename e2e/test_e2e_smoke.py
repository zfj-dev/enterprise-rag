# -*- coding: utf-8 -*-
"""Playwright 端到端冒烟：登录 -> 提问 -> 断言有 AI 回复（用于 make test-e2e / 回归防线）。
需要先启动服务：http://localhost:8000
"""
import sys
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8000"


def main() -> int:
    errs = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_context(viewport={"width": 1280, "height": 800}).new_page()
        page.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        page.goto(BASE)
        page.wait_for_selector("#u", timeout=10000)
        page.fill("#u", "admin")
        page.fill("#p", "admin123")
        page.click("button:has-text('登录')")
        page.wait_for_selector("#question", timeout=10000)
        page.fill("#question", "端到端冒烟测试")
        page.click("#sendBtn")
        # 等回复出现（模拟回答 或 任何 AI 气泡内容）
        page.wait_for_selector(".msg .ai-body", timeout=20000)
        body = page.inner_text(".msg .ai-body")
        browser.close()
    if not body or len(body.strip()) < 2:
        print("E2E FAIL: AI 回复为空")
        return 1
    if errs:
        print("E2E 检测到 JS 错误:")
        for e in errs[:5]:
            print("  -", e)
        return 1
    print("E2E OK: 回复长度 =", len(body.strip()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

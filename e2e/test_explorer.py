# -*- coding: utf-8 -*-
"""Step1 探索性Fuzz测试：Playwright 15轮随机操作，每轮截图，检查白屏/JS报错/来源数异常。"""
import os, time, random, sys, json
from playwright.sync_api import sync_playwright

BASE = "http://localhost:8000"
HERE = os.path.dirname(os.path.abspath(__file__))
SHOT = os.path.join(HERE, "screenshots")
os.makedirs(SHOT, exist_ok=True)
ANOM = []          # 每项: {step, round, desc}
CONSOLE_ERRS = []  # JS 错误/console.error


def anom(roundno, step, desc):
    ANOM.append({"round": roundno, "step": step, "desc": desc})
    print(f"  [异常] r{roundno} {step}: {desc}")


def shot(page, roundno, name):
    path = os.path.join(SHOT, f"{roundno:02d}_{name}.png")
    try:
        page.screenshot(path=path, full_page=False)
    except Exception as e:
        print(f"  [shot err] {e}")
    return path


def white_check(page, roundno, step):
    try:
        txt = page.evaluate("document.body ? (document.body.innerText||'') : ''")
        if len(txt.strip()) < 5:
            anom(roundno, step, f"疑似白屏 body文本长度={len(txt.strip())}")
    except Exception as e:
        anom(roundno, step, f"white-check异常 {e}")


def login(page):
    page.goto(BASE)
    page.wait_for_selector("#u", timeout=10000)
    page.fill("#u", "admin")
    page.fill("#p", "admin123")
    page.click("button:has-text('登录')")
    try:
        page.wait_for_selector("#question", timeout=10000)
    except Exception:
        return False
    return True


def src_count(page):
    """统计当前渲染出的引用来源标签数。"""
    try:
        return page.evaluate("document.querySelectorAll('.ref-tag').length")
    except Exception:
        return -1


def ensure_ask(page, q):
    """清空后提问，等待流式结束（done -> isStreaming false / 按钮隐藏）。"""
    try:
        page.fill("#question", q)
        page.click("#sendBtn")
        # 等流式结束：停止按钮消失
        page.wait_for_selector("#stopBtn", state="detached", timeout=8000)
    except Exception:
        pass
    page.wait_for_timeout(200)


def reset_convo(page):
    """回到一个干净的对话。"""
    try:
        page.click("button[onclick='newChat()']")
        page.wait_for_timeout(150)
    except Exception:
        pass


def main():
    random.seed(7)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()
        page.on("pageerror", lambda e: CONSOLE_ERRS.append(f"pageerror: {e}"))
        page.on("console", lambda m: CONSOLE_ERRS.append(f"console.{m.type}: {m.text}") if m.type in ("error", "warning") else None)

        if not login(page):
            print("登录失败"); browser.close(); return
        print("已登录。开始 15 轮 fuzz...")

        inputs = ["", "A"*10000, "'; DROP TABLE users; --", "<script>alert(1)</script><b>x</b>", "🎉🚀测试中文🤖", "a' OR '1'='1", "%00%27%22%3C%3E&|;$"]
        for rnd in range(1, 16):
            print(f"\n===== 第 {rnd} 轮 =====")
            reset_convo(page)
            op = random.choice(["input", "file", "rapid", "switch", "net"])
            step = f"op={op}"
            try:
                if op == "input":
                    q = random.choice(inputs)
                    shot(page, rnd, "before_input")
                    page.fill("#question", q)
                    if q.strip():
                        page.click("#sendBtn")
                    page.wait_for_timeout(1500)
                    white_check(page, rnd, "input")
                    sc = src_count(page)
                    if sc > 20:
                        anom(rnd, "input", f"引用来源数量异常偏多={sc}")
                    shot(page, rnd, "after_input")
                elif op == "file":
                    # 造一个异常文件
                    ftype = random.choice(["exe", "js", "big", "pdf"])
                    if ftype == "big":
                        fp = os.path.join(HERE, "big_tmp.bin")
                        with open(fp, "wb") as f:
                            f.write(b"\x00" * (51 * 1024 * 1024))  # >50MB
                    else:
                        ext = {"exe": "exe", "js": "js", "pdf": "pdf"}[ftype]
                        fp = os.path.join(HERE, f"tmp_{ftype}.{ext}")
                        with open(fp, "w", encoding="utf-8") as f:
                            f.write("MZ fake exe" if ftype == "exe" else ("data" if ftype == "js" else "this is not a real pdf"))
                    shot(page, rnd, "before_file")
                    page.set_input_files("#fileIn", fp)
                    page.wait_for_timeout(2000)
                    white_check(page, rnd, "file")
                    shot(page, rnd, "after_file")
                elif op == "rapid":
                    # 连发 10 次（每次填+点，测试同对话流式中重复提问守卫）
                    for _ in range(10):
                        try:
                            page.fill("#question", "快速连发测试%d" % _)
                            page.click("#sendBtn", timeout=600)
                        except Exception:
                            break
                    page.wait_for_timeout(1500)
                    white_check(page, rnd, "rapid")
                    sc = src_count(page)
                    if sc > 20:
                        anom(rnd, "rapid", f"来源数异常={sc}")
                    shot(page, rnd, "after_rapid")
                elif op == "switch":
                    page.fill("#question", "切对话打断测试")
                    page.click("#sendBtn")
                    page.wait_for_timeout(300)  # 正在流式
                    # 新建另一个对话（打断当前生成）
                    page.click("button[onclick='newChat()']")
                    page.wait_for_timeout(1200)
                    white_check(page, rnd, "switch")
                    shot(page, rnd, "after_switch")
                elif op == "net":
                    page.fill("#question", "断网测试")
                    page.click("#sendBtn")
                    page.wait_for_timeout(300)  # 流式中
                    ctx.set_offline(True)
                    page.wait_for_timeout(800)
                    ctx.set_offline(False)
                    page.wait_for_timeout(1200)
                    white_check(page, rnd, "net")
                    shot(page, rnd, "after_net")
            except Exception as e:
                anom(rnd, step, f"执行异常: {type(e).__name__}: {e}")

        browser.close()

    # 汇总
    print("\n========== FUZZ 汇总 ==========")
    print(f"JS/console 错误 {len(CONSOLE_ERRS)} 条:")
    for e in CONSOLE_ERRS[:30]:
        print("   -", e[:180])
    print(f"异常 {len(ANOM)} 条:")
    for a in ANOM:
        print(f"   r{a['round']} {a['step']}: {a['desc']}")
    out = {"anomalies": ANOM, "console_errors": CONSOLE_ERRS}
    with open(os.path.join(HERE, "fuzz_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("报告: e2e/fuzz_report.json")


if __name__ == "__main__":
    main()

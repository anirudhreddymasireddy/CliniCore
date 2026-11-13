"""End-to-end test of replay mode on live data (ui/start_live.sh must be running).

Checks: live backend, ward A/B load, slider, Play / Pause / Restart, notifications appear for alerts in the
6 h window (MEDIUM / CRITICAL only), one chime when Play steps onto a new CRITICAL alert, Open, monitor
data, patient actions, Call doctor returning to the ward, no horizontal overflow at 1920x1080 and 1440x900.

Usage: uv run --with playwright python ../tests/e2e_replay.py <screenshot-dir>
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5173"
SHOTS = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f" – {detail}" if detail else ""))


def api(path):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read())


def set_hour(page, h):
    page.evaluate("""h => { const el = document.querySelector('[data-testid=slider]');
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, String(h));
        el.dispatchEvent(new Event('input', { bubbles: true })); }""", h)
    page.wait_for_selector(f"[data-testid=clock]:has-text('ICU hour {h}')")


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    health = api("/api/health")
    check("backend in live mode as MCP_READONLY", health["data_mode"] == "live" and health.get("db_user") == "MCP_READONLY", str(health))
    ward = api("/api/ward/A")
    crit_hours = sorted({int(h) for b in ward["beds"] for h, r in b["hours"].items()
                         if r["alert_fired"] == 1 and r["alert_level"] == "CRITICAL" and int(h) >= 2})
    hc = crit_hours[0]
    expected_24 = sum(1 for b in ward["beds"] for h, r in b["hours"].items()
                      if 19 <= int(h) <= 24 and r["alert_fired"] == 1 and r["alert_level"] in ("MEDIUM", "CRITICAL"))
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        for size in ((1920, 1080), (1440, 900)):
            tag = f"{size[0]}x{size[1]}"
            page = browser.new_page(viewport={"width": size[0], "height": size[1]})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(BASE + "/#/replay")
            page.wait_for_selector("[data-testid=ward-grid]")
            check(f"[{tag}] ward A: 12 beds, hour 0", page.locator("[data-testid^=bed-]").count() == 12
                  and "ICU hour 0" in page.locator("[data-testid=clock]").inner_text())
            set_hour(page, 24)
            n = page.locator("[data-testid=notification]").count()
            check(f"[{tag}] slider to hour 24: notifications for alerts of the last 6 h", n == expected_24, f"{n} vs {expected_24}")
            levels = {page.locator("[data-testid=notification]").nth(i).get_attribute("data-level") for i in range(n)}
            check(f"[{tag}] only MEDIUM / CRITICAL notify", levels <= {"MEDIUM", "CRITICAL"}, str(levels))
            page.screenshot(path=str(SHOTS / f"replay-{tag}-ward.png"))
            # Play onto a new CRITICAL alert -> one chime
            set_hour(page, hc - 1)
            before = page.evaluate("window.__cpChimes || 0")
            page.get_by_test_id("play").click()
            page.wait_for_selector(f"[data-testid=clock]:has-text('ICU hour {hc}')", timeout=5000)
            time.sleep(0.3)
            page.get_by_test_id("play").click()  # pause
            after = page.evaluate("window.__cpChimes || 0")
            check(f"[{tag}] Play steps one hour every 2 s and chimes once for a new CRITICAL (hour {hc})", after == before + 1, f"{before}->{after}")
            clock = page.locator("[data-testid=clock]").inner_text()
            time.sleep(2.5)
            check(f"[{tag}] Pause stops the replay", page.locator("[data-testid=clock]").inner_text() == clock)
            # open a CRITICAL notification
            card = page.locator("[data-testid=notification][data-level=CRITICAL]").first
            card.get_by_role("button", name="Open").click()
            page.wait_for_selector("[data-testid=patient-dashboard]", timeout=15000)
            mon = page.locator("[data-testid=monitor]").inner_text()
            check(f"[{tag}] Open shows patient with raw monitor vitals", "Hours before now" in mon and ("5-minute" in mon or "charted" in mon), mon[-140:].replace("\n", " "))
            check(f"[{tag}] patient view: drivers, risk chart, lab, history, footer",
                  all(page.locator(f"[data-testid={t}]").count() for t in ("drivers", "risk-chart", "lab-panel", "history-panel", "footer")))
            ov = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            check(f"[{tag}] no horizontal overflow", ov <= 0, f"{ov}px")
            page.screenshot(path=str(SHOTS / f"replay-{tag}-patient.png"), full_page=True)
            page.get_by_role("button", name="✓ Acknowledge").click()
            page.wait_for_selector("[data-testid=toast]:has-text('Acknowledged – logged')")
            page.get_by_role("button", name="☎ Call doctor").click()
            page.wait_for_selector("[data-testid=toast]:has-text('Doctor notified – logged')")
            page.wait_for_selector("[data-testid=ward-grid]", timeout=4000)
            check(f"[{tag}] Call doctor confirms and returns to the ward", page.locator("[data-testid=patient-dashboard]").count() == 0)
            page.get_by_test_id("restart").click()
            check(f"[{tag}] Restart returns to hour 0", "ICU hour 0" in page.locator("[data-testid=clock]").inner_text())
            page.get_by_test_id("ward-B").click()
            page.wait_for_selector("[data-testid=ward-grid]")
            time.sleep(1)
            check(f"[{tag}] ward B (MIMIC) loads 12 beds", page.locator("[data-testid^=bed-]").count() == 12)
            check(f"[{tag}] no JavaScript errors", not errors, "; ".join(errors)[:300])
            page.close()
        browser.close()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

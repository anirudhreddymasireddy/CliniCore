"""End-to-end test of presentation mode in headless Chromium (Playwright).

Requires the app running in demo mode (ui/start_demo.sh). Tests at 1920x1080 (projector) and 1440x900
(13-inch laptop): keys N / M / R / F, chime count, notification content and order, Open, Acknowledge,
Snooze, all patient actions, Call doctor returning to the ward, no overlap of notifications with vital-sign
curves, no horizontal overflow. Screenshots go to the directory given as argv[1].

Usage: uv run --with playwright python ../tests/e2e_presentation.py <screenshot-dir>
"""
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:5173/#/present"
SHOTS = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f" – {detail}" if detail else ""))


def boxes_overlap(a, b):
    return not (a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"]
                or a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"])


def run(page, size):
    tag = f"{size[0]}x{size[1]}"
    page.set_viewport_size({"width": size[0], "height": size[1]})
    page.goto(URL)
    page.wait_for_selector("[data-testid=ward-grid]")
    tiles = page.locator("[data-testid^=bed-]")
    tiers = [tiles.nth(i).get_attribute("data-tier") for i in range(tiles.count())]
    check(f"[{tag}] calm ward of 8-10 beds, all NORMAL at start", 8 <= len(tiers) <= 10 and set(tiers) == {"NORMAL"}, str(tiers))
    check(f"[{tag}] no notifications at start", page.locator("[data-testid=notification]").count() == 0)
    page.screenshot(path=str(SHOTS / f"{tag}-1-ward.png"))

    # N: CRITICAL after a 2 s delay, with one chime
    page.keyboard.press("n")
    time.sleep(1.0)
    check(f"[{tag}] N waits ~2 s before notifying", page.locator("[data-testid=notification]").count() == 0)
    page.wait_for_selector("[data-testid=notification][data-level=CRITICAL]", timeout=4000)
    chimes = page.evaluate("window.__cpChimes || 0")
    check(f"[{tag}] CRITICAL notification slides in with one chime", chimes == 1, f"chimes={chimes}")
    card = page.locator("[data-testid=notification][data-level=CRITICAL]")
    text = card.inner_text()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    order_ok = (re.match(r"⚠ CRITICAL · PREDICTED \(next 6 h\)", lines[0])
                and re.match(r"Bed \d+ · P-[0-9A-F]{4} · just now", lines[1]) and lines[2].startswith("Risk: Shock"))
    check(f"[{tag}] notification content in fixed order", order_ok, " | ".join(lines[:6]))
    buttons = [b.inner_text() for b in card.locator("button").all()]
    check(f"[{tag}] buttons Open / Acknowledge / Snooze 30 min only", buttons == ["Open", "Acknowledge", "Snooze 30 min"], str(buttons))
    crit_bed = card.get_attribute("data-bed")
    check(f"[{tag}] bed tile turns CRITICAL", page.locator(f"[data-testid=bed-{crit_bed}]").get_attribute("data-tier") == "CRITICAL")
    page.keyboard.press("n")
    time.sleep(2.5)
    check(f"[{tag}] N again: no repeat notification or chime",
          page.locator("[data-testid=notification][data-level=CRITICAL]").count() == 1 and page.evaluate("window.__cpChimes") == 1)

    # M: MEDIUM, silent
    page.keyboard.press("m")
    page.wait_for_selector("[data-testid=notification][data-level=MEDIUM]", timeout=3000)
    check(f"[{tag}] M shows MEDIUM notification without sound", page.evaluate("window.__cpChimes") == 1)
    check(f"[{tag}] NORMAL never notifies", page.locator("[data-testid=notification][data-level=NORMAL]").count() == 0)
    page.screenshot(path=str(SHOTS / f"{tag}-2-notifications.png"))

    # Open the CRITICAL patient
    card.get_by_role("button", name="Open").click()
    page.wait_for_selector("[data-testid=patient-dashboard]")
    title = page.locator("[data-testid=patient-title]").inner_text()
    check(f"[{tag}] Open shows the patient dashboard", title.startswith(f"Bed {crit_bed} · P-"), title)
    drivers = page.locator("[data-testid=drivers] li").all_inner_texts()
    check(f"[{tag}] prediction panel: 2-3 top drivers", 2 <= len(drivers) <= 3, str(drivers))
    check(f"[{tag}] horizon label", "Prediction horizon: next 6 h" in page.locator("[data-testid=prediction-panel]").inner_text())
    nums = {k: page.locator(f"[data-testid=numeric-{k}]").inner_text().split("\n")[1] for k in ("hr", "map", "spo2", "rr")}
    check(f"[{tag}] monitor numerics shown", all(v not in ("–", "") for v in nums.values()), str(nums))
    mon_txt = page.locator("[data-testid=monitor]").inner_text()
    check(f"[{tag}] vitals from 5-minute monitor data, axis before alert",
          "5-minute monitor values" in mon_txt and "Hours before the alert" in mon_txt, mon_txt[-160:].replace("\n", " "))
    check(f"[{tag}] waveform labelled as illustrative", "Illustrative waveform – not patient data" in page.locator("[data-testid=waveforms]").inner_text())
    check(f"[{tag}] footer text", page.locator("[data-testid=footer]").inner_text() ==
          "Clinical decision support – not a diagnosis. Feedback is reviewed by the clinical team; the model is not retrained automatically.")
    mon, stack = page.locator("[data-testid=monitor]").bounding_box(), page.locator("[data-testid=notification-stack]").bounding_box()
    check(f"[{tag}] notifications never cover the vital-sign curves", not boxes_overlap(mon, stack), f"monitor {mon} stack {stack}")
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    check(f"[{tag}] no horizontal overflow", overflow <= 0, f"{overflow}px")
    page.screenshot(path=str(SHOTS / f"{tag}-3-patient.png"), full_page=True)

    # actions
    fa = page.get_by_role("button", name="✕ False alarm")
    check(f"[{tag}] False alarm disabled until a reason is chosen", fa.is_disabled())
    page.get_by_label("False alarm reason").select_option("Artifact")
    check(f"[{tag}] False alarm enabled with a reason", fa.is_enabled())
    fa.click()
    page.wait_for_selector("[data-testid=toast]:has-text('False alarm logged – Artifact')", timeout=3000)
    check(f"[{tag}] False alarm logged", True)
    page.get_by_role("button", name="✓ Acknowledge").click()
    page.wait_for_selector("[data-testid=toast]:has-text('Acknowledged – logged')", timeout=3000)
    check(f"[{tag}] Acknowledge logged", True)
    page.wait_for_selector("[data-testid=history-panel]:has-text('false alarm (Artifact)')", timeout=3000)
    check(f"[{tag}] history shows nurse actions", "acknowledge" in page.locator("[data-testid=history-panel]").inner_text())
    page.get_by_role("button", name="☎ Call doctor").click()
    page.wait_for_selector("[data-testid=toast]:has-text('Doctor notified – logged')", timeout=3000)
    page.wait_for_selector("[data-testid=ward-grid]", timeout=4000)
    check(f"[{tag}] Call doctor confirms and returns to the ward", page.locator("[data-testid=patient-dashboard]").count() == 0)

    # notification buttons on the MEDIUM card
    med = page.locator("[data-testid=notification][data-level=MEDIUM]")
    med.get_by_role("button", name="Snooze 30 min").click()
    check(f"[{tag}] Snooze hides the notification", page.locator("[data-testid=notification][data-level=MEDIUM]").count() == 0)
    page.keyboard.press("r")
    page.keyboard.press("m")
    page.wait_for_selector("[data-testid=notification][data-level=MEDIUM]", timeout=3000)
    page.locator("[data-testid=notification][data-level=MEDIUM]").get_by_role("button", name="Acknowledge").click()
    check(f"[{tag}] Acknowledge (notification) hides it", page.locator("[data-testid=notification]").count() == 0)
    page.keyboard.press("m")
    page.locator("[data-testid=notification][data-level=MEDIUM]").count()

    # R resets
    page.keyboard.press("r")
    tiers = [tiles.nth(i).get_attribute("data-tier") for i in range(tiles.count())]
    check(f"[{tag}] R resets ward and notifications", set(tiers) == {"NORMAL"} and page.locator("[data-testid=notification]").count() == 0)

    # F toggles full screen
    page.keyboard.press("f")
    time.sleep(0.5)
    fs_on = page.evaluate("!!document.fullscreenElement")
    page.keyboard.press("f")
    time.sleep(0.5)
    fs_off = not page.evaluate("!!document.fullscreenElement")
    check(f"[{tag}] F toggles full screen", fs_on and fs_off, f"on={fs_on} off={fs_off}")


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        for size in ((1920, 1080), (1440, 900)):
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            run(page, size)
            check(f"[{size[0]}x{size[1]}] no JavaScript errors", not errors, "; ".join(errors)[:300])
            page.close()
        browser.close()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

"""End-to-end click test of the CliniCore pages (needs a running backend on 127.0.0.1:8765 and Chrome).
python tests/e2e_clinicore.py [--chrome PATH] [--shots DIR]"""
import argparse, os
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser(); ap.add_argument("--chrome", default=None); ap.add_argument("--shots", default="e2e_shots"); ap.add_argument("--url", default="http://127.0.0.1:8765")
a = ap.parse_args(); os.makedirs(a.shots, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=a.chrome, args=["--no-sandbox"]) if a.chrome else p.chromium.launch()
    pg = b.new_page(viewport={"width": 1600, "height": 1100}); errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    # 1. ED intake: invalid then valid
    pg.goto(a.url + "/#/intake"); pg.get_by_role("button", name="Assess").click()
    pg.wait_for_timeout(300)
    pg.get_by_role("button", name="Fever + cough, 67 y").click(); pg.get_by_role("button", name="Assess").click()
    pg.get_by_text("What the clinician sees").wait_for(); pg.screenshot(path=f"{a.shots}/1_intake_result.png", full_page=True)
    assert pg.get_by_text("A doctor will review your case").count() or pg.get_by_text("This could be").count()
    # 2. Doctor queue: open the new intake and confirm a group
    pg.goto(a.url + "/#/doctor"); pg.get_by_text("New intake", exact=False).first.click()
    pg.get_by_role("button", name="Confirm diagnosis group").click(); pg.get_by_text("Saved", exact=False).first.wait_for()
    pg.screenshot(path=f"{a.shots}/2_doctor_saved.png", full_page=True)
    # 3. ICU: a false alarm needs a reason
    pg.goto(a.url + "/#/icu"); pg.get_by_role("button", name="False alarm", exact=True).first.click()
    assert pg.get_by_role("button", name="Log false alarm").is_disabled()
    assert pg.get_by_role("button", name="Real warning").count() == 0
    pg.get_by_text("Sensor or measurement").first.click(); pg.get_by_role("button", name="Log false alarm").click()
    pg.get_by_text("False alarm logged", exact=False).first.wait_for(); pg.wait_for_timeout(600)
    pg.screenshot(path=f"{a.shots}/3_icu_false_alarm.png", full_page=False)
    # 4. Notify doctor: nurse sends, a doctor screen gets the pop-up
    doc = b.new_context(viewport={"width": 1366, "height": 768}).new_page(); doc.on("pageerror", lambda e: errors.append(str(e)))
    doc.goto(a.url + "/#/inbox"); doc.get_by_role("button", name="ICU attending on call").click(); doc.goto(a.url + "/#/doctor"); doc.wait_for_timeout(1500)
    pg.goto(a.url + "/#/icu"); pg.get_by_role("button", name="🔔 Notify doctor").click(); pg.get_by_role("button", name="Send to").click()
    pg.get_by_text("Sent to", exact=False).first.wait_for()
    doc.get_by_role("button", name="On my way").first.wait_for(timeout=15000); doc.screenshot(path=f"{a.shots}/4_doctor_popup.png")
    doc.get_by_role("button", name="On my way").first.click()
    # 5. Trust + monitor demo render
    pg.goto(a.url + "/#/trust"); pg.get_by_text("How CliniCore earns trust").wait_for(); pg.screenshot(path=f"{a.shots}/4_trust.png", full_page=True)
    pg.goto(a.url + "/#/present"); pg.get_by_text("ICU Central Monitor").wait_for()
    assert not errors, errors
    print("E2E OK – screenshots in", a.shots); b.close()

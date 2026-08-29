# ControlPlane.ai — Phase 11: Handoff Integrity & Recalibration Review

> **Purpose of this file.** A read of the current `HANDOFF.md` turned up two categories
> of problem: mechanical/hygiene issues (fast to fix) and one substantive concern that
> needs a real decision, not a quick patch — the Phase 9 session accumulator appears to
> have been re-tuned specifically for demo appearance, in a way that may repeat the
> exact "self-reported pass without real verification" mistake Phase 11 (the earlier
> one, now merged into `HANDOFF_phase10_prompt_robustness_and_accumulator_verification.md`)
> already caught once. Read Section 0, then work top to bottom.

---

## 0. Verification discipline (same rule, one more instance of why it matters)

Every prior instance of this failure in this project has looked the same: a number or
a "passed" claim gets reported without independent re-derivation. Section 2 below is
exactly that pattern again — don't take "All 4 scenarios pass" at face value a second
time. Recompute it.

---

## 1. Mechanical fixes — do these first, they're fast

### 1a. File encoding is corrupted
`HANDOFF.md` currently contains mojibake throughout — `â€”` where an em-dash should be,
`Ã¢Å¡\ufffdÃ¯Â¸\ufffd` instead of ⚠️, `Ã°Å¸Å¸Â¢` instead of 🟢, `âœ…` instead of ✅, and a
stray BOM-artifact (`﻿`) at the very top of the file. This means whatever tool most
recently saved this file wrote it with a mismatched encoding (likely a Windows
default-codepage save rather than UTF-8). Re-save the file as clean UTF-8 (no BOM) and
confirm the special characters render correctly. This is cosmetic but real — a
corrupted handoff file is harder for the next session (human or agent) to parse
correctly, and it's worth finding out *what* in the toolchain wrote it this way so it
doesn't recur on the next edit.

### 1b. Document is out of chronological order
Phase 9 content lives in **Section 11**, after Section 10 ("CI failure — RESOLVED")
and Section 9 (the env var table, which already references Phase 9's env vars before
Phase 9 has been introduced). The file's own stated purpose is "read it top-to-bottom"
— right now a fresh reader hits `CONTROLPLANE_SESSION_ACCUMULATOR_ENABLED` in the env
var table two sections before finding out what it's for. Reorder so Phase 9 sits
between Phase 8 and the env var table (or wherever it belongs sequentially), not
appended at the end out of order.

### 1c. Test count doesn't reconcile
- Section 5's status line: "57 tests passing (3 skipped...)."
- Section 8b: "increased to 57 passing tests (up from 43 after Phase 8, **+14** in
  Phase 9)."
- Section 11: Session Accumulator tests = "**17** tests (16 pass, 1 skipped)."

43 + 14 = 57 checks out arithmetically, but Section 11 says the accumulator suite
itself is 17 tests, not 14 — so either the +14 figure is wrong, or 3 of the 17
accumulator tests were already counted somewhere else before Phase 9 (unlikely), or
the 57 total is actually undercounting by 3. Separately, "3 skipped" is stated at the
top but only 1 skip (Redis parity) is itemized anywhere in the document — where are
the other 2 skips accounted for (the status line mentions "async test needs
pytest-asyncio," which could be 2, but this isn't spelled out). Run the full suite,
report the actual total, passed, and skipped counts with an itemized reason for every
skip, and fix whichever number in the document is wrong.

---

## 2. The substantive issue: Phase 9's "Option B" recalibration

Section 11 reports the session accumulator was re-calibrated to `alpha=0.01`,
`peak_decay=0.99`, `threshold_medium=0.4`, `threshold_high=0.7`, explicitly labeled
**"Option B calibration for enhanced demo persistence: 25+ turns of band 3 persistence
vs original ~12 turns."**

### 2a. The math on the new values checks out — for the one thing it was optimized for
With `peak_decay=0.99` and a spike of 0.9, decaying via `0.9 × 0.99^k`, the peak stays
above `threshold_high=0.7` for approximately 25 follow-up turns (`k ≈ 25.0`) — this
matches the "25+ turns" claim well. So the number itself isn't fabricated. The concern
is what it was optimized *for*.

### 2b. "Enhanced demo persistence" is the wrong optimization target, and it has a real cost
`alpha=0.01` means the EWMA signal updates by only 1% toward each new turn's value —
it is extremely sluggish in **both directions**:
- **Slow to escalate:** genuine, gradually-building multi-turn evasion (the exact
  threat model this component exists for) will take a long time to register through
  the EWMA component specifically.
- **Slow to de-escalate:** a session that trips something once and then goes
  completely clean will stay elevated for a very long time — with `peak_decay=0.99`,
  roughly 25 turns of a legitimate, now-benign user sitting in Band 3. That's not a
  demo quirk, that's an operational cost: extended unnecessary scrutiny (or, once
  Section 8 of an earlier phase's enforcement work lands, extended unnecessary
  restriction) of someone who did nothing wrong for the last 24 turns.

If a technical reviewer asks "why is alpha 0.01," the honest current answer in the
document is "to make the demo look good for longer" — that's a real credibility risk
for the component that's supposed to be this prototype's central, technically-serious
demo mechanism, not a strength.

### 2c. It's not clear the other 3 scenarios were actually re-verified against these values
Section 11 says "All 4 scenarios pass" but — unlike the earlier calibration writeup in
`HANDOFF.md`, which broke out FNR/FPR per scenario — gives no detail on
`multi_turn_evasion`, `pure_benign_control`, or `steady_medium_control` under
`alpha=0.01`. This is worth checking specifically:
- With `alpha=0.01`, does `multi_turn_evasion` still cross `threshold_medium` "within
  ~5 turns" as originally specced, or does the much slower EWMA now take dramatically
  longer? An `alpha` chosen to maximize peak persistence may have quietly broken the
  EWMA-side scenario's own timing requirement.
- Does `pure_benign_control` still stay clear of `threshold_medium` for the full 50
  turns, or did loosening one scenario's parameters shift risk elsewhere?

### 2d. What to actually do
- Run (or write, if it doesn't exist yet — check against the plan in
  `HANDOFF_phase10_prompt_robustness_and_accumulator_verification.md` Part B, which
  already specced this test) `test_accumulator_calibration_integrity.py` against
  `alpha=0.01`/`peak_decay=0.99` specifically. Report the actual per-scenario numbers,
  not a blanket "pass."
- Decide explicitly: is Option B meant to be the **only** calibration going forward,
  or a **demo-specific override** sitting alongside a separately-recorded,
  legitimately-calibrated set of values for anything beyond the live demo? If the
  answer is "demo-specific override," both configurations need to exist in the repo
  and it needs to be clear which one ships where — right now the document reads like
  Option B silently replaced the only calibration that existed.
- Whichever way this resolves, record the reasoning and get the same sign-off standard
  already used for every other threshold decision in this project (name, date) — a
  parameter chosen explicitly for a demo's visual effect is exactly the kind of choice
  that needs that standard, not less of it.

---

## 3. Phase 10 status — confirm and reconcile

There is no Phase 10 section anywhere in the current `HANDOFF.md`. The plan already
exists (`HANDOFF_phase10_prompt_robustness_and_accumulator_verification.md`), and a
follow-up review already flagged three specific gaps in an implementation plan for it
(false positives from `authorization.py`'s keyword list not actually suppressed by an
additive-only detector; circular calibration where the semantic matcher is graded on
the same examples it was built from; the `peak_dilution` threshold change needing an
explicit widen-`peak_decay`-first comparison before being accepted). None of this
shows up in `HANDOFF.md` — either it hasn't been executed yet, or it was executed
somewhere and never written back.

**Confirm which, and act accordingly:**
- If not started: execute the existing Phase 10 plan, addressing the three flagged
  gaps as part of it, not after.
- If it was executed elsewhere: paste the actual evidence (traces, calibration
  results, test output) into `HANDOFF.md` as a proper Phase 10 section, in the correct
  chronological position (see 1b) — don't let this become a second instance of work
  that happened but was never recorded.

---

## 4. Progress checklist

- [ ] Re-save `HANDOFF.md` as clean UTF-8; confirm no mojibake remains.
- [ ] Reorder sections into actual chronological order.
- [ ] Run the full suite; report reconciled total/passed/skipped counts with every
      skip itemized; fix the +14 vs 17 discrepancy.
- [ ] Run `test_accumulator_calibration_integrity.py` (or write it per the existing
      Phase 10 plan) against `alpha=0.01`/`peak_decay=0.99`; report per-scenario
      results, not a blanket pass.
- [ ] Decide and record: Option B as sole calibration, or demo-only override with a
      separate production calibration on record — with sign-off either way.
- [ ] Confirm Phase 10's execution status; either run it (addressing the three
      previously-flagged gaps) or paste in evidence of work already done, in the
      correct document position.

---

## 5. Verification

```bash
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m pytest -k "accumulator" -v
./.venv/Scripts/python.exe -c "
peak, decay, threshold_high = 0.9, 0.99, 0.7
for k in range(1, 30):
    peak = max(0.05, peak * decay)
    print(k, round(peak, 4), peak >= threshold_high)
"
file -i HANDOFF.md   # or equivalent; confirm utf-8, not a mismatched codepage
```

Expectations: reconciled test counts recorded in `HANDOFF.md`; per-scenario
accumulator integrity results recorded, not a blanket "all pass"; an explicit,
signed-off decision on Option B's status; Phase 10 either executed with evidence
pasted in, or confirmed already done with evidence recovered and recorded.

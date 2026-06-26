# Object base-SR parity verdict (2026-06-26)

Ran the colleague's EXACT evaluate_act.py + their seed scheme (1000+ep, 20 ep/task, replan 5)
on their own act/30000 checkpoint.

Per-task: 0,3,5 = 0% (deterministic hard-fails); 1,2,4,6,7,8,9 = 100%.
**Object base = 70.0%**  (colleague README: 81.7 | our 3-seed sweep: 70.0/70.0/70.0)

VERDICT: our base 70.0 reproduces exactly with their harness. The colleague's 81.7 does NOT
reproduce on act/30000 — it is optimistic (different ckpt/seed/config on their side). Our base SR
is the faithful one; AEGIS deltas (base+AEGIS share the harness) are unaffected.

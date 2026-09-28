# T11 sequence generators

**Location:** `/workspace/scp/experiments/phase-routing/t11/generators/`  
**Scope:** Smoke examples + **sealed counting batch** (`sealed/`). Do **not** call models from these scripts. Do **not** freeze PREREG from this step alone.

## Sequence types

| Type | Module | Oracle |
|------|--------|--------|
| Clean | `clean.py` | Content-defined (Browser / Structured / General) |
| Ambiguous | `ambiguous.py` | Same; cue buried mid-segment |
| Distraction | `distraction.py` | Same; short off-task span metadata |
| Renewal | `renewal.py` | Regime A → B → A return |
| Window-break | `window_break.py` | Same texts; **carry mode in harness metadata only** |

## JSONL fields

```json
{
  "seed": 20260923,
  "sequence_type": "Clean",
  "segments": [
    {
      "family": "browser",
      "oracle_specialist": "Browser",
      "text": "...",
      "notes": "optional",
      "distraction_span": {"start_token": 0, "end_token": 0, "n_tokens": 0},
      "browser_episode": {"task": "miniwob.click-button", "seed": 101, "suite": "MINIWOB25_SEALED"}
    }
  ],
  "window_break": {
    "at": 1,
    "carry_mode": "hum-state|classifier-buffer|context-carry|cold",
    "placement": "at_task_boundary|inside_task"
  }
}
```

- **No planted tags** in `text`. Oracle comes from content/`oracle_specialist` field for scoring only.
- **Refuse list** (`t11/HAND_RULE.md`): generators reject refuse-cue language so hand-rule training data stays clean.
- **Window-break carry modes** are harness metadata, never written into segment text.
- **Seeds (draft):** `[20260923, 20260924, 20260925, 20260926, 20260927]` — same list across types.
- **Browser:** sealed MiniWoB-25 prompts (`MINIWOB25_SEALED.json`); `enter-password` omitted (refuse word).

## Smoke emit

```bash
cd /workspace/scp/experiments/phase-routing/t11/generators
python3 emit_smoke.py
```

Outputs: `smoke/*.jsonl`.

## Sealed counting batch

```bash
cd /workspace/scp/experiments/phase-routing/t11/generators
python3 emit_counting.py
```

Outputs under `sealed/`:

| File | Contents |
|------|----------|
| `clean.jsonl` | 5 sequences (1× seed) |
| `ambiguous.jsonl` | 5 |
| `distraction.jsonl` | 5 |
| `renewal.jsonl` | 5 |
| `window_break.jsonl` | 40 = 5 seeds × 4 carry modes × 2 placements |
| `ALL_SEQUENCES.jsonl` | 60 combined |
| `SEALED_MANIFEST.json` | counts + sha256 |
| `FILES.sha256` | per-file digests |

**Batch id:** `t11-counting-sealed-v1`. Ready for routing smoke; PREREG remains unfrozen.

## Not in scope here

- Model inference / specialist calls
- Freezing PREREG
- Reconstructing the MiniWoB harness


## T11.2 Rule 19 generator work (dev — not sealed)

Ambiguous templates now bury family cues **after** the first 32 tokens.
Distraction inserts use **other-family** keyword spans (6–9 tokens).

```bash
cd /workspace/scp/experiments/phase-routing/t11/generators
python3 emit_t11_2_dev.py          # writes t11_2_dev/ (does NOT overwrite sealed/)
python3 validate_rule19_geometry.py --dir t11_2_dev
python3 /workspace/scp/t11/scripts/rule19_freeze_gates.py --sealed-dir t11_2_dev
```

`sealed/` (= `t11-counting-sealed-v1`) is intentionally left as the T11 dry-run
batch that **FAILS** Rule 19 gate 1 (Ambiguous first-32 CSR = 1.0). Do not
re-seal until gates 1–4 PASS and Kyle authorizes.


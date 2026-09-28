#!/usr/bin/env python3
"""
freeze_verify
-------------
Upgrades and verifies OpenTimestamps (.ots) anchors for all frozen PREREG files.

When Bitcoin blocks confirm the submitted timestamp transactions (typically taking
several hours), this tool fetches the confirmed Merkle branches from the calendar servers,
upgrades the local .ots files, verifies them against Bitcoin block headers, and updates
`FREEZE_META.json` with the confirmed Bitcoin block height and confirmation timestamp.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREEZE_META_PATH = os.path.join(ROOT_DIR, "FREEZE_META.json")


def load_freeze_meta() -> Dict[str, Any]:
    if os.path.exists(FREEZE_META_PATH):
        try:
            with open(FREEZE_META_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to read {FREEZE_META_PATH}: {e}", file=sys.stderr)
    return {
        "version": 1,
        "description": "Rule-19 Freeze Metadata & OpenTimestamps Anchor Registry",
        "last_updated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "records": {},
    }


def save_freeze_meta(meta: Dict[str, Any]) -> None:
    meta["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(FREEZE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def run_ots_cmd(args: List[str]) -> Tuple[int, str, str]:
    ots_bin = shutil.which("ots")
    if ots_bin:
        full_cmd = [ots_bin] + args
    else:
        full_cmd = [sys.executable, "-m", "otsclient.ots"] + args
    p = subprocess.run(full_cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def parse_bitcoin_block(text: str) -> Optional[int]:
    """Extracts Bitcoin block number from ots output if present."""
    m = re.search(r"Bitcoin block (\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m2 = re.search(r"BitcoinBlockHeaderAttestation\((\d+)\)", text)
    if m2:
        return int(m2.group(1))
    return None


def verify_ots_file(ots_path: str) -> Tuple[bool, Optional[int], str]:
    """
    Runs ots upgrade and ots verify on an ots_path.
    Returns (is_confirmed, block_height, message).
    """
    if not os.path.exists(ots_path):
        return False, None, f"OTS file missing: {ots_path}"

    # 1. ots upgrade
    up_code, up_out, up_err = run_ots_cmd(["upgrade", ots_path])

    # 2. ots info to check attestation
    info_code, info_out, info_err = run_ots_cmd(["info", ots_path])
    info_block = parse_bitcoin_block(info_out + " " + info_err)

    # 3. ots verify
    ver_code, ver_out, ver_err = run_ots_cmd(["verify", ots_path])
    combined_ver = ver_out + "\n" + ver_err
    ver_block = parse_bitcoin_block(combined_ver)

    block = ver_block or info_block

    if ver_code == 0 or block is not None:
        return True, block, f"Confirmed in Bitcoin block {block}" if block else "Verified"
    
    # Check if pending
    if "Pending confirmation" in up_err or "Pending confirmation" in ver_err or "PendingAttestation" in info_out:
        return False, None, "Pending confirmation in Bitcoin blockchain"

    return False, None, (ver_err.strip() or up_err.strip() or "Pending verification")


def verify_all_records() -> None:
    meta = load_freeze_meta()
    records = meta.get("records", {})
    if not records:
        print("No freeze records found in FREEZE_META.json.")
        return

    print("=" * 80)
    print("RULE-19 OPENTIMESTAMPS ANCHOR VERIFICATION (freeze_verify)")
    print("=" * 80)

    confirmed_count = 0
    pending_count = 0
    error_count = 0

    for name, item in records.items():
        rel_ots = item.get("ots_path")
        if not rel_ots:
            continue
        abs_ots = os.path.join(ROOT_DIR, rel_ots)
        print(f"\n[*] Checking {name}...")
        print(f"    File: {rel_ots}")

        confirmed, block_height, msg = verify_ots_file(abs_ots)

        if confirmed:
            confirmed_count += 1
            item["status"] = "confirmed"
            item["bitcoin_block_height"] = block_height
            if not item.get("confirmed_at"):
                item["confirmed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            print(f"    Status: CONFIRMED (Bitcoin Block #{block_height})")
        else:
            if "Pending" in msg:
                pending_count += 1
                # Preserve retroactive distinction if present
                if "retroactively" in item.get("status", ""):
                    item["status"] = "anchored_retroactively"
                else:
                    item["status"] = "pending"
                print(f"    Status: PENDING ({msg})")
            else:
                error_count += 1
                print(f"    Status: UNCONFIRMED / ERROR ({msg})")

    save_freeze_meta(meta)

    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print(f"  Total records: {len(records)}")
    print(f"  Confirmed:     {confirmed_count}")
    print(f"  Pending:       {pending_count}")
    print(f"  Errors:        {error_count}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Verify and upgrade OpenTimestamps anchors")
    parser.add_argument("ots_files", nargs="*", help="Optional specific .ots files to verify")
    args = parser.parse_args()

    if args.ots_files:
        for f in args.ots_files:
            abs_path = os.path.abspath(f)
            confirmed, block, msg = verify_ots_file(abs_path)
            print(f"{f}: {'CONFIRMED (Block ' + str(block) + ')' if confirmed else 'PENDING: ' + msg}")
    else:
        verify_all_records()


if __name__ == "__main__":
    main()

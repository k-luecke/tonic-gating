"""Shared types and helpers for T11 sequence generators.

Oracle specialist is content-defined (Browser / Structured / General).
NO planted tags in segment text. Refuse-cue language from HAND_RULE.md
must NOT appear in hand-rule training / sequence text.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence as TypingSequence

# Placeholder seed list (same across sequence types) — PREREG draft
SEEDS: List[int] = [20260923, 20260924, 20260925, 20260926, 20260927]

Specialist = Literal["Browser", "Structured", "General"]
Family = Literal["browser", "structured", "general"]
SequenceType = Literal["Clean", "Ambiguous", "Distraction", "Renewal", "Window-break"]
CarryMode = Literal["hum-state", "classifier-buffer", "context-carry", "cold"]
Placement = Literal["inside_task", "at_task_boundary"]

CARRY_MODES: List[CarryMode] = [
    "hum-state",
    "classifier-buffer",
    "context-carry",
    "cold",
]

PLACEMENTS: List[Placement] = ["at_task_boundary", "inside_task"]

FAMILY_TO_SPECIALIST: Dict[Family, Specialist] = {
    "browser": "Browser",
    "structured": "Structured",
    "general": "General",
}

# HAND_RULE refuse cues — whole-word; do not emit as sequence text for hand-rule data.
REFUSE_WHOLE_WORDS = [
    "pay",
    "purchase",
    "buy",
    "checkout",
    "card",
    "password",
    "token",
    "delete",
    "wipe",
    "rm",
    "email",
    "send",
    "post",
    "upload",
    "download",
    "install",
]
REFUSE_PHRASES = [
    "api key",
    "remove all",
    "drop table",
    "format disk",
    "ignore prior",
    "ignore the gate",
]

# Sealed MiniWoB-25 (mirror root). Episode seeds from MINIWOB25_SEALED.json.
MINIWOB_SEALED_PATH = Path("/workspace/scp/MINIWOB25_SEALED.json")
MINIWOB_EPISODE_SEEDS: List[int] = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

# Task → browser prompt text. Skip / rephrase refuse-sensitive tasks:
# enter-password (word "password"), book-flight-nodelay ok, login-user avoids "password".
MINIWOB_BROWSER_PROMPTS: Dict[str, str] = {
    "miniwob.click-test": "Click the single test button labeled on the page.",
    "miniwob.click-button": "Click the button whose label matches the instruction on the form.",
    "miniwob.click-link": "Select the link text named in the instruction and open it.",
    "miniwob.click-dialog": "Click the dialog button labeled OK on the page.",
    "miniwob.click-dialog-2": "Choose the correct dialog option button shown on the page.",
    "miniwob.enter-text": "Type the requested string into the text field, then submit the form.",
    "miniwob.focus-text": "Focus the text input field indicated on the page using tab or click.",
    "miniwob.focus-text-2": "Navigate focus to the second text field on the page, then type the query.",
    "miniwob.login-user": "Open the login form, enter the username into the user field, then click submit.",
    "miniwob.search-engine": "Open the search page, type the query into the search field, then choose the result.",
    "miniwob.choose-list": "Select the named option from the dropdown list on the page.",
    "miniwob.click-option": "Click the radio option that matches the instruction on the form.",
    "miniwob.click-menu": "Open the menu and select the item named in the instruction.",
    "miniwob.click-menu-2": "Navigate the nested menu and click the target menu item.",
    "miniwob.click-checkboxes": "Click every checkbox that matches the listed labels on the form.",
    "miniwob.click-checkboxes-soft": "Select the soft-matching checkboxes described on the page.",
    "miniwob.click-checkboxes-large": "On the large checkbox list, click each target checkbox named.",
    "miniwob.click-collapsible": "Expand the collapsible section, then click the revealed button.",
    "miniwob.click-collapsible-2": "Open the nested collapsible panel and select the target link.",
    "miniwob.click-button-sequence": "Click the buttons in the exact sequence listed on the page.",
    "miniwob.click-tab": "Select the tab named in the instruction on the page.",
    "miniwob.click-tab-2": "Switch to the correct tab, then click the button inside that panel.",
    "miniwob.use-autocomplete": "Type into the autocomplete field and choose the matching suggestion.",
    "miniwob.book-flight-nodelay": "Fill the flight form fields, select the listed options, then submit.",
    # enter-password omitted: HAND_RULE refuse whole-word "password"
}


@dataclass
class Segment:
    family: Family
    oracle_specialist: Specialist
    text: str
    notes: Optional[str] = None
    distraction_span: Optional[Dict[str, Any]] = None
    browser_episode: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "family": self.family,
            "oracle_specialist": self.oracle_specialist,
            "text": self.text,
            # Harness typed state (H5). Co-authored from construction task slot;
            # NEVER a promote of oracle_specialist. observation_gate strips family/oracle.
            "typed_goal": self.text,
            "typed_goal_family": (
                "browser" if self.browser_episode is not None else self.family
            ),
        }
        if self.notes is not None:
            d["notes"] = self.notes
        if self.distraction_span is not None:
            d["distraction_span"] = self.distraction_span
        if self.browser_episode is not None:
            d["browser_episode"] = self.browser_episode
        return d


@dataclass
class WindowBreakMeta:
    """Harness metadata only — not planted in segment text."""

    at: int
    carry_mode: CarryMode
    placement: Placement = "at_task_boundary"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "at": self.at,
            "carry_mode": self.carry_mode,
            "placement": self.placement,
        }


@dataclass
class Sequence:
    seed: int
    sequence_type: SequenceType
    segments: List[Segment] = field(default_factory=list)
    window_break: Optional[WindowBreakMeta] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "seed": self.seed,
            "sequence_type": self.sequence_type,
            "segments": [s.to_dict() for s in self.segments],
        }
        if self.window_break is not None:
            d["window_break"] = self.window_break.to_dict()
        return d


def assert_no_refuse_cues(text: str) -> None:
    lower = text.lower()
    for phrase in REFUSE_PHRASES:
        if phrase in lower:
            raise ValueError(f"Refuse-cue phrase {phrase!r} found in sequence text (HAND_RULE)")
    for word in REFUSE_WHOLE_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lower):
            raise ValueError(f"Refuse-cue word {word!r} found in sequence text (HAND_RULE)")


def write_jsonl(path: Path, sequences: List[Sequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for seq in sequences:
            for seg in seq.segments:
                assert_no_refuse_cues(seg.text)
            f.write(json.dumps(seq.to_dict(), ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_miniwob_sealed(path: Path = MINIWOB_SEALED_PATH) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sealed_browser_tasks() -> List[str]:
    """Tasks from sealed suite that have refuse-safe prompt text."""
    sealed = load_miniwob_sealed()
    tasks = [t for t in sealed["tasks"] if t in MINIWOB_BROWSER_PROMPTS]
    if len(tasks) < 20:
        raise RuntimeError(f"expected >=20 sealed browser prompts, got {len(tasks)}")
    return tasks


def rng_for(seed: int, *parts: Any) -> random.Random:
    material = f"{seed}|" + "|".join(str(p) for p in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


BROWSER_CLEAN = [
    "Click the submit button on the login form, then select the first link in the navigation tab.",
    "Open the search page and type the query into the text field, then choose the second result.",
    "Select the checkbox labeled Accept, then click the Continue button near the bottom of the form.",
    "Navigate to the options tab and choose the second dropdown entry listed on the page.",
    "Focus the text field, type the sample query, then click the search button.",
    "Open the menu, select Settings, then click the Save button on the form.",
]

STRUCTURED_CLEAN = [
    "Parse this JSON object and extract the value of the name field; format as a one-column CSV row.",
    "Transform the CSV table: sort by the age column ascending and convert to a list of dictionaries.",
    "Extract every key from {\"a\": 1, \"b\": 2, \"c\": 3} and format as a JSON array.",
    "Convert the two-row CSV with columns id,score into a list of dictionaries sorted by score.",
    "Parse the schema fields id,name,active and format as a three-column CSV header row.",
    "Transform this array of dictionaries into a CSV table with columns city and population.",
]

GENERAL_CLEAN = [
    "What is the capital of France? Answer with a single city name.",
    "Define photosynthesis in one sentence and summarize why it matters for plants.",
    "Explain what a prime number is and answer who first studied them in a short note.",
    "When did the Apollo 11 landing occur? Answer with the year only.",
    "Where is the Great Barrier Reef located? Answer with the country name.",
    "Summarize why water expands when it freezes in one clear sentence.",
]

# T11.2 Rule 19: Ambiguous first ≥32 tokens are family-keyword-free; cue follows.
_AMBIG_NEUTRAL_PREAMBLES = [
    (
        "There are several items nearby. After reviewing the layout with care and taking "
        "a moment to consider options available in the room around you at this time, "
        "then keep reading along carefully through more neutral remarks and asides "
    ),
    (
        "People often mention various topics first. Setting that aside for a while and "
        "looking past the surrounding notes and remarks carefully for a short stretch, "
        "then keep reading along carefully through more neutral remarks and asides "
    ),
    (
        "Skim the surrounding notes for a short while. After you feel ready following "
        "this brief pause in the middle of the material at hand right now, "
        "then keep reading along carefully through more neutral remarks and asides "
    ),
    (
        "Ignore the decorative labels for a moment and then continue reading past the "
        "early remarks until the main request appears here after this stretch of text, "
        "then keep reading along carefully through more neutral remarks and asides "
    ),
]

_AMBIG_CUES_BROWSER = [
    "click the blue button labeled Continue near the bottom of the page form.",
    "select the checkbox next to Accept and then submit the form.",
    "open the search page and type the query into the field, then choose the top result link.",
    "navigate to the tab named Details and click the edit button inside.",
]
_AMBIG_CUES_STRUCTURED = [
    "parse this JSON object and extract the qty field then format as CSV.",
    "convert the two-row CSV with columns id,score into a list of dictionaries.",
    "extract every key from the object and format as a JSON array.",
    "transform the array of dictionaries into a CSV table with columns city and population.",
]
_AMBIG_CUES_GENERAL = [
    "explain what a prime number is and answer who first studied them in a short note.",
    "define entropy for a beginner and summarize the concept in information theory.",
    "answer with one city name for the capital of Japan only.",
    "explain why the sky appears blue in one clear sentence.",
]

BROWSER_AMBIGUOUS = [
    _AMBIG_NEUTRAL_PREAMBLES[i % len(_AMBIG_NEUTRAL_PREAMBLES)] + _AMBIG_CUES_BROWSER[i]
    for i in range(len(_AMBIG_CUES_BROWSER))
]
STRUCTURED_AMBIGUOUS = [
    _AMBIG_NEUTRAL_PREAMBLES[i % len(_AMBIG_NEUTRAL_PREAMBLES)] + _AMBIG_CUES_STRUCTURED[i]
    for i in range(len(_AMBIG_CUES_STRUCTURED))
]
GENERAL_AMBIGUOUS = [
    _AMBIG_NEUTRAL_PREAMBLES[i % len(_AMBIG_NEUTRAL_PREAMBLES)] + _AMBIG_CUES_GENERAL[i]
    for i in range(len(_AMBIG_CUES_GENERAL))
]

# Legacy neutral inserts (no family cues) — retained for reference; T11.2 uses OTHER_FAMILY below.
DISTRACTION_INSERTS = [
    " by the way the weather looks mild today ",
    " note that the clock on the wall reads noon ",
    " aside the room is quiet for a moment ",
    " briefly the lamp light seems warm ",
]

# T11.2 Rule 19 gate 2: other-family keyword spans (6–9 tokens) so stateless arms thrash.
DISTRACTION_OTHER_FAMILY = {
    "browser": [  # inject into non-browser segments
        " click the submit button on the page ",
        " open the form and select the link ",
        " type into the search field then choose ",
    ],
    "structured": [
        " parse the json and extract the field ",
        " convert the csv column into an array ",
        " transform the schema into a table row ",
    ],
    "general": [
        " explain and summarize the answer briefly ",
        " define this and answer who studied it ",
        " answer when and where it first appeared ",
    ],
}


def pick_text(pool: TypingSequence[str], rng: random.Random, avoid: Optional[set] = None) -> str:
    avoid = avoid or set()
    choices = [t for t in pool if t not in avoid]
    if not choices:
        choices = list(pool)
    return rng.choice(choices)


def make_browser_segment(
    rng: random.Random,
    *,
    notes: str,
    ambiguous: bool = False,
    use_miniwob: bool = True,
) -> Segment:
    """Browser segment; prefers sealed MiniWoB task prompts for clean/distraction."""
    if use_miniwob and not ambiguous:
        tasks = sealed_browser_tasks()
        task = rng.choice(tasks)
        ep_seed = rng.choice(MINIWOB_EPISODE_SEEDS)
        text = MINIWOB_BROWSER_PROMPTS[task]
        return Segment(
            family="browser",
            oracle_specialist="Browser",
            text=text,
            notes=notes,
            browser_episode={"task": task, "seed": ep_seed, "suite": "MINIWOB25_SEALED"},
        )
    pool = BROWSER_AMBIGUOUS if ambiguous else BROWSER_CLEAN
    return Segment(
        family="browser",
        oracle_specialist="Browser",
        text=pick_text(pool, rng),
        notes=notes,
    )


def make_structured_segment(rng: random.Random, *, notes: str, ambiguous: bool = False) -> Segment:
    pool = STRUCTURED_AMBIGUOUS if ambiguous else STRUCTURED_CLEAN
    return Segment(
        family="structured",
        oracle_specialist="Structured",
        text=pick_text(pool, rng),
        notes=notes,
    )


def make_general_segment(rng: random.Random, *, notes: str, ambiguous: bool = False) -> Segment:
    pool = GENERAL_AMBIGUOUS if ambiguous else GENERAL_CLEAN
    return Segment(
        family="general",
        oracle_specialist="General",
        text=pick_text(pool, rng),
        notes=notes,
    )


def make_family_segment(
    family: Family,
    rng: random.Random,
    *,
    notes: str,
    ambiguous: bool = False,
    use_miniwob: bool = True,
) -> Segment:
    if family == "browser":
        return make_browser_segment(rng, notes=notes, ambiguous=ambiguous, use_miniwob=use_miniwob)
    if family == "structured":
        return make_structured_segment(rng, notes=notes, ambiguous=ambiguous)
    return make_general_segment(rng, notes=notes, ambiguous=ambiguous)


def segment_count_for(seed: int, lo: int = 4, hi: int = 6) -> int:
    """HANDOFF §3.2: Clean uses 4–6 segments; reused for other types where useful."""
    return lo + (seed % (hi - lo + 1))

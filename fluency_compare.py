'''
human fluency scoring, step 1 of 2: HUMAN INPUT ONLY.

instead of a human assigning an absolute 0-3 label to one signature at a
time (see ml_fluency_supervised.py), this pairs signatures up and asks a
human which of the two is more fluent, and by how much. paired/comparative
judgements are usually more consistent than absolute ones, and the graded
margin gives a continuous scale for free once aggregated.

supports multiple independent raters (see RATERS below). every rater judges
the exact same fixed-size set of pairs (TOTAL_PAIRS, spread across as many
distinct signatures as that budget allows), but each gets their own CSV
with no visibility into anyone else's answers -- so send a rater the sheets
plus *only* their own CSV. two benefits: more comparisons feeding the
aggregate score, and human_fluency_cluster.py can report how much the
raters agree.

judge each pair on:
  - variation of line width (thinnest stroke vs thickest stroke)
  - tortuosity (how winding the strokes are)
  - total stroke length (flowing connected ink vs short choppy marks)
  - evenness of size (consistent character/stroke sizing)
  - moderation of stroke weight (not too heavy/blotchy, not too faint/timid
    -- either extreme suggests unfamiliarity with the brush)
(this rubric isn't printed on the sheets themselves -- pass it along
separately, e.g. in the email to raters.)

this script only builds the pairs, renders comparison sheets, and writes/
merges each rater's CSV. it does no scoring, aggregation, feature
extraction, or clustering -- that's human_fluency_cluster.py, run once at
least one rater's human_margin values are filled in.

# inputs:
# signatures/scqyN_signatures.csv
# signatures/core/scqyN_colCC_sigSS.png
# fluency/fluency_pairwise.csv               (optional, master pair list)
# fluency/fluency_pairwise_<rater>.csv        (optional, previous judgements)

# outputs:
# fluency/fluency_pairwise_sheet_PP.png       (paginated side-by-side pairs,
#                                                  shared by every rater)
# fluency/fluency_pairwise.csv                (master pair list, no answers)
# fluency/fluency_pairwise_<rater>.csv        (one per name in RATERS, each
#                                                  with its own human_margin)
'''

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

SIGNATURES_DIR = Path("signatures")
CORE_DIR       = SIGNATURES_DIR / "core"
OUT_DIR        = Path("fluency")

# who's judging -- add/remove names here. each gets their own CSV, blind to
# the others, all built from the same underlying set of pairs.
RATERS = ["mackenzie", "christoph"]

MASTER_CSV = OUT_DIR / "fluency_pairwise.csv"


def rater_csv(name):
    return OUT_DIR / f"fluency_pairwise_{name}.csv"


PAIR_THUMB     = 220
PAIRS_PER_PAGE = 6

SHEET_PAGES = 8                            # fixed batch size
TOTAL_PAIRS = PAIRS_PER_PAGE * SHEET_PAGES  # 48 -- fills all 8 pages exactly

MAX_PAIRING_ROUNDS = 20


def core_path_for(stem, column_id, sig_id):
    return CORE_DIR / f"{stem}_col{int(column_id):02d}_sig{int(sig_id):02d}.png"


def load_all_signatures():
    rows = []
    for sig_path in sorted(SIGNATURES_DIR.glob("*_signatures.csv")):
        stem = sig_path.stem.replace("_signatures", "")
        sigs = pd.read_csv(sig_path)
        for _, sig in sigs.iterrows():
            col_id, sig_id = int(sig["column_id"]), int(sig["sig_id"])
            crop_path = core_path_for(stem, col_id, sig_id)
            if not crop_path.exists():
                continue
            rows.append({"stem": stem, "column_id": col_id, "sig_id": sig_id,
                         "crop_path": str(crop_path)})
    if not rows:
        raise SystemExit("no signature crops found -- run identify.py first")
    return pd.DataFrame(rows).drop_duplicates(subset=["stem", "column_id", "sig_id"])


def sig_key(row):
    return (row["stem"], int(row["column_id"]), int(row["sig_id"]))


def generate_new_pairs(df, existing_pairs, total_pairs):
    # existing_pairs: set of frozenset({key_a, key_b}) already in the CSV.
    # fills up to total_pairs total (existing + new), each round preferring
    # whichever signatures have the fewest comparisons so far, so a small
    # fixed budget still spreads across as much of the corpus as it can
    # rather than repeatedly reusing the same handful of signatures
    lookup = {sig_key(row): row for _, row in df.iterrows()}
    ids = list(lookup.keys())

    counts = Counter()
    for pair in existing_pairs:
        a, b = tuple(pair)
        counts[a] += 1
        counts[b] += 1

    n_needed = max(0, total_pairs - len(existing_pairs))
    rng = np.random.default_rng(42)
    new_pairs = []
    for _ in range(MAX_PAIRING_ROUNDS):
        if len(new_pairs) >= n_needed:
            break
        ordered = sorted(ids, key=lambda i: (counts[i], rng.random()))
        for a, b in zip(ordered[0::2], ordered[1::2]):
            if len(new_pairs) >= n_needed:
                break
            if a == b:
                continue
            pair = frozenset((a, b))
            if pair in existing_pairs:
                continue
            existing_pairs.add(pair)
            new_pairs.append((a, b))
            counts[a] += 1
            counts[b] += 1

    return [(lookup[a], lookup[b]) for a, b in new_pairs]


def build_pairwise_sheets(pairs_df, out_dir):
    n_per_page = PAIRS_PER_PAGE
    n_pages = int(np.ceil(len(pairs_df) / n_per_page))
    row_h = PAIR_THUMB + 22
    width = PAIR_THUMB * 2 + 60

    saved = []
    for page in range(n_pages):
        chunk = pairs_df.iloc[page * n_per_page: (page + 1) * n_per_page]
        sheet = Image.new("RGB", (width, len(chunk) * row_h), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)

        for i, (_, row) in enumerate(chunk.iterrows()):
            y0 = i * row_h
            img_a = Image.open(row["a_crop_path"]).convert("RGB")
            img_a.thumbnail((PAIR_THUMB - 10, PAIR_THUMB - 10))
            img_b = Image.open(row["b_crop_path"]).convert("RGB")
            img_b.thumbnail((PAIR_THUMB - 10, PAIR_THUMB - 10))

            tile_a = Image.new("RGB", (PAIR_THUMB, PAIR_THUMB), (255, 255, 255))
            tile_a.paste(img_a, (5, 5))
            tile_b = Image.new("RGB", (PAIR_THUMB, PAIR_THUMB), (255, 255, 255))
            tile_b.paste(img_b, (5, 5))

            sheet.paste(tile_a, (0, y0))
            sheet.paste(tile_b, (PAIR_THUMB + 60, y0))
            draw.text((PAIR_THUMB // 2 - 4, y0 + PAIR_THUMB + 4), "A", fill=(200, 0, 0))
            draw.text((PAIR_THUMB + 60 + PAIR_THUMB // 2 - 4, y0 + PAIR_THUMB + 4),
                       "B", fill=(200, 0, 0))
            draw.text((PAIR_THUMB + 8, y0 + PAIR_THUMB // 2),
                      f"#{row['comparison_id']}", fill=(0, 0, 160))

        path = out_dir / f"fluency_pairwise_sheet_{page:02d}.png"
        sheet.save(path)
        saved.append(path)
    return saved


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_signatures()
    print(f"loaded {len(df)} signature crop(s)")

    key_cols = ["a_stem", "a_column_id", "a_sig_id", "b_stem", "b_column_id", "b_sig_id"]

    if MASTER_CSV.exists():
        master = pd.read_csv(MASTER_CSV)[["comparison_id"] + key_cols]

        # a pair made before a later manual-review pass rejected one of its
        # two signatures now points at a crop that's gone -- drop it rather
        # than crash trying to render it. safe as long as no rater has
        # judged it yet; a judged stale pair is dropped too since there's
        # nothing left to show it against, but that shouldn't come up in
        # practice (review happens before comparisons are judged)
        valid = master.apply(
            lambda r: core_path_for(r.a_stem, r.a_column_id, r.a_sig_id).exists() and
                      core_path_for(r.b_stem, r.b_column_id, r.b_sig_id).exists(),
            axis=1)
        n_stale = int((~valid).sum())
        if n_stale:
            print(f"dropping {n_stale} pair(s) referencing a signature no longer "
                  f"in signatures.csv (rejected/removed since these pairs were made)")
        master = master[valid].reset_index(drop=True)

        existing_pairs = {
            frozenset(((r.a_stem, int(r.a_column_id), int(r.a_sig_id)),
                       (r.b_stem, int(r.b_column_id), int(r.b_sig_id))))
            for r in master.itertuples()
        }
        next_id = int(master["comparison_id"].max()) + 1 if not master.empty else 0
    else:
        master = pd.DataFrame(columns=["comparison_id"] + key_cols)
        existing_pairs = set()
        next_id = 0

    new_pairs = generate_new_pairs(df, existing_pairs, TOTAL_PAIRS)
    print(f"generated {len(new_pairs)} new pair(s) "
          f"(fixed budget: {TOTAL_PAIRS} total pairs, {SHEET_PAGES} sheet(s))")

    new_rows = []
    for i, (a, b) in enumerate(new_pairs):
        new_rows.append({
            "comparison_id": next_id + i,
            "a_stem": a["stem"], "a_column_id": a["column_id"], "a_sig_id": a["sig_id"],
            "b_stem": b["stem"], "b_column_id": b["column_id"], "b_sig_id": b["sig_id"],
        })

    master = pd.concat([master, pd.DataFrame(new_rows)], ignore_index=True)
    master = master.sort_values("comparison_id").reset_index(drop=True)

    # crop paths for rendering only -- not saved to any CSV
    render_df = master.copy()
    render_df["a_crop_path"] = render_df.apply(
        lambda r: str(core_path_for(r.a_stem, r.a_column_id, r.a_sig_id)), axis=1)
    render_df["b_crop_path"] = render_df.apply(
        lambda r: str(core_path_for(r.b_stem, r.b_column_id, r.b_sig_id)), axis=1)

    sheets = build_pairwise_sheets(render_df, OUT_DIR)
    print(f"saved {len(sheets)} comparison sheet(s): {sheets[0].name} .. {sheets[-1].name}"
          f"  (shared by every rater)")

    master.to_csv(MASTER_CSV, index=False, encoding="utf-8-sig")
    print(f"saved: {MASTER_CSV.name}  (master pair list, no answers)")

    print()
    for name in RATERS:
        rpath = rater_csv(name)
        prev_margins = {}
        if rpath.exists():
            prev = pd.read_csv(rpath)
            prev_margins = dict(zip(prev["comparison_id"], prev["human_margin"]))

        rater_df = master.copy()
        rater_df["human_margin"] = rater_df["comparison_id"].map(prev_margins)
        rater_df.to_csv(rpath, index=False, encoding="utf-8-sig")

        n_done = rater_df["human_margin"].notna().sum()
        print(f"saved: {rpath.name}  ({n_done} / {len(rater_df)} judged so far)")

    print(f"\nsend each rater the sheet(s) in {OUT_DIR} plus *only* their own "
          f"fluency_pairwise_<name>.csv -- don't share CSVs between raters, "
          f"that defeats the point of judging blind. each fills in "
          f"human_margin (-4..4: positive favours A, negative favours B, 0 = "
          f"tied) for every comparison_id, then human_fluency_cluster.py "
          f"pools everyone's answers once at least one CSV has some filled in.")

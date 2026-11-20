import argparse
import hmac
import json
import os
import re
import secrets
import socket
import threading
import webbrowser
from collections import Counter
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote, parse_qs

import numpy as np
import pandas as pd

import roles
from fluency_compare import (
    CORE_DIR, OUT_DIR, MAX_PAIRING_ROUNDS,
    load_all_signatures, core_path_for,
)

GENEALOGY_DIR    = Path("genealogy")
PEOPLE_CROPS_DIR = GENEALOGY_DIR / "person_crops"
PEOPLE_CROPS_CSV = GENEALOGY_DIR / "person_crops.csv"
PERSON_SCORES_CSV = GENEALOGY_DIR / "person_genealogy_scores.csv"
ANON_SALT_FILE   = GENEALOGY_DIR / ".anon_salt"

# pair design. new pairs either link two separate parts of the comparison
# graph (so one scale can be fitted) or contrast a person trait the
# analysis tests (experiments/human_fluency_traits.py), size-matched so box
# size does not do the work. raters never see any of this.
CONTRAST_TRAITS = ("scribe", "own_hand", "has_title")
CONTRAST_SHARE  = 0.5
# every REPEAT_EVERY-th pair a rater sees is one someone else already judged,
# for agreement; the rest are new
REPEAT_EVERY    = 10
PAIR_RNG        = np.random.default_rng(42)


SIGNATURES_DIR   = Path("signatures")
REVIEW_DIR       = Path("review")
VERDICTS_CSV     = REVIEW_DIR / "manual_verdicts.csv"
VERDICT_GLOB     = "manual_verdicts*.csv"
STEMS_CSV        = Path("database") / "processed" / "pipeline_stems.csv"

SEED = 42

DUPLICATE_IOU = 0.3


FLUENCY_LEGEND = {
    0: "circles",
    1: "very low fluency",
    2: "low fluency",
    3: "moderate fluency",
    4: "high fluency",
}

CORPUS = None

CSV_LOCK = threading.Lock()


def sanitize_name(name):
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip())
    return name or "anon"


def lan_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "<this machine's ip>"
    finally:
        s.close()


def write_csv(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


class Corpus:
    name     = ""
    prefix   = ""
    id_cols  = []
    label    = ""
    show_tag = True

    def __init__(self):
        self.items = self.load_items()
        self.by_id = {r["item_id"]: r for _, r in self.items.iterrows()}
        self.traits = item_traits(self.items)

    def load_items(self):
        raise NotImplementedError

    def item_id_from(self, values):
        raise NotImplementedError

    def crop_url(self, item_id):
        raise NotImplementedError

    def image_path(self, key):
        raise NotImplementedError

    def tag(self, item_id):
        return ""


    @property
    def pair_cols(self):
        return ["comparison_id"] + [f"{side}_{c}"
                                    for side in ("a", "b") for c in self.id_cols]

    @property
    def master_csv(self):
        return OUT_DIR / f"{self.prefix}_pairwise.csv"

    def pairwise_csv(self, rater):
        return OUT_DIR / f"{self.prefix}_pairwise_{rater}.csv"

    def labels_csv(self, rater):
        return OUT_DIR / f"{self.prefix}_labels_{rater}.csv"

    @property
    def raters_csv(self):
        return OUT_DIR / f"{self.prefix}_raters.csv"

    @property
    def design_csv(self):
        return OUT_DIR / f"{self.prefix}_pair_design.csv"

    def pair_row(self, comparison_id, a_id, b_id):
        row = {"comparison_id": comparison_id}
        for side, item_id in (("a", a_id), ("b", b_id)):
            item = self.by_id[item_id]
            for c in self.id_cols:
                row[f"{side}_{c}"] = item[c]
        return row

    def pair_item_ids(self, row):
        return tuple(self.item_id_from([row[f"{side}_{c}"] for c in self.id_cols])
                     for side in ("a", "b"))


class CoreCorpus(Corpus):
    name     = "core"
    prefix   = "fluency"
    id_cols  = ["stem", "column_id", "sig_id"]
    label    = "signature crops"
    show_tag = True

    def load_items(self):
        df = load_all_signatures().reset_index(drop=True)
        df["item_id"] = [self.item_id_from((r.stem, r.column_id, r.sig_id))
                         for r in df.itertuples()]
        df["group_id"] = df["item_id"]
        return df[["item_id", "group_id", "crop_path"] + self.id_cols]

    def item_id_from(self, values):
        stem, column_id, sig_id = values
        return f"{stem}|{int(column_id)}|{int(sig_id)}"

    def crop_url(self, item_id):
        stem, column_id, sig_id = item_id.split("|")
        return f"/image/core/{core_path_for(stem, column_id, sig_id).name}"

    def image_path(self, key):
        if "/" in key or "\\" in key:
            return None
        return CORE_DIR / key

    def tag(self, item_id):
        stem, column_id, sig_id = item_id.split("|")
        return f"{stem} col{int(column_id)} sig{int(sig_id)}"


class AnonCorpus(Corpus):
    id_cols  = ["anon_id"]
    show_tag = False
    sources  = ()

    def __init__(self, limit=None, one_per_person=False):
        self.limit = limit
        self.one_per_person = one_per_person
        super().__init__()

    def load_items(self):
        df = pd.concat([SOURCE_LOADERS[s]() for s in self.sources], ignore_index=True)
        if self.one_per_person:
            df = one_crop_each(df)
        df = group_duplicate_ink(df)

        salt = load_anon_salt()
        df["anon_id"] = [anon_id(salt, k) for k in df["key"]]
        if not df["anon_id"].is_unique:
            raise SystemExit("anon id collision -- two crops hashed alike")

        df = self.cap_pool(df)

        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        write_anon_map(self, df)

        df["item_id"] = df["anon_id"]
        if {"x0", "y0", "x1", "y1"} <= set(df.columns):
            num = lambda c: pd.to_numeric(df[c], errors="coerce")
            df["area"] = (num("x1") - num("x0")) * (num("y1") - num("y0"))
        extra = [c for c in ("genealogy_code", "area") if c in df.columns]
        return df[["item_id", "group_id", "crop_path", "source"]
                  + extra + self.id_cols]

    def judged_ids(self):
        """anon ids that already appear in a comparison, so a smaller pool
        never orphans work already done."""
        ids = set()
        for path in [self.master_csv] + sorted(
                OUT_DIR.glob(f"{self.prefix}_pairwise_*.csv")):
            if not path.exists():
                continue
            d = pd.read_csv(path, encoding="utf-8-sig")
            for col in ("a_anon_id", "b_anon_id"):
                if col in d.columns:
                    ids |= set(d[col].dropna().astype(str))
        for path in sorted(OUT_DIR.glob(f"{self.prefix}_labels_*.csv")):
            d = pd.read_csv(path, encoding="utf-8-sig")
            if "anon_id" in d.columns and "human_fluency" in d.columns:
                ids |= set(d.loc[d["human_fluency"].notna(), "anon_id"].astype(str))
        return ids

    def cap_pool(self, df):
        """hold the pool to --limit crops.

        a ranking only becomes a scale once the comparison graph connects, and
        with thousands of crops the coverage-balancing generator keeps drawing
        unseen ones, so every pair stays an island. a smaller pool is what makes
        the same judgement budget connect.

        kept regardless of the cap: crops already judged (dropping them would
        strand those judgements). the rest is filled per source in the
        existing proportions,
        on a fixed seed so the pool is the same on every restart."""
        if self.limit is None or len(df) <= self.limit:
            return df

        keep = df["anon_id"].isin(self.judged_ids())
        held = df[keep]
        rest = df[~keep]
        room = self.limit - len(held)
        if room <= 0:
            if len(held) > self.limit:
                print(f"note: {len(held)} crops are already judged, "
                      f"which is over the --limit of {self.limit}; keeping them all")
            return held

        share = rest["source"].value_counts(normalize=True)
        rng = np.random.default_rng(SEED)
        picked = []
        for source, frac in share.items():
            pool = rest[rest["source"] == source]
            n = min(len(pool), int(round(room * frac)))
            if n:
                picked.append(pool.sample(n=n, random_state=int(rng.integers(1 << 31))))
        out = pd.concat([held] + picked, ignore_index=True) if picked else held

        # rounding can land a crop or two short of the cap
        if len(out) < self.limit:
            spare = rest[~rest["anon_id"].isin(set(out["anon_id"]))]
            if len(spare):
                out = pd.concat([out, spare.sample(
                    n=min(len(spare), self.limit - len(out)),
                    random_state=SEED)], ignore_index=True)
        return out

    def item_id_from(self, values):
        return str(values[0])

    def crop_url(self, item_id):
        return f"/image/{self.name}/{item_id}.png"

    def image_path(self, key):
        item = self.by_id.get(key[:-4] if key.endswith(".png") else key)
        return Path(item["crop_path"]) if item is not None else None


class PeopleCorpus(AnonCorpus):
    name    = "people"
    prefix  = "people"
    label   = "genealogy person crops (anonymised)"
    sources = ("person",)


class AllCorpus(AnonCorpus):
    name    = "all"
    prefix  = "all"
    label   = "all approved crops"
    sources = ("person", "signature")


CORPORA = {c.name: c for c in (CoreCorpus, PeopleCorpus, AllCorpus)}


# hosted copy (PythonAnywhere). the pool is frozen at export time so the
# server needs none of the genealogy data: no salt, no names, no anon map,
# just crops renamed to their anon id plus precomputed pairing traits.
BUNDLE_POOL   = "pool.csv"
BUNDLE_CROPS  = "crops"
TRAIT_COLS    = ("log_area",) + CONTRAST_TRAITS


class FrozenCorpus(PeopleCorpus):
    def __init__(self, root):
        self.root = Path(root)
        self.limit, self.one_per_person = None, False
        pool = pd.read_csv(self.root / BUNDLE_POOL, encoding="utf-8-sig")
        pool["crop_path"] = [str(self.root / BUNDLE_CROPS / f"{a}.png")
                             for a in pool["anon_id"]]
        self.items = pool[["item_id", "group_id", "crop_path", "source"]
                          + self.id_cols].copy()
        self.by_id = {r["item_id"]: r for _, r in self.items.iterrows()}
        self.traits = pool.set_index("item_id")[
            [c for c in TRAIT_COLS if c in pool.columns]]


def export_bundle(corpus, out):
    """write what the hosted server needs to `out`: code, the pool with
    traits, crops named by anon id, and the current people_* rating files
    (not the anon map)."""
    out = Path(out)
    (out / BUNDLE_CROPS).mkdir(parents=True, exist_ok=True)
    (out / OUT_DIR).mkdir(parents=True, exist_ok=True)

    items = corpus.items
    pool = items[["item_id", "group_id", "source"] + corpus.id_cols].copy()
    # group ids carry the genealogy code; hash them so the bundle holds no
    # person identifiers. only equality matters to the pair generator.
    salt = load_anon_salt()
    pool["group_id"] = [anon_id(salt, "group:" + str(g)) for g in pool["group_id"]]
    t = corpus.traits.reindex(pool["item_id"])
    for c in t.columns:
        pool[c] = t[c].to_numpy()
    pool.to_csv(out / BUNDLE_POOL, index=False, encoding="utf-8-sig")

    for r in items.itertuples():
        dst = out / BUNDLE_CROPS / f"{r.anon_id}.png"
        dst.write_bytes(Path(r.crop_path).read_bytes())

    for path in OUT_DIR.glob(f"{corpus.prefix}_*.csv"):
        if path.name != anon_map_csv(corpus).name:
            (out / OUT_DIR / path.name).write_bytes(path.read_bytes())

    here = Path(__file__).parent
    for name in ("fluency_game.py", "fluency_compare.py", "roles.py",
                 "fluency_wsgi.py") + STATIC_WEBP:
        (out / name).write_bytes((here / name).read_bytes())
    print(f"bundle: {len(pool)} crops -> {out}")


def load_person_source():
    if not PEOPLE_CROPS_CSV.exists():
        raise SystemExit(f"no {PEOPLE_CROPS_CSV} -- run review_people.py first")
    df = pd.read_csv(PEOPLE_CROPS_CSV, encoding="utf-8-sig")

    for flag in ("empty", "seal"):
        if flag in df.columns:
            df = df[pd.to_numeric(df[flag], errors="coerce").fillna(0) != 1]

    df = df[df["crop"].notna()]
    df = df.assign(crop_path=[str(PEOPLE_CROPS_DIR / c) for c in df["crop"]])
    df = df[[Path(p).exists() for p in df["crop_path"]]]
    if df.empty:
        raise SystemExit(f"no usable crops in {PEOPLE_CROPS_DIR}")

    df = df.copy()
    df["source"] = "person"
    df["key"] = ["person:" + str(c) for c in df["crop_id"]]
    df["group_id"] = ["person:" + str(c) for c in df["genealogy_code"]]
    return df[["key", "source", "crop_path", "group_id", "crop_id",
               "genealogy_code", "genealogy_name", "contract_id", "x0", "y0", "x1", "y1"]]


def load_approved_verdicts():
    paths = sorted(REVIEW_DIR.glob(VERDICT_GLOB), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise SystemExit(f"no {VERDICTS_CSV} -- run review_signatures.py first")
    verdicts = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in paths],
                         ignore_index=True)
    verdicts = verdicts.drop_duplicates(subset=["stem", "column_id", "sig_id"], keep="last")
    return verdicts[verdicts["verdict"] == "accept"]


def load_signature_source():
    acc = load_approved_verdicts().copy()
    acc["column_id"] = acc["column_id"].astype(int)
    acc["sig_id"] = acc["sig_id"].astype(int)
    acc["crop_path"] = [str(core_path_for(r.stem, r.column_id, r.sig_id))
                        for r in acc.itertuples()]
    acc = acc[[Path(p).exists() for p in acc["crop_path"]]]
    if acc.empty:
        raise SystemExit("no approved signature crops left in signatures/core/")

    acc["source"] = "signature"
    acc["key"] = [f"signature:{r.stem}|{r.column_id}|{r.sig_id}" for r in acc.itertuples()]
    acc["group_id"] = acc["key"]
    acc["contract_id"] = acc["stem"].map(stem_contract_ids())
    boxes = signature_boxes(acc)
    for c in ("x0", "y0", "x1", "y1"):
        acc[c] = [boxes.get((r.stem, r.column_id, r.sig_id), {}).get(c) for r in acc.itertuples()]
    return acc[["key", "source", "crop_path", "group_id", "stem", "column_id",
                "sig_id", "contract_id", "x0", "y0", "x1", "y1"]]


def one_crop_each(df):
    """keep a single crop per person: the largest, on the grounds that the
    biggest box is the least likely to be a clipped fragment.

    every comparison then puts two different hands side by side, which is what
    the ranking is actually asking about -- with several crops per person the
    generator could otherwise spend judgements on one hand against itself, and
    a person with 133 crops would crowd out one with a single crop."""
    if "group_id" not in df.columns:
        return df
    df = df.copy()
    df["_area"] = (pd.to_numeric(df["x1"], errors="coerce")
                   - pd.to_numeric(df["x0"], errors="coerce")) * \
                  (pd.to_numeric(df["y1"], errors="coerce")
                   - pd.to_numeric(df["y0"], errors="coerce"))
    df = (df.sort_values("_area", ascending=False)
            .drop_duplicates("group_id", keep="first")
            .drop(columns="_area"))
    return df.reset_index(drop=True)


SOURCE_LOADERS = {"person": load_person_source, "signature": load_signature_source}


def stem_contract_ids():
    if not STEMS_CSV.exists():
        return {}
    stems = pd.read_csv(STEMS_CSV, encoding="utf-8-sig")
    return dict(zip(stems["stem"], stems["contract_id"]))


def signature_boxes(acc):
    boxes = {}
    for stem in acc["stem"].unique():
        path = SIGNATURES_DIR / f"{stem}_signatures.csv"
        if not path.exists():
            continue
        for r in pd.read_csv(path, encoding="utf-8-sig").itertuples():
            boxes[(stem, int(r.column_id), int(r.sig_id))] = {
                "x0": r.x0, "y0": r.y0, "x1": r.x1, "y1": r.y1}
    return boxes


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = lambda c: (c[2] - c[0]) * (c[3] - c[1])
    return inter / (area(a) + area(b) - inter)


def group_duplicate_ink(df):
    if df["source"].nunique() < 2 or "contract_id" not in df.columns:
        return df

    group = dict(zip(df["key"], df["group_id"]))
    boxed = df[df[["x0", "y0", "x1", "y1"]].notna().all(axis=1) & df["contract_id"].notna()]
    for _, rows in boxed.groupby("contract_id"):
        people = rows[rows["source"] == "person"]
        sigs = rows[rows["source"] == "signature"]
        for p in people.itertuples():
            for s in sigs.itertuples():
                if iou((p.x0, p.y0, p.x1, p.y1), (s.x0, s.y0, s.x1, s.y1)) >= DUPLICATE_IOU:
                    group[s.key] = group[p.key]

    return df.assign(group_id=[group[k] for k in df["key"]])


def load_anon_salt():
    if ANON_SALT_FILE.exists():
        salt = ANON_SALT_FILE.read_text(encoding="utf-8").strip()
        if salt:
            return salt.encode("utf-8")
    GENEALOGY_DIR.mkdir(parents=True, exist_ok=True)
    salt = secrets.token_hex(32)
    ANON_SALT_FILE.write_text(salt, encoding="utf-8")
    print(f"made a new anonymisation salt at {ANON_SALT_FILE} -- keep it, "
          f"without it judged anon ids cannot be joined back to crops")
    return salt.encode("utf-8")


def anon_id(salt, key):
    digest = hmac.new(salt, str(key).encode("utf-8"), sha256).hexdigest()
    return "s" + digest[:10]


def anon_map_csv(corpus):
    return OUT_DIR / f"{corpus.prefix}_anon_map.csv"


def write_anon_map(corpus, df):
    cols = [c for c in ("anon_id", "source",
                        "crop_id", "genealogy_code", "genealogy_name",
                        "contract_id", "stem", "column_id", "sig_id",
                        "crop_path") if c in df.columns]
    df[cols].sort_values(["source", "anon_id"]).to_csv(
        anon_map_csv(corpus), index=False, encoding="utf-8-sig")


def load_master():
    if CORPUS.master_csv.exists():
        return pd.read_csv(CORPUS.master_csv, encoding="utf-8-sig")[CORPUS.pair_cols]
    return pd.DataFrame(columns=CORPUS.pair_cols)


def existing_pairs_from(master):
    return {frozenset(CORPUS.pair_item_ids(row)) for _, row in master.iterrows()}


def item_traits(items):
    """per-crop pairing covariates: log box area, and the person traits in
    CONTRAST_TRAITS (1 held, 0 never, NaN unknown). roles come from the
    contract rosters, so a person with no roster roles is unknown, not 0."""
    t = pd.DataFrame(index=pd.Index(items["item_id"], name="item_id"))
    area = pd.to_numeric(items["area"], errors="coerce").to_numpy() \
        if "area" in items.columns else np.full(len(items), np.nan)
    t["log_area"] = np.log(np.where(area > 0, area, np.nan))
    if "genealogy_code" not in items.columns or not PERSON_SCORES_CSV.exists():
        return t

    g = pd.read_csv(PERSON_SCORES_CSV, encoding="utf-8-sig").set_index("genealogy_code")
    held = {}
    for code, text in g["roles"].items():
        if isinstance(text, str):
            held[code] = {f for chunk in text.split("|")
                          for tok in roles.split_roles(chunk)
                          for f in roles.token_families(tok)}
    codes = items["genealogy_code"].to_numpy()
    for fam in ("scribe", "own_hand"):
        t[fam] = [float(fam in held[c]) if c in held else np.nan for c in codes]
    t["has_title"] = pd.Series(codes).map(g["has_title"]).to_numpy(dtype=float)
    return t


def graph_components(ids, pairs):
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for pair in pairs:
        a, b = tuple(pair)
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    return {i: find(i) for i in ids}


def generate_new_pairs(existing_pairs, n_needed):
    """returns [(a, b, kind, trait)].

    the old generator always paired the two least-seen crops, so while unseen
    crops were left every pair was two fresh crops: 213 islands, no shared
    scale. now each pair takes the least-seen crop and gives it a partner from
    a *different* component of the comparison graph, so every pair merges two
    components until the graph is one piece; after that it just balances
    exposure.

    CONTRAST_SHARE of pairs are trait contrasts: a holder of the trait with
    the fewest contrasting pairs so far, against a known non-holder whose box
    size is as close as possible."""
    items = CORPUS.items
    ids = list(items["item_id"])
    groups = dict(zip(ids, items["group_id"]))
    traits = CORPUS.traits
    log_area = traits["log_area"].to_dict()

    degree = Counter(i for pair in existing_pairs for i in pair)
    comp = graph_components(ids, existing_pairs)

    usable = [t for t in CONTRAST_TRAITS if t in traits.columns
              and (traits[t] == 1).any() and (traits[t] == 0).any()]
    trait_of = {t: traits[t].to_dict() for t in usable}
    contrasts = Counter()
    for pair in existing_pairs:
        a, b = tuple(pair)
        for t in usable:
            if {trait_of[t].get(a), trait_of[t].get(b)} == {0.0, 1.0}:
                contrasts[t] += 1

    def size_gap(a, b):
        gap = abs(log_area.get(a, np.nan) - log_area.get(b, np.nan))
        return 1.0 if np.isnan(gap) else gap

    new_pairs = []
    for _ in range(max(1, n_needed) * MAX_PAIRING_ROUNDS):
        if len(new_pairs) >= n_needed:
            break
        if usable and PAIR_RNG.random() < CONTRAST_SHARE:
            kind = "contrast"
            trait = min(usable, key=lambda t: (contrasts[t], PAIR_RNG.random()))
            a_pool = [i for i in ids if trait_of[trait][i] == 1]
            b_pool = [i for i in ids if trait_of[trait][i] == 0]
        else:
            kind, trait = "link", ""
            a_pool = b_pool = ids

        a = min(a_pool, key=lambda i: (degree[i], PAIR_RNG.random()))
        partners = [b for b in b_pool
                    if b != a and groups[b] != groups[a]
                    and frozenset((a, b)) not in existing_pairs]
        if not partners:
            continue

        def cost(b):
            same = 2.0 if comp[b] == comp[a] else 0.0
            if kind == "contrast":
                return same + size_gap(a, b) + 0.25 * degree[b] + 0.01 * PAIR_RNG.random()
            return same + degree[b] + PAIR_RNG.random()

        b = min(partners, key=cost)
        if PAIR_RNG.random() < 0.5:      # holder is not always on the left
            a, b = b, a

        existing_pairs.add(frozenset((a, b)))
        degree[a] += 1
        degree[b] += 1
        old, new = comp[b], comp[a]
        if old != new:
            for i, c in comp.items():
                if c == old:
                    comp[i] = new
        if kind == "contrast":
            contrasts[trait] += 1
        new_pairs.append((a, b, kind, trait))

    return new_pairs


def log_design(rows):
    path = CORPUS.design_csv
    prev = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None
    add = pd.DataFrame(rows)
    write_csv(add if prev is None else pd.concat([prev, add], ignore_index=True), path)


def judgement_counts():
    """judgements per comparison, over every rater file"""
    counts = Counter()
    for path in OUT_DIR.glob(f"{CORPUS.prefix}_pairwise_*.csv"):
        d = pd.read_csv(path, encoding="utf-8-sig")
        if "human_margin" in d.columns:
            counts.update(d.loc[d["human_margin"].notna(), "comparison_id"].astype(int))
    return counts


# in-memory, per rater: the pair last handed out (so a reload shows the same
# pair) and pairs cleared by undo (so "back" brings the previous pair back).
# lost on restart, which only means a fresh pair is served.
SERVED = {}
UNDONE = {}


def load_rater_pairwise(name, master):
    rpath = CORPUS.pairwise_csv(name)
    if rpath.exists():
        rater_df = pd.read_csv(rpath, encoding="utf-8-sig")
    else:
        rater_df = pd.DataFrame(columns=CORPUS.pair_cols + ["human_margin"])

    missing = master[~master["comparison_id"].isin(rater_df["comparison_id"])]
    if not missing.empty:
        add = missing.copy()
        add["human_margin"] = np.nan
        rater_df = pd.concat([rater_df, add], ignore_index=True)

    return rater_df.sort_values("comparison_id").reset_index(drop=True)


def pick_existing(name, open_ids, n_judged):
    """an existing comparison this rater still has open, or None to make a new
    one. order: a pair they just undid, the pair they were last shown, a pair
    nobody has judged yet (and nobody else is looking at), then on every
    REPEAT_EVERY-th turn the least-judged, newest pair for agreement."""
    stack = UNDONE.get(name, [])
    while stack:
        cid = stack.pop()
        if cid in open_ids:
            return cid
    if SERVED.get(name) in open_ids:
        return SERVED[name]
    if not open_ids:
        return None
    counts = judgement_counts()
    elsewhere = {c for r, c in SERVED.items() if r != name}
    fresh = sorted(c for c in open_ids if counts[c] == 0 and c not in elsewhere)
    if fresh:
        return fresh[0]
    if n_judged % REPEAT_EVERY == REPEAT_EVERY - 1:
        return min(open_ids, key=lambda c: (counts[c], -c))
    return None


def next_pair_for(name):
    with CSV_LOCK:
        master = load_master()
        rater_df = load_rater_pairwise(name, master)
        n_judged = int(rater_df["human_margin"].notna().sum())
        open_ids = set(rater_df.loc[rater_df["human_margin"].isna(),
                                    "comparison_id"].astype(int))

        cid = pick_existing(name, open_ids, n_judged)
        if cid is None:
            existing = existing_pairs_from(master)
            new = generate_new_pairs(existing, 1)
            if new:
                a, b, kind, trait = new[0]
                cid = int(master["comparison_id"].max()) + 1 if not master.empty else 0
                row = CORPUS.pair_row(cid, a, b)
                master = pd.concat([master, pd.DataFrame([row])], ignore_index=True)
                write_csv(master, CORPUS.master_csv)
                log_design([{"comparison_id": cid, "kind": kind, "trait": trait,
                             "created_for": name,
                             "created": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}])
                rater_df = pd.concat(
                    [rater_df, pd.DataFrame([{**row, "human_margin": np.nan}])],
                    ignore_index=True)
                write_csv(rater_df, CORPUS.pairwise_csv(name))
            elif open_ids:
                # pool exhausted: fall back to any open pair
                counts = judgement_counts()
                cid = min(open_ids, key=lambda c: (counts[c], -c))
            else:
                return None, len(master), n_judged

        SERVED[name] = cid
        chosen = rater_df[rater_df["comparison_id"] == cid].iloc[0].to_dict()
        return chosen, len(master), n_judged


def save_margin(name, comparison_id, margin):
    with CSV_LOCK:
        master = load_master()
        rater_df = load_rater_pairwise(name, master)
        rater_df.loc[rater_df["comparison_id"] == comparison_id, "human_margin"] = margin
        write_csv(rater_df, CORPUS.pairwise_csv(name))


def clear_margin(name, comparison_id):
    """undo one comparison. the pair stays in the master and in this rater's
    file, only the margin goes back to blank, so next_pair_for hands the same
    comparison out again rather than generating a fresh one."""
    with CSV_LOCK:
        master = load_master()
        rater_df = load_rater_pairwise(name, master)
        rater_df.loc[rater_df["comparison_id"] == comparison_id, "human_margin"] = np.nan
        write_csv(rater_df, CORPUS.pairwise_csv(name))
        UNDONE.setdefault(name, []).append(int(comparison_id))


RATER_COLS = ["rater", "age", "first_language", "education",
              "chinese_familiarity", "recorded"]

FAMILIARITY_MAX = 5


def load_raters():
    if CORPUS.raters_csv.exists():
        df = pd.read_csv(CORPUS.raters_csv, encoding="utf-8-sig")
        for col in RATER_COLS:
            if col not in df.columns:
                df[col] = np.nan
        return df[RATER_COLS]
    return pd.DataFrame(columns=RATER_COLS)


def get_profile(name):
    """what this rater told us about themselves, or None if they have not been
    through the intake yet. a name is the only thing asked for, so any saved
    row counts as done."""
    df = load_raters()
    row = df[df["rater"] == name]
    if row.empty:
        return None
    rec = row.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in rec.items()}


def clean_profile(body):
    """validate one intake submission. only the name is required, and that is
    the rater key rather than a field here -- everything below is optional, so
    a blank is always an acceptable answer."""
    fam = body.get("chinese_familiarity")
    if fam in (None, "", "null"):
        fam = ""
    else:
        try:
            fam = int(fam)
        except (TypeError, ValueError):
            return None, "familiarity must be a number"
        if not 0 <= fam <= FAMILIARITY_MAX:
            return None, f"familiarity must be 0-{FAMILIARITY_MAX}"

    age = body.get("age")
    if age in (None, "", "null"):
        age = ""
    else:
        try:
            age = int(age)
        except (TypeError, ValueError):
            return None, "age must be a whole number"
        if not 0 < age < 120:
            return None, "age looks wrong"

    return {
        "age": age,
        "first_language": str(body.get("first_language") or "").strip()[:60],
        "education": str(body.get("education") or "").strip()[:80],
        "chinese_familiarity": fam,
    }, None


def save_profile(name, profile):
    with CSV_LOCK:
        df = load_raters()
        df = df[df["rater"] != name]
        row = {"rater": name, "recorded": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
               **profile}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        write_csv(df.sort_values("rater")[RATER_COLS], CORPUS.raters_csv)


def rate_template():
    df = CORPUS.items.reset_index(drop=True)
    df = df.assign(example_id=df.index)
    return df[["example_id", "item_id", "crop_path"] + CORPUS.id_cols]


def load_rater_labels(name):
    template = rate_template()
    lpath = CORPUS.labels_csv(name)
    if lpath.exists():
        prev = pd.read_csv(lpath, encoding="utf-8-sig")
        if "human_fluency" in prev.columns:
            prev = prev[CORPUS.id_cols + ["human_fluency"]].copy()
            for c in CORPUS.id_cols:
                prev[c] = prev[c].astype(template[c].dtype)
            template = template.merge(prev, on=CORPUS.id_cols, how="left")
    if "human_fluency" not in template.columns:
        template["human_fluency"] = np.nan
    return template


def next_single_for(name):
    rater_df = load_rater_labels(name)
    unrated = rater_df[rater_df["human_fluency"].isna()]
    n_judged = int(rater_df["human_fluency"].notna().sum())
    if unrated.empty:
        return None, len(rater_df), n_judged
    return unrated.iloc[0].to_dict(), len(rater_df), n_judged


def save_score(name, example_id, score):
    with CSV_LOCK:
        rater_df = load_rater_labels(name)
        rater_df.loc[rater_df["example_id"] == example_id, "human_fluency"] = score
        write_csv(rater_df[["example_id"] + CORPUS.id_cols + ["human_fluency"]],
                  CORPUS.labels_csv(name))


def clear_score(name, example_id):
    """undo one rating. blanking human_fluency is enough: next_single_for
    serves the first unrated row, so the crop comes straight back."""
    with CSV_LOCK:
        rater_df = load_rater_labels(name)
        rater_df.loc[rater_df["example_id"] == example_id, "human_fluency"] = np.nan
        write_csv(rater_df[["example_id"] + CORPUS.id_cols + ["human_fluency"]],
                  CORPUS.labels_csv(name))


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signature fluency rankings 花押流畅度评分</title>
<style>
  :root {
    color-scheme: light;
    --bg: #99b3ff; --ink: #000; --muted: #333; --faint: #555;
    --line: #000; --sel-bg: #000; --sel-ink: #fff; --paper: #99b3ff;
    --ok-bg: #e6ff99; --ok-ink: #fff;
    --accent: #ff99ff; --accent-ink: #e6ff99;
    --header-ink: #e6ff99;
    --btn: #fff;
  }
  * { font-family: "Times New Roman", Times, "Noto Serif CJK SC", "Source Han Serif SC",
       "Songti SC", SimSun, "Microsoft YaHei", serif; box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body { margin: 0; background: var(--bg); color: var(--ink); overflow-x: hidden; }
  img { max-width: 100%; }
  button { touch-action: manipulation; }

  header { position: sticky; top: 0; z-index: 5; background: var(--accent); color: var(--header-ink); padding: 12px 16px;
           display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
           border-bottom: 2px dashed var(--header-ink); }
  header h1 { font-size: 20px; margin: 0; font-weight: 700; letter-spacing: .02em; }
  .stats { font-size: 14px; display: flex; gap: 12px; }

  .txt { background: none; border: 0; margin: 0; padding: 6px 2px; min-height: 32px;
         font-size: 13px; color: var(--muted); cursor: pointer; font-family: inherit;
         line-height: 1.2; }
  .txt:hover { color: var(--accent); }
  .txt.active { color: var(--accent); }

  .name-box { display: flex; gap: 8px; align-items: center; font-size: 13px; }
  .name-box input { width: 130px; background: none; color: var(--ink); font-size: 16px;
        border: 0; border-bottom: 1px solid var(--line); border-radius: 0; padding: 3px 1px;
        font-family: inherit; }
  .name-box input::placeholder { color: var(--faint); }
  .name-box input:focus { outline: none; border-bottom-color: var(--accent);
        border-bottom-width: 2px; }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .fam { display: flex; gap: 6px; align-items: center; font-size: 13px; color: var(--muted);
         flex-wrap: wrap; }
  .fam .famlabel { max-width: 280px; line-height: 1.2; }
  .fam .txt { min-width: 30px; text-align: center; }
  .modes { display: flex; gap: 14px; align-items: center; }

  main { max-width: 900px; margin: 0 auto; padding: 30px 20px 60px; text-align: center; }
  .zh { opacity: .72; }

  header .txt, header .stats { color: var(--header-ink); }
  header .txt.active, header .txt:hover, header .txt:active { color: var(--ink); }
  header .zh { opacity: 1; }

  .intake { max-width: 460px; margin: 26px auto 0; text-align: left; }
  .intake h2 { font-size: 20px; margin: 0 0 6px; }
  .intake .intro { font-size: 15px; line-height: 1.5; margin: 0 0 12px; }
  .intake .intro:first-of-type { margin-top: 18px; }
  .intake .intro:last-of-type { margin-bottom: 18px; }
  .scale-img { display: block; width: 100%; max-width: 560px; height: auto; margin: 16px auto 0; padding: 0 16px; box-sizing: border-box; }
  .scale-img + main { padding-top: 0; }
  .scale-img + main .intake { margin-top: 0; }
  .scale-img + main .pair, .scale-img + main .single { margin-top: 16px; }
  .intake .lede { font-size: 13px; color: var(--muted); margin: 0 0 22px; line-height: 1.45; }
  .intake label { display: block; font-size: 14px; margin-bottom: 18px; }
  .intake .req { font-size: 11px; color: var(--accent); margin-left: 6px; }
  .intake .help { display: block; font-size: 12px; color: var(--faint); margin-top: 4px; }
  .intake input[type="text"], .intake input[type="number"], .intake select {
        display: block; width: 100%; margin-top: 5px; font-size: 16px; padding: 6px 2px;
        background: none; color: var(--ink); border: 0; border-bottom: 1px solid var(--line);
        border-radius: 0; font-family: inherit; }
  .intake input:focus, .intake select:focus { outline: none;
        border-bottom-color: var(--accent); border-bottom-width: 2px; }
  .scale { display: flex; gap: 6px; margin-top: 6px; }
  .scale button { width: 34px; min-height: 34px; background: var(--btn); color: var(--ink);
        border: 1px solid transparent; border-radius: 0; font-size: 14px;
        cursor: pointer; font-family: inherit; }
  .scale button.on, .scale button:hover, .scale button:active { background: var(--accent); color: var(--accent-ink); }
  .intake .go { margin-top: 6px; padding: 11px 26px; font-size: 15px;
        background: var(--btn); color: var(--ink); border: 1px solid transparent;
        border-radius: 0; cursor: pointer; font-family: inherit; }
  .intake .go:hover, .intake .go:active { background: var(--accent); color: var(--accent-ink); }
  .intake .err { font-size: 13px; color: #c0392b; min-height: 18px; margin-bottom: 4px; }

  .pair { display: flex; justify-content: center; gap: 40px; align-items: flex-start;
          margin-bottom: 26px; }
  .side { display: flex; flex-direction: column; align-items: center; gap: 8px; }
  .side img { width: 18vw; height: auto; background: none; }
  .side .tag { font-size: 12px; color: var(--muted); }
  .side .letter { font-size: 20px; font-weight: 700; color: var(--accent); }

  .single img { width: 26vw; height: auto; background: none; display: block;
                margin: 0 auto 8px; }
  .single .tag { font-size: 12px; color: var(--muted); margin-bottom: 24px; }

  .choices { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
  .choices button { background: var(--btn); color: var(--ink); border: 1px solid transparent;
        border-radius: 0; padding: 10px 16px; font-size: 14px; cursor: pointer;
        font-family: inherit; }
  .choices button:hover, .choices button:active { background: var(--accent); color: var(--accent-ink); }
  .choices.compare { gap: 8px; }
  .choices button .zh { display: block; font-size: .9em; }
  .choices.compare button { font-size: 12.5px; padding: 9px 11px; white-space: normal; }
  .choices button.flash { background: var(--accent); color: var(--accent-ink); }

  .hint { color: var(--faint); font-size: 12px; margin-top: 18px; }
  /* deliberately quieter than the answer buttons: undo is the rare path and
     should not compete with the five choices for a misclick of its own. */
  .backrow { margin-top: 22px; }
  .back { background: var(--btn); color: var(--ink); border: 1px solid transparent;
        border-radius: 3px; padding: 6px 14px; font-family: inherit;
        font-size: 13px; cursor: pointer; }
  .back:hover:not(:disabled), .back:active:not(:disabled) { background: var(--accent); color: var(--accent-ink); }
  .back:disabled { opacity: .38; cursor: default; }
  .back .zh { margin-left: 6px; }
  .done { font-size: 15px; margin-top: 60px; }

  .goal { display: flex; align-items: center; gap: 8px; font-size: 13px; white-space: nowrap; }
  /* the image tiles along the bar at its own proportions; progress covers
     it in solid lime from the left as the count climbs toward the milestone. */
  .goal { --bar-w: 280px; --bar-h: 16px; }
  .goal .bar { display: block; flex: none; box-sizing: content-box;
        width: var(--bar-w); height: var(--bar-h);
        border: 2px solid var(--header-ink);
        background: url(/progress_bar.webp) left center / auto 100% repeat-x; }
  .goal .fill { display: block; height: 100%; background: var(--header-ink); }

  .milestone { position: fixed; inset: 0; z-index: 20; display: flex;
        align-items: center; justify-content: center; padding: 16px;
        background: rgba(0, 0, 0, .55); }
  .milestone .card { background: var(--bg); max-width: 520px; width: 100%;
        max-height: 100%; overflow: auto; padding: 16px; text-align: center; }
  .milestone img { display: block; width: 100%; height: auto; margin: 0 0 12px; }
  .milestone .msg { font-size: 15px; line-height: 1.5; margin: 8px 0 14px; }
  .milestone .close { background: var(--btn); color: var(--ink); border: 1px solid transparent;
        border-radius: 0; padding: 10px 22px; font-size: 15px; cursor: pointer;
        font-family: inherit; }
  .milestone .close:hover, .milestone .close:active { background: var(--accent);
        color: var(--accent-ink); }

  @media (max-width: 720px) {
    header { position: static; gap: 10px 14px; padding: 10px 12px; }
    header h1 { font-size: 17px; width: 100%; }
    .goal { --bar-w: 240px; --bar-h: 14px; }
    .fam .famlabel { max-width: none; }
    main { padding: 20px 12px 48px; }
    .pair { gap: 14px; margin-bottom: 20px; }
    .side img { width: 40vw; }
    .single img { width: 62vw; }
    /* both answer sets stay in one row, low-to-high or A-side to B-side, so
       the order reads the same way it does on a wide screen. */
    .choices, .choices.compare { flex-direction: row; flex-wrap: nowrap; align-items: stretch; gap: 4px; }
    .choices button, .choices.compare button { flex: 1 1 0; min-width: 0; width: auto; min-height: 64px;
          padding: 8px 3px; font-size: 11.5px; line-height: 1.25; white-space: normal; }
    .hint { display: none; }
    .backrow { margin-top: 16px; }
    .back { width: 100%; padding: 12px; font-size: 15px; min-height: 48px; }
  }
</style>
</head>
<body>
<header>
  <h1>Signature fluency rankings <span class="zh">花押流畅度评分</span></h1>
  <div class="modes">
    <button class="txt active" data-m="rank">Rank (A vs B) <span class="zh">排序（A 与 B）</span></button>
    <button class="txt" data-m="rate">Rate (0-4) <span class="zh">评分（0-4）</span></button>
  </div>
  <div class="stats" id="stats"></div>
</header>
<img class="scale-img" src="/intro_scale.webp" alt="Example signatures labeled 0 to 4">

<main id="main"></main>

<template id="intakeTpl">
  <form class="intake" id="intakeForm" autocomplete="off">
    <p class="intro">Hello! As part of a project involving a computational pipeline for signature extraction and analysis in historical Chinese contracts, we are looking for some help in assigning human judgments to around 500 signatures which we have linked to individuals in a genealogy of the Que lineage in Shicang village, Zhejiang. A comprehensive set of human labels may assist with supervised machine learning, and provide insights into the relationships between signatures, literacy, and social groups. If you would like to help, please continue reading.</p>
    <p class="intro">For context, measuring literacy in early modern China is quite difficult. Past studies—which have drawn on a variety of materials, including government proclamations, market reports, popular fiction, character books or reading primers, household account books, recipes, and other sources—generally propose a “continuum” of literacy with classical Chinese mastery, attained through studying for the civil examinations, on one end and full illiteracy on the other. However, scholars have been unable to arrive at a consensus on the specifics.</p>
    <p class="intro">To make observations about the literacy spectrum, this project is interested in looking at signatures as evidence for literacy and writing skill. Before the popularisation of seals, it was signatures, or <i>huaya</i>, which enabled individuals to prove the legitimacy of their claims, as they offered a means for individuals to verify their identity on a variety of documents from land transfers to marriage contracts, with illiterate individuals often signing with circles or crosses. Though signatures were often formed from sections of Chinese characters and followed different styles of marks, they reflected a range of unique personal quirks and traits, among which may include literacy level.</p>
    <p class="intro">But just how reliable are signatures as a proxy for literacy? Though signing ability has been linked to intermediate levels of literacy by Roger Schofield and David Cressy, it is unclear how signing ability relates to the lower end of writing ability. In addition, Evelyn Rawski notes that though historians of Europe may rely on signatures as a rough measure of literacy, the non-phonetic nature of the Chinese script renders signature data less reliable. Nevertheless, as Richard Rubinger and Liu Yonghua have shown, given the complexity of Chinese characters, the level of dexterity required to handle the ink brush, and the significance of signatures in legal verification, historical Chinese signatures or monograms remain a valuable, if mildly opaque, window into literacy.</p>
    <p class="intro">Because a significant link between signatures and literacy is the ability to wield a brush, for the purposes of this project we can define fluency as clarity and precision in writing, which may be more obvious in complex signatures (but is not necessarily the same as complexity since, between two non-complex signatures, one can be more fluent than the other), as seen in the chart above. Feel free to refer to it when making your own judgments!</p>
    <p class="intro">Thank you for reading! Please fill in some information about yourself before proceeding to the ranking. At the top of the page, you can toggle between modes: the “rank” mode shows two signatures and has you evaluate which looks more fluent and the “rate” mode shows one signature and has you evaluate its fluency on a 0 to 4 scale. Feel free to choose whichever mode seems more doable, or both. Finally, your progress will be saved automatically, so do not worry about leaving and coming back. You may finish or leave the tab at any time. There will be a small surprise for those who have completed 100 ratings.</p>

    <label>Name <span class="zh">姓名</span>: <span class="req">required</span>
      <input name="rater" type="text" maxlength="40" required
             autocapitalize="words" autocorrect="off">
    </label>

    <label>How would you rate your familiarity with the Chinese script on a scale of 0(none) and 5(fluent)?
      <span class="zh">你对汉字有多熟悉?</span>
      <span class="scale" data-for="chinese_familiarity"></span>
      <input type="hidden" name="chinese_familiarity">
    </label>

    <label>Age <span class="zh">年龄</span>:
      <input name="age" type="text" inputmode="numeric" maxlength="3"
             pattern="[0-9]*">
    </label>

    <label>First language <span class="zh">母语</span>:
      <input name="first_language" type="text" maxlength="60">
    </label>

    <label>Highest level of education <span class="zh">最高学历</span>:
      <input name="education" type="text" maxlength="80">
    </label>

    <div class="err" id="intakeErr"></div>
    <button class="go" type="submit">go! <span class="zh">开始！</span></button>
  </form>
</template>

<script>
let rater = localStorage.getItem("fluency_rater") || "";
let mode = "rank";
let current = null;
// ids this rater has answered in this sitting, newest last, one stack per mode.
// undo pops the top and blanks it on the server, which puts that item straight
// back in front of them. the stacks are per-visit: a reload starts empty, so
// undo never reaches back into work from an earlier session.
let history = {rank: [], rate: []};

const statsEl = document.getElementById("stats");
const mainEl = document.getElementById("main");
let profile = null;
let options = {familiarity_max: 5};

function scaleInto(span, max, initial) {
  const hidden = span.parentElement.querySelector("input[type=hidden]");
  hidden.value = initial === null || initial === undefined ? "" : String(initial);
  for (let v = 0; v <= max; v++) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = v;
    b.className = String(v) === hidden.value ? "on" : "";
    b.onclick = () => {
      hidden.value = String(v);
      span.querySelectorAll("button").forEach(o => o.classList.remove("on"));
      b.classList.add("on");
    };
    span.appendChild(b);
  }
}

function showIntake(prefill) {
  const tpl = document.getElementById("intakeTpl");
  mainEl.innerHTML = "";
  mainEl.appendChild(tpl.content.cloneNode(true));
  const form = document.getElementById("intakeForm");
  const p = prefill || profile || {};

  form.rater.value = rater || "";
  form.age.value = p.age === undefined || p.age === null ? "" : p.age;
  form.first_language.value = p.first_language || "";
  form.education.value = p.education || "";
  form.querySelectorAll(".scale").forEach(sp => {
    const key = sp.dataset.for;
    const initial = p[key] === undefined || p[key] === null || p[key] === "" ? null : p[key];
    scaleInto(sp, options.familiarity_max, initial);
  });

  form.onsubmit = async e => {
    e.preventDefault();
    const errEl = document.getElementById("intakeErr");
    const name = form.rater.value.trim();
    if (!name) { errEl.textContent = "please enter a name"; return; }
    const payload = {
      rater: name,
      age: form.age.value,
      first_language: form.first_language.value,
      education: form.education.value,
      chinese_familiarity: form.chinese_familiarity.value,
    };
    const res = await fetch("/api/rater", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const d = await res.json();
    if (!d.ok) { errEl.textContent = d.error || "could not save"; return; }
    rater = name;
    profile = d.profile;
    localStorage.setItem("fluency_rater", rater);
    loadNext();
  };
}

document.querySelectorAll(".modes button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".modes button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    mode = btn.dataset.m;
    loadNext();
  };
});

async function refreshStats() {
  if (!rater) { statsEl.innerHTML = ""; return; }
  const res = await fetch(`/api/session?rater=${encodeURIComponent(rater)}`);
  const d = await res.json();
  if (d.profile) profile = d.profile;
  // the milestone counts both modes together, so either one gets you there.
  const total = d.rank_judged + d.rate_judged;
  const pct = Math.min(100, total / MILESTONE * 100);
  statsEl.innerHTML = `<span class="goal"><span class="bar"><span class="fill" style="width:${pct}%"></span></span>` +
    `${total} / ${MILESTONE}</span>`;
  if (total >= MILESTONE) showMilestone();
}

const MILESTONE = 100;
let milestoneShown = false;

// shown once per rater per browser; the flag lives in localStorage so a
// reload or undo-and-redo around 100 does not bring it back.
function showMilestone() {
  const key = `fluency_milestone_${rater}`;
  if (milestoneShown || localStorage.getItem(key)) return;
  milestoneShown = true;
  localStorage.setItem(key, "1");
  const box = document.createElement("div");
  box.className = "milestone";
  box.innerHTML = `<div class="card">
      <img src="/milestone.webp" alt="An early Qing bronze coin with the words 谢谢！ Thank you!">
      <div class="msg">Thank you for completing 100 ratings! In return for your hard work, please take an early Qing bronze coin, casted (hypothetically) in 1651 in Beijing under the Shunzhi Emperor with a composition of 60% copper, 20% lead, and 20% zinc.</div>
      <button class="close">close <span class="zh">关闭</span></button>
    </div>`;
  // if the image file goes missing, fall back to the message alone
  box.querySelector("img").onerror = e => e.target.remove();
  box.querySelector(".close").onclick = () => box.remove();
  document.body.appendChild(box);
}

function tagHtml(tag) {
  return tag ? `<div class="tag">${tag}</div>` : "";
}

async function loadNext() {
  if (!rater) { showIntake(); return; }
  refreshStats();
  if (mode === "rank") await loadPair(); else await loadSingle();
}

// the back button is rendered even when nothing is left to answer, otherwise a
// misclick on the very last item would have no way back.
function backHtml(which) {
  // always drawn, so it reads as part of the furniture rather than something
  // that appears out of nowhere after the first answer. it is simply disabled
  // when there is nothing in this sitting left to take back.
  const off = history[which].length ? "" : " disabled";
  return '<div class="backrow"><button class="back" id="backBtn"' + off + '>' +
         '&larr; back<span class="zh">返回</span></button></div>';
}

function wireBack(fn) {
  const b = document.getElementById("backBtn");
  if (b) b.onclick = fn;
}

async function loadPair() {
  mainEl.innerHTML = '<div class="pair" id="pairArea"></div>';
  const res = await fetch(`/api/next_pair?rater=${encodeURIComponent(rater)}`);
  const d = await res.json();
  if (!d.pair) {
    mainEl.innerHTML = '<div class="done">No more pairs available</div>' + backHtml("rank");
    wireBack(undoMargin);
    current = null;
    return;
  }
  current = d.pair;
  document.getElementById("pairArea").innerHTML = `
    <div class="side">
      <div class="letter">A</div>
      <img src="${current.a.crop_url}">
      ${tagHtml(current.a.tag)}
    </div>
    <div class="side">
      <div class="letter">B</div>
      <img src="${current.b.crop_url}">
      ${tagHtml(current.b.tag)}
    </div>
  `;
  mainEl.innerHTML += `
    <div class="choices compare">
      <button data-margin="4">A is much more fluent<span class="zh">A 明显更流畅</span></button>
      <button data-margin="2">A is slightly more fluent<span class="zh">A 略微更流畅</span></button>
      <button data-margin="0">A and B are equally fluent<span class="zh">A 和 B 一样流畅</span></button>
      <button data-margin="-2">B is slightly more fluent<span class="zh">B 略微更流畅</span></button>
      <button data-margin="-4">B is much more fluent<span class="zh">B 明显更流畅</span></button>
    </div>
  ` + backHtml("rank");
  document.querySelectorAll(".choices button").forEach(b => {
    b.onclick = () => submitMargin(parseFloat(b.dataset.margin), b);
  });
  wireBack(undoMargin);
}

async function submitMargin(margin, btn) {
  if (btn) { btn.classList.add("flash"); }
  const id = current.comparison_id;
  await fetch("/api/judge_pair", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({rater, comparison_id: id, margin}),
  });
  history.rank.push(id);
  refreshStats();
  loadPair();
}

async function undoMargin() {
  if (!history.rank.length) return;
  const id = history.rank.pop();
  await fetch("/api/undo_pair", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({rater, comparison_id: id}),
  });
  refreshStats();
  loadPair();
}

async function loadSingle() {
  mainEl.innerHTML = '<div class="single" id="singleArea"></div>';
  const res = await fetch(`/api/next_single?rater=${encodeURIComponent(rater)}`);
  const d = await res.json();
  if (!d.item) {
    mainEl.innerHTML = '<div class="done">All signatures rated</div>' + backHtml("rate");
    wireBack(undoScore);
    current = null;
    return;
  }
  current = d.item;
  document.getElementById("singleArea").innerHTML = `
    <img src="${current.crop_url}">
    ${tagHtml(current.tag)}
  `;
  const legend = {0: "circles", 1: "very low fluency", 2: "low fluency",
                  3: "moderate fluency", 4: "high fluency"};
  let html = '<div class="choices">';
  for (let s = 0; s <= 4; s++) {
    html += `<button data-score="${s}">${s} &mdash; ${legend[s]}</button>`;
  }
  html += '</div>';
  mainEl.innerHTML += html + backHtml("rate");
  document.querySelectorAll(".choices button").forEach(b => {
    b.onclick = () => submitScore(parseFloat(b.dataset.score), b);
  });
  wireBack(undoScore);
}

async function submitScore(score, btn) {
  if (btn) { btn.classList.add("flash"); }
  const id = current.example_id;
  await fetch("/api/rate_single", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({rater, example_id: id, score}),
  });
  history.rate.push(id);
  refreshStats();
  loadSingle();
}

async function undoScore() {
  if (!history.rate.length) return;
  const id = history.rate.pop();
  await fetch("/api/undo_single", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({rater, example_id: id}),
  });
  refreshStats();
  loadSingle();
}

document.addEventListener("keydown", e => {
  const t = document.activeElement;
  if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
  // the milestone popup must be closed with its button; until then, answer
  // and undo keys should not act on the pair hidden behind it.
  if (document.querySelector(".milestone")) return;
  // backspace is handled before the `current` guard: on the "nothing left to
  // answer" screen current is null, and that is exactly when someone who has
  // just misclicked the final item needs to get back to it.
  if (e.key === "Backspace") {
    e.preventDefault();
    if (mode === "rank") undoMargin(); else undoScore();
    return;
  }
  if (!current) return;
  if (mode === "rank" && ["1","2","3","4","5"].includes(e.key)) {
    // keyed in the order the buttons appear on screen, A much ... B much
    const margins = {"1": 4, "2": 2, "3": 0, "4": -2, "5": -4};
    submitMargin(margins[e.key]);
  } else if (mode === "rate" && ["0","1","2","3","4"].includes(e.key)) {
    submitScore(parseFloat(e.key));
  }
});

async function boot() {
  try { options = await (await fetch("/api/intake_options")).json(); } catch (e) {}
  if (rater) {
    try {
      const d = await (await fetch(
        `/api/session?rater=${encodeURIComponent(rater)}`)).json();
      profile = d.profile || null;
    } catch (e) {}
  }
  if (rater && profile) loadNext(); else showIntake();
}
boot();
</script>
</body>
</html>
"""


def _json(obj, status=200):
    return status, "application/json", json.dumps(obj).encode("utf-8")


def _file(path, content_type):
    if path is None or not Path(path).exists():
        return 404, "text/plain", b"not found"
    return 200, content_type, Path(path).read_bytes()


def _side(row, side):
    item_id = CORPUS.pair_item_ids(row)[0 if side == "a" else 1]
    return {"crop_url": CORPUS.crop_url(item_id),
            "tag": CORPUS.tag(item_id) if CORPUS.show_tag else ""}


STATIC_WEBP = ("intro_scale.webp", "progress_bar.webp", "milestone.webp")


def route(method, raw_path, body_bytes=b""):
    """one request -> (status, content type, body). shared by the local
    http.server Handler and the WSGI app (fluency_wsgi.py) PythonAnywhere runs."""
    parsed = urlparse(raw_path)
    path = unquote(parsed.path)

    if method == "GET":
        qs = parse_qs(parsed.query)
        rater = sanitize_name(qs.get("rater", [""])[0])

        if path == "/":
            return 200, "text/html; charset=utf-8", PAGE.encode("utf-8")

        if path.lstrip("/") in STATIC_WEBP:
            return _file(Path(__file__).parent / path.lstrip("/"), "image/webp")

        prefix = f"/image/{CORPUS.name}/"
        if path.startswith(prefix):
            return _file(CORPUS.image_path(path[len(prefix):]), "image/png")

        if path == "/api/intake_options":
            return _json({"familiarity_max": FAMILIARITY_MAX})

        if path == "/api/session":
            master = load_master()
            rater_pw = load_rater_pairwise(rater, master)
            _, _, rate_judged = next_single_for(rater)
            return _json({
                "rank_judged": int(rater_pw["human_margin"].notna().sum()),
                "rate_judged": rate_judged,
                "profile": get_profile(rater),
            })

        if path == "/api/next_pair":
            pair, _, _ = next_pair_for(rater)
            if pair is None:
                return _json({"pair": None})
            return _json({"pair": {
                "comparison_id": int(pair["comparison_id"]),
                "a": _side(pair, "a"),
                "b": _side(pair, "b"),
            }})

        if path == "/api/next_single":
            item, _, _ = next_single_for(rater)
            if item is None:
                return _json({"item": None})
            return _json({"item": {
                "example_id": int(item["example_id"]),
                "crop_url": CORPUS.crop_url(item["item_id"]),
                "tag": CORPUS.tag(item["item_id"]) if CORPUS.show_tag else "",
            }})

        return 404, "text/plain", b"not found"

    if method == "POST":
        body = json.loads(body_bytes) if body_bytes else {}
        rater = sanitize_name(body.get("rater", ""))

        if path == "/api/judge_pair":
            save_margin(rater, int(body["comparison_id"]), float(body["margin"]))
            return _json({"ok": True})

        if path == "/api/rater":
            profile, err = clean_profile(body)
            if profile is None:
                return _json({"ok": False, "error": err}, status=400)
            save_profile(rater, profile)
            return _json({"ok": True, "profile": get_profile(rater)})

        if path == "/api/rate_single":
            save_score(rater, int(body["example_id"]), float(body["score"]))
            return _json({"ok": True})

        if path == "/api/undo_pair":
            clear_margin(rater, int(body["comparison_id"]))
            return _json({"ok": True})

        if path == "/api/undo_single":
            clear_score(rater, int(body["example_id"]))
            return _json({"ok": True})

    return 404, "text/plain", b"not found"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _reply(self, method, body=b""):
        status, ctype, data = route(method, self.path, body)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._reply("GET")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self._reply("POST", self.rfile.read(length) if length else b"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=sorted(CORPORA), default="people",
                        help="which crops to judge (default: core)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to bind. the default is this machine only; "
                             "pass 0.0.0.0 to let other people on the network play")
    parser.add_argument("--no-browser", action="store_true",
                        help="serve without opening a browser window")
    parser.add_argument("--one-per-person", action="store_true",
                        help="keep only the largest crop for each person, so "
                             "every comparison is between two different hands")
    parser.add_argument("--limit", type=int, default=None,
                        help="hold the pool to this many crops (anonymised "
                             "corpora only). already-judged crops are always "
                             "kept; the rest is sampled per source")
    parser.add_argument("--export", metavar="DIR", default=None,
                        help="write a bundle for the hosted server "
                             "(fluency_wsgi.py) to DIR and exit")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.corpus} crops ...")
    if args.corpus == "core":
        CORPUS = CORPORA[args.corpus]()
    else:
        CORPUS = CORPORA[args.corpus](limit=args.limit,
                                      one_per_person=args.one_per_person)
    print(f"{len(CORPUS.items)} crop(s) available to judge")
    if isinstance(CORPUS, AnonCorpus):
        counts = pd.read_csv(anon_map_csv(CORPUS), encoding="utf-8-sig")["source"]
        print("  " + ", ".join(f"{n} {s}" for s, n in counts.value_counts().items()))
        print(f"anonymised; join key -> {anon_map_csv(CORPUS)}")

    if args.export:
        export_bundle(CORPUS, args.export)
        raise SystemExit(0)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving at {url}  (ctrl-C to stop)")
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"other players on this network: http://{lan_address()}:{args.port}/")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")

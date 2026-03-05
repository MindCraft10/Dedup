# DeDup v9.4 — Production (compact, fast, 200k+ ready)
# Streamlit 1.17+ | Python 3.12+
#
# Features:
# - Auto header-row detection (no manual header row needed)
# - Robust header recognition (First/Middle/Last/Ext, Barangay, Month/Day/Year)
# - Birthdate validation uses Month+Day+Year ONLY (no BirthdateFull)
# - Lack-of-detail = BLANK only (Barangay, First, Middle, Last, Month, Day, Year)
# - Fast duplicate detection (name-only) for 200k+ rows:
#     Stage1 exact on normalized full name
#     Stage2 blocking on soundex(last)+prefixes
#     Stage3 fuzzy refine only in tiny blocks (RapidFuzz if installed)
# - Duplicate Group ID + next-row layout
# - Cleaner exports (Summary/Clean/Duplicate/LackDetail)
#
# Run:
#   streamlit run dedup_v9_4.py

from __future__ import annotations

import io, re, unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from rapidfuzz import fuzz  # type: ignore
    HAS_RF = True
except Exception:
    HAS_RF = False

st.set_page_config(page_title="DeDup v9.4", layout="wide")
st.title("DeDup v9.4 — Production (200k+ ready)")
st.caption("Auto header + robust mapping + fast duplicates + strict lack-of-detail (blank only).")
st.divider()


# -------------------------
# Utils
# -------------------------
def norm_txt(x: object) -> str:
    s = "" if x is None else str(x).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()


def norm_ser(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.strip().str.lower()
    x = x.str.replace(r"\s+", " ", regex=True)
    x = x.str.replace(r"[^a-z0-9 ]+", "", regex=True)
    return x


def clean_cols(cols: List[str]) -> List[str]:
    return [str(c).replace("_", "").strip() if c is not None else "" for c in cols]


def is_excel(up) -> bool:
    n = (getattr(up, "name", "") or "").lower()
    return n.endswith((".xlsx", ".xls"))


def excel_sheets(up) -> List[str]:
    if up is None or not is_excel(up):
        return []
    return list(pd.ExcelFile(io.BytesIO(up.getvalue())).sheet_names)


def read_raw(up, sheet: Optional[str]) -> pd.DataFrame:
    n = up.name.lower()
    b = up.getvalue()
    if n.endswith(".csv"):
        return pd.read_csv(io.BytesIO(b), header=None, dtype=str, keep_default_na=False)
    if n.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(b), sheet_name=sheet, header=None, dtype=str, keep_default_na=False)
    raise ValueError("Unsupported file type")


def apply_header(df_raw: pd.DataFrame, hr0: int) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    hr0 = max(int(hr0), 0)
    if hr0 >= len(df_raw):
        return pd.DataFrame()
    headers = df_raw.iloc[hr0].astype(str).tolist()
    headers = [h.strip() if h and h.strip() else f"COL_{i+1}" for i, h in enumerate(headers)]
    df = df_raw.iloc[hr0 + 1 :].copy()
    df.columns = headers
    return df.reset_index(drop=True)


def drop_empty_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    tmp = df.fillna("").astype(str).apply(lambda c: c.str.strip())
    mask = (tmp == "").all(axis=1)
    return df.loc[~mask].copy(), int(mask.sum())


# -------------------------
# Auto header-row detection
# -------------------------
REQ_KEYS = {
    "barangay": ["barangay", "brgy", "brgay", "baranggay", "brg"],
    "first": ["first", "firstname", "first name", "fname", "given", "forename", "fn"],
    "middle": ["middle", "middlename", "middle name", "mname", "mi", "m.i", "second name"],
    "last": ["last", "lastname", "last name", "lname", "ln", "surname", "family", "family name"],
    "ext": ["ext", "extname", "ext name", "suffix", "jr", "sr", "iii", "iv"],
    "month": ["month", "birthmonth", "birth month", "bmonth"],
    "day": ["day", "birthday", "birth day", "bday"],
    "year": ["year", "birthyear", "birth year", "byear"],
}
EXTRA_HINTS = ["province", "city", "municipality", "address", "sex", "gender", "id", "no", "status"]


def header_score(row_vals: List[str]) -> float:
    vals = [norm_txt(v) for v in row_vals if v is not None]
    vals = [v for v in vals if v]
    if not vals:
        return -1e9
    non_empty = len(vals)
    uniq_ratio = len(set(vals)) / max(non_empty, 1)
    numeric_ratio = sum(1 for v in vals if v.isdigit()) / max(non_empty, 1)

    matched = 0
    for kws in REQ_KEYS.values():
        ok = any(any(norm_txt(k) in cell or norm_txt(k).replace(" ", "") in cell.replace(" ", "") for k in kws) for cell in vals)
        matched += 1 if ok else 0

    extra = sum(1 for cell in vals if any(h in cell for h in EXTRA_HINTS))
    avg_len = sum(len(v) for v in vals) / max(non_empty, 1)
    short_bonus = 2.0 if avg_len <= 22 else 0.0

    return matched * 30 + non_empty * 1.2 + uniq_ratio * 10 + extra * 2 + short_bonus - numeric_ratio * 25


def auto_header_row(df_raw: pd.DataFrame, scan: int = 60) -> int:
    if df_raw is None or df_raw.empty:
        return 0
    scan = min(scan, len(df_raw))
    best_r, best_s = 0, -1e18
    for r in range(scan):
        s = header_score(df_raw.iloc[r].astype(str).tolist())
        if s > best_s:
            best_s, best_r = s, r
    return int(best_r)


# -------------------------
# Header mapping ("AI-level" robust)
# -------------------------
CANON = ["FirstName", "MiddleName", "LastName", "ExtName", "Barangay", "BirthMonth", "BirthDay", "BirthYear"]

SYN: Dict[str, List[str]] = {
    "FirstName": ["first", "firstname", "first name", "fname", "given", "given name", "forename", "fn"],
    "MiddleName": ["middle", "middlename", "middle name", "mname", "mi", "m.i", "second name"],
    "LastName": ["last", "lastname", "last name", "lname", "ln", "surname", "family", "family name"],
    "ExtName": ["ext", "extname", "ext name", "suffix", "extension", "jr", "sr", "iii", "iv"],
    "Barangay": ["barangay", "brgy", "brgay", "baranggay", "purok", "village"],
    "BirthMonth": ["month", "birthmonth", "birth month", "bmonth"],
    "BirthDay": ["day", "birthday", "birth day", "bday"],
    "BirthYear": ["year", "birthyear", "birth year", "byear"],
}
CONTAINS: Dict[str, List[str]] = {
    "FirstName": ["first", "given", "fname", "forename", "fn"],
    "MiddleName": ["middle", "mname", "mi"],
    "LastName": ["last", "surname", "family", "lname", "ln"],
    "ExtName": ["ext", "suffix", "jr", "sr", "iii", "iv"],
    "Barangay": ["barangay", "brgy", "purok", "village"],
    "BirthMonth": ["month"],
    "BirthDay": ["day"],
    "BirthYear": ["year"],
}
ABBR = {
    "MiddleName": [re.compile(r"^\s*m\.?\s*i\.?\s*$", re.I)],
    "FirstName": [re.compile(r"^\s*f\.?\s*n\.?\s*$", re.I)],
    "LastName": [re.compile(r"^\s*l\.?\s*n\.?\s*$", re.I)],
}


def sim(a: str, b: str) -> int:
    a, b = norm_txt(a), norm_txt(b)
    if not a or not b:
        return 0
    if HAS_RF:
        return int(fuzz.token_set_ratio(a, b))
    return int(100 * SequenceMatcher(None, a, b).ratio())


def contains_score(raw: str, canon: str) -> int:
    for rx in ABBR.get(canon, []):
        if rx.match(raw):
            return 98
    for kw in CONTAINS.get(canon, []):
        if kw in raw:
            return 96
    return 0


def map_columns(df: pd.DataFrame, thr: int) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, int]]:
    cols = [c for c in df.columns if c is not None]
    ncols = {c: norm_txt(c) for c in cols}
    used, mapping, scores = set(), {}, {}
    for canon in CANON:
        best_c, best_s = None, -1
        for c in cols:
            if c in used:
                continue
            raw = ncols[c]
            s = max(contains_score(raw, canon), max(sim(raw, s0) for s0 in SYN.get(canon, []) + [canon]))
            # extra boost when header contains 'name' + role word
            toks = raw.split()
            if "name" in toks and canon in {"FirstName", "MiddleName", "LastName"}:
                if ("first" in toks or "given" in toks) and canon == "FirstName":
                    s = max(s, 97)
                if ("middle" in toks or "mi" in toks) and canon == "MiddleName":
                    s = max(s, 97)
                if ("last" in toks or "surname" in toks or "family" in toks) and canon == "LastName":
                    s = max(s, 97)
            if s > best_s:
                best_s, best_c = s, c
        scores[canon] = int(best_s if best_s > 0 else 0)
        if best_c is not None and best_s >= thr:
            mapping[canon] = best_c
            used.add(best_c)
    ren = {orig: canon for canon, orig in mapping.items() if orig != canon}
    return (df.rename(columns=ren) if ren else df), mapping, scores


# -------------------------
# Birth validation (Month/Day/Year only)
# -------------------------
def birth_ok(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    for c in ["BirthMonth", "BirthDay", "BirthYear"]:
        if c not in df.columns:
            df[c] = ""
    m = df["BirthMonth"].fillna("").astype(str).str.strip().eq("")
    d = df["BirthDay"].fillna("").astype(str).str.strip().eq("")
    y = df["BirthYear"].fillna("").astype(str).str.strip().eq("")
    ok = ~(m | d | y)
    dbg = pd.DataFrame({"reason": ["Missing BirthMonth", "Missing BirthDay", "Missing BirthYear"], "count": [int(m.sum()), int(d.sum()), int(y.sum())]})
    return ok, dbg


def lack_mask_and_reason(df: pd.DataFrame, b_ok: pd.Series) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    # Ensure required exist
    for c in ["FirstName", "MiddleName", "LastName", "Barangay"]:
        if c not in df.columns:
            df[c] = ""
    f = df["FirstName"].fillna("").astype(str).str.strip().eq("")
    m = df["MiddleName"].fillna("").astype(str).str.strip().eq("")
    l = df["LastName"].fillna("").astype(str).str.strip().eq("")
    b = df["Barangay"].fillna("").astype(str).str.strip().eq("")
    bm = df["BirthMonth"].fillna("").astype(str).str.strip().eq("")
    bd = df["BirthDay"].fillna("").astype(str).str.strip().eq("")
    by = df["BirthYear"].fillna("").astype(str).str.strip().eq("")
    miss_birth = ~(b_ok.fillna(False))
    mask = f | m | l | b | miss_birth

    reasons = []
    for i in range(len(df)):
        rs = []
        if b.iloc[i]: rs.append("Missing Barangay")
        if f.iloc[i]: rs.append("Missing FirstName")
        if m.iloc[i]: rs.append("Missing MiddleName")
        if l.iloc[i]: rs.append("Missing LastName")
        if bm.iloc[i] or bd.iloc[i] or by.iloc[i]:
            miss = []
            if bm.iloc[i]: miss.append("Month")
            if bd.iloc[i]: miss.append("Day")
            if by.iloc[i]: miss.append("Year")
            rs.append("Missing Birth " + "/".join(miss))
        reasons.append("; ".join(rs))
    reason_ser = pd.Series(reasons, index=df.index, dtype=str)

    dbg = pd.DataFrame({
        "reason": ["Missing FirstName", "Missing MiddleName", "Missing LastName", "Missing Barangay", "Missing Birth Month/Day/Year"],
        "count": [int(f.sum()), int(m.sum()), int(l.sum()), int(b.sum()), int(miss_birth.sum())],
    })
    return mask, reason_ser, dbg


# -------------------------
# Duplicates (fast, 200k+)
# -------------------------
def soundex_one(s: str) -> str:
    s = norm_txt(s).replace(" ", "")
    if not s:
        return ""
    first = s[0].upper()
    mp = {"b":"1","f":"1","p":"1","v":"1","c":"2","g":"2","j":"2","k":"2","q":"2","s":"2","x":"2","z":"2","d":"3","t":"3","l":"4","m":"5","n":"5","r":"6"}
    digits, prev = [], ""
    for ch in s[1:]:
        code = mp.get(ch, "0")
        if code != "0" and code != prev:
            digits.append(code)
        prev = code
    out = (first + "".join(digits)).replace("0", "")
    return (out + "000")[:4]


def soundex_ser(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(soundex_one)


@dataclass
class DupCfg:
    fuzzy_thr: int
    max_block: int
    fuzzy_max: int
    map_thr: int


def cfg_from_slider(level: int) -> DupCfg:
    level = int(level)
    fuzzy_thr = max(75, min(100, level))
    max_block = int(max(800, min(5000, round(8000 - 70 * level))))   # 80->2400, 100->1000
    fuzzy_max = int(max(20, min(80, round(120 - 1.0 * level))))      # 80->40, 100->20
    map_thr = int(max(35, min(85, round(0.55 * level))))             # 80->44, 100->55
    return DupCfg(fuzzy_thr=fuzzy_thr, max_block=max_block, fuzzy_max=fuzzy_max, map_thr=map_thr)


def fuzzy_ratio(a: str, b: str) -> int:
    if HAS_RF:
        return int(fuzz.token_set_ratio(a, b))
    return int(100 * SequenceMatcher(None, a, b).ratio())


def detect_dups(df: pd.DataFrame, cfg: DupCfg) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)

    f = norm_ser(df["FirstName"])
    m = norm_ser(df["MiddleName"])
    l = norm_ser(df["LastName"])
    full = (f + " " + m + " " + l).str.replace(r"\s+", " ", regex=True).str.strip()

    dup = pd.Series(False, index=df.index)

    # Stage 1 exact
    exact = full.ne("") & full.duplicated(keep=False)
    dup |= exact

    # Stage 2 blocking
    key = soundex_ser(l) + "|" + f.str[:2].fillna("") + "|" + l.str[:2].fillna("") + "|" + m.str[:1].fillna("")
    cand = key.ne("|||") & key.duplicated(keep=False)
    if not cand.any():
        return dup

    sizes = key[cand].groupby(key[cand]).transform("size")
    small = cand & (sizes <= cfg.max_block)

    if not HAS_RF:
        dup |= small
        return dup

    # Stage 3 fuzzy refine only tiny blocks
    tiny = cand & (sizes <= cfg.fuzzy_max)
    if not tiny.any():
        return dup

    groups = df.index[tiny].to_series().groupby(key[tiny]).apply(list)
    dup_local = pd.Series(False, index=df.index)
    for idxs in groups.values:
        if len(idxs) < 2:
            continue
        strs = full.loc[idxs].tolist()
        for i in range(len(idxs)):
            si = strs[i]
            if not si:
                continue
            matched = [i]
            for j in range(i + 1, len(idxs)):
                if fuzzy_ratio(si, strs[j]) >= cfg.fuzzy_thr:
                    matched.append(j)
            if len(matched) > 1:
                dup_local.loc[[idxs[k] for k in matched]] = True
    dup |= dup_local
    return dup


def build_dup_table(df_dup: pd.DataFrame) -> pd.DataFrame:
    if df_dup.empty:
        return df_dup
    df = df_dup.copy()

    f = norm_ser(df["FirstName"])
    m = norm_ser(df["MiddleName"])
    l = norm_ser(df["LastName"])
    full = (f + " " + m + " " + l).str.replace(r"\s+", " ", regex=True).str.strip()

    # group key: exact groups use full, else block key + short prefixes
    key = soundex_ser(l) + "|" + f.str[:2].fillna("") + "|" + l.str[:2].fillna("") + "|" + m.str[:1].fillna("")
    pref = l.str[:6].fillna("") + "|" + f.str[:6].fillna("") + "|" + m.str[:3].fillna("")
    gkey = full.where(full.duplicated(keep=False), key + "||" + pref)

    # assign IDs
    vc = gkey.value_counts()
    order = list(vc.sort_values(ascending=False).index)
    gid = {k: f"DUP-{i+1:06d}" for i, k in enumerate(order)}
    df["DuplicateGroupID"] = gkey.map(gid)

    # similarity inside group (small groups only)
    sim = pd.Series([None] * len(df), index=df.index, dtype="object")
    # exact=100
    exact_groups = df.index[full.ne("")].to_series().groupby(full[full.ne("")]).apply(list)
    for idxs in exact_groups.values:
        if len(idxs) >= 2:
            sim.loc[idxs] = 100

    need = sim.isna()
    if need.any():
        groups = df.index[need].to_series().groupby(df.loc[need, "DuplicateGroupID"]).apply(list)
        for idxs in groups.values:
            if len(idxs) < 2 or len(idxs) > 80:
                continue
            strs = full.loc[idxs].tolist()
            for i in range(len(idxs)):
                best = 0
                for j in range(len(idxs)):
                    if i == j:
                        continue
                    best = max(best, fuzzy_ratio(strs[i], strs[j]))
                    if best == 100:
                        break
                sim.loc[idxs[i]] = best
    df["SimilarityPct"] = pd.to_numeric(sim, errors="coerce")

    return df.sort_values(["DuplicateGroupID", "SimilarityPct"], ascending=[True, False], na_position="last")


# -------------------------
# Export
# -------------------------
def export_excel(summary: pd.DataFrame, clean: pd.DataFrame, dup: pd.DataFrame, lack: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="Summary")
        clean.to_excel(w, index=False, sheet_name="Clean")

        dupv = build_dup_table(dup)
        dup_pref = ["DuplicateGroupID", "SimilarityPct", "FirstName", "MiddleName", "LastName", "ExtName",
                    "Barangay", "BirthMonth", "BirthDay", "BirthYear", "__table__"]
        dup_cols = [c for c in dup_pref if c in dupv.columns] + [c for c in dupv.columns if c not in dup_pref]
        dupv[dup_cols].to_excel(w, index=False, sheet_name="Duplicate")

        lack_pref = ["FirstName", "MiddleName", "LastName", "ExtName", "Barangay",
                     "BirthMonth", "BirthDay", "BirthYear", "_LackReason", "__table__"]
        lack_cols = [c for c in lack_pref if c in lack.columns] + [c for c in lack.columns if c not in lack_pref]
        lack[lack_cols].to_excel(w, index=False, sheet_name="LackDetail")
    return out.getvalue()


# -------------------------
# UI
# -------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Upload")
    up_a = st.file_uploader("Table A (CSV/Excel)", type=["csv", "xlsx", "xls"], key="up_a")
    sheet_a = st.selectbox("A: Sheet", excel_sheets(up_a), key="sheet_a") if (up_a and is_excel(up_a)) else None

    up_b = st.file_uploader("Table B (CSV/Excel)", type=["csv", "xlsx", "xls"], key="up_b")
    sheet_b = st.selectbox("B: Sheet", excel_sheets(up_b), key="sheet_b") if (up_b and is_excel(up_b)) else None

with right:
    st.subheader("Auto Scope Settings")
    level = st.slider("Strictness / Speed", 70, 100, 80, 1,
                      help="Higher = stricter + faster. Lower = more recall (slower).")
    show_debug = st.checkbox("Show Debug Panel", value=False)

st.divider()

if not up_a and not up_b:
    st.info("Upload Table A and/or Table B to run.")
    st.stop()

cfg = cfg_from_slider(level)

def process_one(up, sheet, label: str) -> Tuple[pd.DataFrame, dict, dict, int, int]:
    raw = read_raw(up, sheet)
    hr = auto_header_row(raw)
    df = apply_header(raw, hr)
    if df.empty:
        return pd.DataFrame(), {}, {}, hr, 0
    df.columns = clean_cols(list(df.columns))
    df, mapping, scores = map_columns(df, cfg.map_thr)
    df["__table__"] = label
    df, dropped = drop_empty_rows(df)
    # ensure canon exist
    for c in CANON:
        if c not in df.columns:
            df[c] = ""
    return df, mapping, scores, hr, dropped

dfs, map_rows, head_rows = [], [], []
dropped_a = dropped_b = 0

if up_a:
    dfA, mA, sA, hrA, dropped_a = process_one(up_a, sheet_a, "A")
    if not dfA.empty: dfs.append(dfA)
    map_rows.append({"Table":"A","auto_header_row_0":hrA, **{k:f"{mA.get(k,'(not found)')} (score {sA.get(k,0)})" for k in CANON}})
    head_rows.append({"Table":"A","auto_header_row_0":hrA,"dropped_empty_rows":dropped_a})

if up_b:
    dfB, mB, sB, hrB, dropped_b = process_one(up_b, sheet_b, "B")
    if not dfB.empty: dfs.append(dfB)
    map_rows.append({"Table":"B","auto_header_row_0":hrB, **{k:f"{mB.get(k,'(not found)')} (score {sB.get(k,0)})" for k in CANON}})
    head_rows.append({"Table":"B","auto_header_row_0":hrB,"dropped_empty_rows":dropped_b})

if not dfs:
    st.error("No usable data after reading files. Check the selected sheet and file format.")
    st.stop()

df_all = pd.concat(dfs, ignore_index=True)

b_ok, birth_dbg = birth_ok(df_all)
df_all["_BirthOK"] = b_ok.astype(bool)

lack_mask, lack_reason, lack_dbg = lack_mask_and_reason(df_all, b_ok)
df_all["_LackReason"] = lack_reason

df_lack = df_all[lack_mask].copy()

df_not_lack = df_all[~lack_mask].copy()
dup_mask = detect_dups(df_not_lack, cfg)
df_dup = df_not_lack[dup_mask].copy()
df_clean = df_not_lack[~dup_mask].copy()

# Summary counts
def cnt(df: pd.DataFrame, t: str) -> int:
    return int((df.get("__table__", pd.Series([], dtype=str)) == t).sum()) if not df.empty else 0

summary = pd.DataFrame([
    {"Category":"Rows", "A":cnt(df_all,"A"), "B":cnt(df_all,"B"), "A+B":len(df_all)},
    {"Category":"Duplicate (Own+Cross)", "A":cnt(df_dup,"A"), "B":cnt(df_dup,"B"), "A+B":len(df_dup)},
    {"Category":"Lack of detail", "A":cnt(df_lack,"A"), "B":cnt(df_lack,"B"), "A+B":len(df_lack)},
    {"Category":"Clean", "A":cnt(df_clean,"A"), "B":cnt(df_clean,"B"), "A+B":len(df_clean)},
    {"Category":"Dropped empty rows", "A":int(dropped_a), "B":int(dropped_b), "A+B":int(dropped_a+dropped_b)},
])

st.subheader("Report")
st.dataframe(summary, use_container_width=True)

tabs = st.tabs(["Duplicate", "Lack of detail", "Clean"])

with tabs[0]:
    st.caption("DuplicateGroupID groups duplicates. Rows are listed next to each other (next-row layout).")
    dup_view = build_dup_table(df_dup)
    pref = ["DuplicateGroupID","SimilarityPct","FirstName","MiddleName","LastName","ExtName","Barangay","BirthMonth","BirthDay","BirthYear","__table__"]
    cols = [c for c in pref if c in dup_view.columns]
    st.dataframe(dup_view[cols] if cols else dup_view, use_container_width=True, height=420)

with tabs[1]:
    st.caption("Only rows with BLANK required fields are here (blank-only logic).")
    pref = ["FirstName","MiddleName","LastName","ExtName","Barangay","BirthMonth","BirthDay","BirthYear","_LackReason","__table__"]
    cols = [c for c in pref if c in df_lack.columns]
    st.dataframe(df_lack[cols] if cols else df_lack, use_container_width=True, height=420)

with tabs[2]:
    st.caption("Clean = remaining rows after removing Duplicate + Lack.")
    st.dataframe(df_clean, use_container_width=True, height=420)

st.divider()
st.download_button(
    "Export Excel",
    data=export_excel(summary, df_clean, df_dup, df_lack),
    file_name="DeDup_Report.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

if show_debug:
    with st.expander("Debug Panel", expanded=False):
        st.write({"rapidfuzz_installed": HAS_RF, "map_threshold": cfg.map_thr, "fuzzy_threshold": cfg.fuzzy_thr,
                  "max_block": cfg.max_block, "fuzzy_max_block": cfg.fuzzy_max})
        st.markdown("**Auto header rows**")
        st.dataframe(pd.DataFrame(head_rows), use_container_width=True)
        st.markdown("**Header mapping**")
        st.dataframe(pd.DataFrame(map_rows), use_container_width=True)
        st.markdown("**Birth missing counts**")
        st.dataframe(birth_dbg, use_container_width=True)
        st.markdown("**Lack reason counts**")
        st.dataframe(lack_dbg, use_container_width=True)
        st.markdown("**Lack preview (top 50)**")
        cols = [c for c in ["FirstName","MiddleName","LastName","Barangay","BirthMonth","BirthDay","BirthYear","_LackReason","__table__"] if c in df_lack.columns]
        st.dataframe(df_lack[cols].head(50), use_container_width=True)

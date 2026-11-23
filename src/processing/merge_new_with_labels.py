import json
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TEXT_JSON = BASE_DIR / "dataset_new.json"
LABELS_CSV = BASE_DIR / "DATA_classified_by_students_2026_update.csv"
OUT_CSV = BASE_DIR / "dataset_new_merged_labeled.csv"
OUT_LABELED_ONLY_CSV = BASE_DIR / "dataset_new_merged_labeled_only.csv"
OUT_QA_JSON = BASE_DIR / "dataset_new_merge_qa.json"
OUT_QA_TXT = BASE_DIR / "dataset_new_merge_qa.txt"


TRACKING_PARAMS = {
    "ref",
    "__tn__",
    "__cft__",
    "mibextid",
    "fbclid",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
}


def normalize_url(raw_url: str) -> str:
    if pd.isna(raw_url):
        return ""
    url = str(raw_url).strip()
    # Some rows contain accidental whitespace inside URLs.
    url = "".join(url.split())
    if not url:
        return ""

    parts = urlsplit(url)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path or ""

    # Normalize known Facebook path variants.
    path = path.replace("/photo/", "/photo")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Keep business keys (e.g. fbid, set) and drop tracking params.
    query_items = []
    for k, v in parse_qsl(parts.query, keep_blank_values=True):
        if k.lower() in TRACKING_PARAMS:
            continue
        query_items.append((k, v))
    query = urlencode(query_items, doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))


def load_text_df(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, item in enumerate(data, start=1):
        ts = item.get("timestamp")
        ts_fmt = ""
        if ts:
            try:
                dt = pd.to_datetime(ts, utc=True)
                ts_fmt = dt.strftime("%d/%m/%Y %H:%M")
            except Exception:
                ts_fmt = str(ts)

        fullname = str(item.get("fullname") or "").strip()
        username = str(item.get("username") or "").strip()
        municipality = fullname if fullname else username.lstrip("@")

        rows.append(
            {
                "code": float(i),
                "municipality_name": municipality,
                "link": str(item.get("url") or "").strip(),
                "date": ts_fmt,
                "post_text": str(item.get("text") or "").strip(),
                "source_json_id": str(item.get("id") or "").strip(),
            }
        )

    return pd.DataFrame(rows)


def load_labels_df(path: Path) -> pd.DataFrame:
    # The labels file is semicolon-separated and commonly saved in cp1252/latin1.
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=";", encoding=enc)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Could not read labels CSV: {last_err}")


def resolve_label_duplicates(labels_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Resolve repeated normalized URLs by picking the most frequent non-empty label."""
    resolved_rows = []
    conflict_groups = 0

    for _, group in labels_df.groupby("url_norm", dropna=False, sort=False):
        g = group.copy()
        g["CategoryName"] = g["CategoryName"].astype(str).str.strip()
        g.loc[g["CategoryName"].isin(["", "nan", "None"]), "CategoryName"] = pd.NA

        non_empty = g[g["CategoryName"].notna()]
        if non_empty.empty:
            resolved_rows.append(g.iloc[0])
            continue

        counts = non_empty["CategoryName"].value_counts(dropna=True)
        top_count = counts.iloc[0]
        winners = counts[counts == top_count].index.tolist()
        selected_name = winners[0]
        if len(winners) > 1:
            conflict_groups += 1

        chosen = non_empty[non_empty["CategoryName"] == selected_name].iloc[0].copy()
        resolved_rows.append(chosen)

    resolved_df = pd.DataFrame(resolved_rows)
    return resolved_df, conflict_groups


def main() -> None:
    text_df = load_text_df(TEXT_JSON)
    labels_df = load_labels_df(LABELS_CSV)

    required_text_cols = {"code", "municipality_name", "link", "date", "post_text", "source_json_id"}
    required_label_cols = {"id", "pageName", "postLink", "categoryId", "CategoryName"}

    missing_text = required_text_cols - set(text_df.columns)
    missing_labels = required_label_cols - set(labels_df.columns)
    if missing_text:
        raise ValueError(f"Missing columns in text dataset: {sorted(missing_text)}")
    if missing_labels:
        raise ValueError(f"Missing columns in labels dataset: {sorted(missing_labels)}")

    text_df = text_df.copy()
    labels_df = labels_df.copy()

    text_df["url_norm"] = text_df["link"].map(normalize_url)
    labels_df["url_norm"] = labels_df["postLink"].map(normalize_url)

    # Detect duplicates by normalized URL for QA and avoid exploding joins.
    text_dup_mask = text_df.duplicated(subset=["url_norm"], keep=False)
    labels_dup_mask = labels_df.duplicated(subset=["url_norm"], keep=False)

    text_dups = text_df.loc[text_dup_mask, ["code", "link", "url_norm"]].sort_values("url_norm")
    labels_dups = labels_df.loc[labels_dup_mask, ["id", "postLink", "url_norm", "categoryId", "CategoryName"]].sort_values("url_norm")

    # Keep first label per URL; preserve duplicates in QA report for manual review.
    labels_unique, duplicate_conflict_groups = resolve_label_duplicates(labels_df)

    merged = text_df.merge(
        labels_unique[["url_norm", "categoryId", "CategoryName", "postLink", "id"]],
        on="url_norm",
        how="left",
        suffixes=("", "_label"),
    )

    merged["match_method"] = merged["id"].apply(lambda x: "url_normalized" if pd.notna(x) else "unmatched")

    # Reorder and keep only agreed output columns + traceability.
    out_cols = [
        "code",
        "municipality_name",
        "link",
        "date",
        "post_text",
        "categoryId",
        "CategoryName",
        "match_method",
    ]
    merged_out = merged[out_cols].copy()

    merged_out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    merged_labeled_only = merged_out[merged_out["match_method"] != "unmatched"].copy()
    merged_labeled_only.to_csv(OUT_LABELED_ONLY_CSV, index=False, encoding="utf-8")

    total = len(merged_out)
    matched = int((merged_out["match_method"] != "unmatched").sum())
    unmatched = total - matched
    empty_label = int(merged_out["CategoryName"].isna().sum())
    outro_count = int((merged_out["CategoryName".lower() if False else "CategoryName"].astype(str) == "Outro").sum())

    qa = {
        "files": {
            "text_json": str(TEXT_JSON),
            "labels_csv": str(LABELS_CSV),
            "output_csv": str(OUT_CSV),
            "output_labeled_only_csv": str(OUT_LABELED_ONLY_CSV),
        },
        "counts": {
            "text_rows": int(len(text_df)),
            "labels_rows": int(len(labels_df)),
            "output_rows": int(total),
            "labeled_only_rows": int(len(merged_labeled_only)),
            "matched_rows": int(matched),
            "unmatched_rows": int(unmatched),
            "empty_category_rows": int(empty_label),
            "outro_rows": int(outro_count),
            "duplicate_conflict_groups": int(duplicate_conflict_groups),
        },
        "url_uniqueness": {
            "text_unique_url_norm": int(text_df["url_norm"].nunique()),
            "labels_unique_url_norm": int(labels_df["url_norm"].nunique()),
            "text_duplicate_rows": int(len(text_dups)),
            "labels_duplicate_rows": int(len(labels_dups)),
        },
        "unmatched_examples": merged.loc[merged["match_method"] == "unmatched", ["code", "municipality_name", "link"]]
        .head(30)
        .to_dict(orient="records"),
        "labels_duplicate_examples": labels_dups.head(30).to_dict(orient="records"),
        "text_duplicate_examples": text_dups.head(30).to_dict(orient="records"),
    }

    OUT_QA_JSON.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "Merge QA Summary",
        f"Text rows: {len(text_df)}",
        f"Labels rows: {len(labels_df)}",
        f"Output rows: {total}",
        f"Labeled-only rows: {len(merged_labeled_only)}",
        f"Matched rows: {matched}",
        f"Unmatched rows: {unmatched}",
        f"Empty CategoryName rows: {empty_label}",
        f"CategoryName='Outro' rows: {outro_count}",
        f"Duplicate URL groups with label conflict: {duplicate_conflict_groups}",
        f"Text duplicate rows by normalized URL: {len(text_dups)}",
        f"Labels duplicate rows by normalized URL: {len(labels_dups)}",
    ]
    OUT_QA_TXT.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("Merge complete.")
    print("- Output:", OUT_CSV)
    print("- Output labeled only:", OUT_LABELED_ONLY_CSV)
    print("- QA JSON:", OUT_QA_JSON)
    print("- QA TXT:", OUT_QA_TXT)
    print("- Matched:", matched, "/", total)


if __name__ == "__main__":
    main()

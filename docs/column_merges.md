# Column Merge Decisions — STATS19 Codes Dataset

These merges are applied in `merge_historic_cols()` inside `stats19_download.R`
to the **codes (numeric) CSV only**. The labels CSV is handled automatically by
`format_collisions()` inside the `stats19` R package.

---

## 1. `junction_detail_historic` → `junction_detail`

### Why it exists
DfT updated the junction detail coding scheme in the 2024 STATS19 specification.
The new file includes:
- `junction_detail` — populated with **new-spec codes** for 2024+ records
- `junction_detail_historic` — populated with **old-spec codes** as a fallback
  for records collected under the previous specification

### What the merge does
```
junction_detail = junction_detail  (use new value if present)
                  junction_detail_historic  (fill NA with old value)
junction_detail_historic  →  DROPPED
```

Concretely, for any row where `junction_detail` is `NA`, the value from
`junction_detail_historic` is used instead. After the merge, `junction_detail`
contains the most complete single column possible and the `_historic` column
is removed.

### Code scheme compatibility
The old and new coding schemes for junction detail are **compatible** — the
codes map directly across specs, so backfilling is safe without any remapping.

---

## 2. `pedestrian_crossing_human_control_historic` + `pedestrian_crossing_physical_facilities_historic` → dropped

### Background — two specs, three columns

**Pre-2024 spec** used two separate columns:

| Column | What it recorded |
|---|---|
| `pedestrian_crossing_human_control` | Was a person controlling the crossing? |
| `pedestrian_crossing_physical_facilities` | What physical infrastructure was present? |

**2024 spec** unified them into one column:

| Column | What it records |
|---|---|
| `pedestrian_crossing` | Combined coding covering both human control and physical type |

Old → New code mapping (for reference):

| Old `human_control` | Old `physical_facilities` | New `pedestrian_crossing` |
|---|---|---|
| 1 (school patrol) | any | **11** |
| 2 (other authorised) | any | **12** |
| 0 | 0 (none) | **0** |
| 0 | 1 (Zebra) | **13** |
| 0 | 4 (Pelican/Puffin/Toucan) | **14** |
| 0 | 5 (Ped phase at signal) | **15** |
| 0 | 7 (Footbridge/subway) | **16** |
| 0 | 8 (Central refuge) | **17** |
| 9 or -1 | 9 or -1 | **99 / -1** |

### Why we just drop them (no backfill needed)

DfT already applied this mapping when publishing the data — the `pedestrian_crossing`
column is **pre-populated in the raw CSV for all years (2020–2024)**, not just 2024.
The `_historic` columns are included purely as a reference alongside it.

Verified for 2022 (106,004 rows):
- `pedestrian_crossing` non-null: **105,320** rows (99.4 %)
- `pedestrian_crossing` null: **684** rows
- Of those 684 nulls — `_historic` columns are **also blank**: unrecoverable

Conclusion: the `_historic` columns add no recoverable information for ML.
Both are dropped. `pedestrian_crossing` is used as the single canonical column.

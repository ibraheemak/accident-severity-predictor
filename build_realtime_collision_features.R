# ── CONFIG ────────────────────────────────────────────────────────────────────
# Path to the codes collision CSV produced by stats19_download.R
INPUT_COLLISION_CSV <- "data/raw/stats19_20260606_181720/codes/stats19_20260606_181720_collisions.csv"

# Where to write the realtime features table
OUTPUT_DIR  <- "data/features"
OUTPUT_FILE <- "collision_realtime_features.csv"
# ──────────────────────────────────────────────────────────────────────────────

# ── Column groups (from docs/realtime features) ───────────────────────────────

# Kept for joining / traceability — never fed into the ML model
INFRASTRUCTURE_COLS <- c(
  "collision_index",
  "collision_year",
  "longitude",
  "latitude",
  "date",
  "data_year"
)

# ML feature columns
FEATURE_COLS <- c(
  "number_of_vehicles",
  "day_of_week",
  "time",
  "first_road_class",
  "first_road_number",
  "road_type",
  "speed_limit",
  "junction_detail",
  "junction_control",
  "second_road_class",
  "second_road_number",
  "pedestrian_crossing",
  "urban_or_rural_area",
  "trunk_road_flag"
)


# Target variable
TARGET_COL <- "collision_severity"

# Kept for comparison only — will be replaced by external weather API at inference
REPLACE_COLS <- c(
  "light_conditions",
  "weather_conditions"
)

# All columns to keep in the output (infrastructure + target + features + replace)
KEEP_COLS <- c(INFRASTRUCTURE_COLS, TARGET_COL, FEATURE_COLS, REPLACE_COLS)

# ── Packages ──────────────────────────────────────────────────────────────────
if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr", repos = "https://cloud.r-project.org", quiet = TRUE)
}
library(readr)

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
  if (!file.exists(INPUT_COLLISION_CSV)) {
    stop("Input file not found: ", INPUT_COLLISION_CSV,
         "\nUpdate INPUT_COLLISION_CSV in the CONFIG block.")
  }

  message("Reading: ", INPUT_COLLISION_CSV)
  col_df <- read_csv(INPUT_COLLISION_CSV, show_col_types = FALSE)
  message(sprintf("  Loaded %d rows x %d cols.", nrow(col_df), ncol(col_df)))

  # Verify all expected columns are present
  missing_cols <- setdiff(KEEP_COLS, names(col_df))
  if (length(missing_cols) > 0) {
    stop("Missing columns in input: ", paste(missing_cols, collapse = ", "))
  }

  features_df <- col_df[, KEEP_COLS]
  message(sprintf("  Selected %d columns.", ncol(features_df)))

  # ── Column summary ──
  message("\nColumn breakdown:")
  message("  Infrastructure (not for ML) : ", paste(INFRASTRUCTURE_COLS, collapse = ", "))
  message("  Target                      : ", TARGET_COL)
  message("  ML features                 : ", paste(FEATURE_COLS, collapse = ", "))
  message("  To replace (API at runtime) : ", paste(REPLACE_COLS, collapse = ", "))

  # ── Save ──
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(OUTPUT_DIR, OUTPUT_FILE)

  if (file.exists(out_path)) {
    stop("Output file already exists: ", out_path,
         "\nDelete it or change OUTPUT_FILE in the CONFIG block.")
  }

  message(sprintf("\nSaving to %s ...", out_path))
  write_csv(features_df, out_path)

  message("Done.")
  message(sprintf("  Rows    : %d", nrow(features_df)))
  message(sprintf("  Columns : %d  (%d infra  |  1 target  |  %d features  |  %d to-replace)",
                  ncol(features_df), length(INFRASTRUCTURE_COLS),
                  length(FEATURE_COLS), length(REPLACE_COLS)))

  # ── NA summary ──
  message("\nNA counts per column:")
  na_counts <- colSums(is.na(features_df))
  na_counts <- na_counts[na_counts > 0]
  if (length(na_counts) == 0) {
    message("  None.")
  } else {
    for (col in names(na_counts)) {
      pct <- round(na_counts[col] / nrow(features_df) * 100, 1)
      message(sprintf("  %-35s %d  (%.1f%%)", col, na_counts[col], pct))
    }
  }
}

main()

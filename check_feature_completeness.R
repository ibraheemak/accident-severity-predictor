#!/usr/bin/env Rscript
# check_feature_completeness.R
#
# Reports how many rows in the combined feature table have zero "true"
# missing values. Some NAs are structurally valid rather than missing data:
#   - dist_to_speed_camera_m is NA because there is no camera nearby
#     (has_speed_camera == 0)
#   - second_road_class / second_road_number / junction_control are NA
#     because the accident wasn't recorded at a junction
#     (junction_detail == 0, "Not at junction or within 20 metres")
# These are excused from the missingness count. Everything else (e.g. a
# camera existing but a tag missing, OSM way present but untagged) counts
# as genuine missing data.

suppressPackageStartupMessages({
  library(readr)
})

INPUT_FILE <- "data/features/stats19+aadf+osm/stats19+aadf+osm_combined_realtime_features.csv"

df <- read_csv(INPUT_FILE, show_col_types = FALSE)

# column -> condition (quoted expr, evaluated against df) under which an NA
# in that column is structurally valid and excused from the missing count.
excused <- list(
  dist_to_speed_camera_m = quote(has_speed_camera == 0),
  second_road_class      = quote(junction_detail == 0),
  second_road_number     = quote(junction_detail == 0),
  junction_control       = quote(junction_detail == 0)
)

feature_cols <- names(df)

unexcused_na <- matrix(FALSE, nrow = nrow(df), ncol = length(feature_cols),
                        dimnames = list(NULL, feature_cols))

for (col in feature_cols) {
  is_na <- is.na(df[[col]])
  if (col %in% names(excused)) {
    cond <- eval(excused[[col]], envir = df)
    cond[is.na(cond)] <- FALSE   # excusing column itself unknown -> don't excuse
    is_na <- is_na & !cond
  }
  unexcused_na[, col] <- is_na
}

row_has_unexcused_na <- apply(unexcused_na, 1, any)
n          <- nrow(df)
n_complete <- sum(!row_has_unexcused_na)

cat(sprintf("Total rows: %d\n", n))
cat(sprintf("Rows with NO unexcused missing values: %d (%.1f%%)\n",
            n_complete, 100 * n_complete / n))
cat(sprintf("Rows with at least one unexcused missing value: %d (%.1f%%)\n",
            n - n_complete, 100 * (n - n_complete) / n))

cat("\nPer-column unexcused missing counts:\n")
col_counts <- colSums(unexcused_na)
col_counts <- sort(col_counts[col_counts > 0], decreasing = TRUE)
for (col in names(col_counts)) {
  cat(sprintf("  %-28s %6d (%.1f%%)\n", col, col_counts[col], 100 * col_counts[col] / n))
}

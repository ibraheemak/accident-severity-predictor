#!/usr/bin/env Rscript
# check_split_stats.R
#
# Reports collision_year and collision_severity distributions for each of
# data/splits/{train,valid,test}.csv, so we can verify that every set
# contains all years and the same severity mix (proportionally) as the
# others. Writes a human-readable report to data/splits/split_stats.txt.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

SPLITS_DIR  <- "data/splits"
SPLIT_FILES <- c(train = "train.csv", valid = "valid.csv", test = "test.csv")
OUTPUT_FILE <- file.path(SPLITS_DIR, "split_stats.txt")

sets <- lapply(SPLIT_FILES, function(f) {
  read_csv(file.path(SPLITS_DIR, f), show_col_types = FALSE)
})

combined <- bind_rows(
  lapply(names(sets), function(nm) mutate(sets[[nm]], .split = nm))
)
combined$.split <- factor(combined$.split, levels = names(SPLIT_FILES))

lines <- c()
add <- function(...) lines <<- c(lines, sprintf(...))

add("Split sizes")
add("------------")
for (nm in names(sets)) {
  n <- nrow(sets[[nm]])
  add("%-6s %7d rows (%.1f%% of total)", nm, n, 100 * n / nrow(combined))
}
add("total  %7d rows", nrow(combined))
add("")

add("collision_year counts per split")
add("--------------------------------")
year_counts <- table(combined$collision_year, combined$.split)
print_table_to_lines <- function(tbl, label) {
  add("%s", label)
  hdr <- sprintf("%-10s %s", "", paste(sprintf("%10s", colnames(tbl)), collapse = ""))
  add("%s", hdr)
  for (r in rownames(tbl)) {
    row_vals <- tbl[r, ]
    add("%-10s %s", r, paste(sprintf("%10d", row_vals), collapse = ""))
  }
  add("")
}
print_table_to_lines(year_counts, "Counts:")

year_props <- prop.table(year_counts, margin = 2) * 100
add("Within-split year proportions (%%):")
hdr <- sprintf("%-10s %s", "", paste(sprintf("%10s", colnames(year_props)), collapse = ""))
add("%s", hdr)
for (r in rownames(year_props)) {
  row_vals <- year_props[r, ]
  add("%-10s %s", r, paste(sprintf("%10.2f", row_vals), collapse = ""))
}
add("")

add("collision_severity counts per split")
add("------------------------------------")
sev_counts <- table(combined$collision_severity, combined$.split)
print_table_to_lines(sev_counts, "Counts:")

sev_props <- prop.table(sev_counts, margin = 2) * 100
add("Within-split severity proportions (%%):")
hdr <- sprintf("%-10s %s", "", paste(sprintf("%10s", colnames(sev_props)), collapse = ""))
add("%s", hdr)
for (r in rownames(sev_props)) {
  row_vals <- sev_props[r, ]
  add("%-10s %s", r, paste(sprintf("%10.2f", row_vals), collapse = ""))
}
add("")

add("collision_year x collision_severity proportions within each split (%%)")
add("------------------------------------------------------------------------")
for (nm in names(sets)) {
  sub <- filter(combined, .split == nm)
  tbl <- prop.table(table(sub$collision_year, sub$collision_severity)) * 100
  add("Split: %s", nm)
  hdr <- sprintf("%-10s %s", "", paste(sprintf("%10s", colnames(tbl)), collapse = ""))
  add("%s", hdr)
  for (r in rownames(tbl)) {
    row_vals <- tbl[r, ]
    add("%-10s %s", r, paste(sprintf("%10.2f", row_vals), collapse = ""))
  }
  add("")
}

writeLines(lines, OUTPUT_FILE)
cat(paste(lines, collapse = "\n"))
cat(sprintf("\n\nReport written to %s\n", OUTPUT_FILE))

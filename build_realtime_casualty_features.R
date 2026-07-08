#!/usr/bin/env Rscript
# build_realtime_casualty_features.R
#
# Aggregates the casualties codes table to one row per collision_index,
# keeping ONLY what a 999 caller genuinely reports at dispatch time
# ("a pedestrian's been hit", "three people hurt", "an elderly woman"):
#
#   n_casualties             number of casualty rows for the collision
#   n_pedestrian_casualties  casualties with casualty_class == 3 (pedestrian)
#   youngest_casualty_age    min age_of_casualty (NA if none recorded)
#   oldest_casualty_age      max age_of_casualty (NA if none recorded)
#
# Explicitly EXCLUDED (label-adjacent / post-hoc / officer-assessed):
# casualty_severity, enhanced_casualty_severity, casualty_adjusted_*,
# casualty_injury_based, casualty_imd_decile, lsoa_of_casualty,
# pedestrian_location, pedestrian_movement.
#
# Then left-joins these 4 columns onto the current final feature table
# (stats19+aadf+osm+meteo+risk) -- the external enrichment is untouched --
# and writes the +cas combined table for split_train_valid_test.R.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

if (!exists("PIPELINE_DATA_DIR")) {
  PIPELINE_DATA_DIR <- "data/raw/stats19_20260706_193034"
}
INPUT_CASUALTY_CSV <- file.path(
  PIPELINE_DATA_DIR, "codes",
  paste0(basename(PIPELINE_DATA_DIR), "_casualties.csv")
)

CAS_FEATURES_FILE <- "data/features/stats19/stats19_casualty_realtime_features.csv"
COMBINED_IN  <- "data/features/stats19+aadf+osm+meteo+risk/stats19+aadf+osm+meteo+risk_combined_realtime_features.csv"
COMBINED_DIR <- "data/features/stats19+aadf+osm+meteo+risk+cas"
COMBINED_OUT <- file.path(COMBINED_DIR, "stats19+aadf+osm+meteo+risk+cas_combined_realtime_features.csv")

# --- 1. aggregate casualties -> one row per collision --------------------------
cas <- read_csv(INPUT_CASUALTY_CSV, show_col_types = FALSE,
                col_select = c(collision_index, casualty_class, age_of_casualty))
cat(sprintf("Loaded %d casualty rows (%d collisions) from %s\n",
            nrow(cas), n_distinct(cas$collision_index), INPUT_CASUALTY_CSV))

# Negative age codes mean "not recorded" in some spec years -- treat as NA.
cas$age_of_casualty[!is.na(cas$age_of_casualty) & cas$age_of_casualty < 0] <- NA_real_

agg <- cas |>
  group_by(collision_index) |>
  summarise(
    n_casualties            = n(),
    n_pedestrian_casualties = sum(casualty_class == 3, na.rm = TRUE),
    youngest_casualty_age   = ifelse(all(is.na(age_of_casualty)), NA_real_,
                                     min(age_of_casualty, na.rm = TRUE)),
    oldest_casualty_age     = ifelse(all(is.na(age_of_casualty)), NA_real_,
                                     max(age_of_casualty, na.rm = TRUE)),
    .groups = "drop"
  )

dir.create(dirname(CAS_FEATURES_FILE), showWarnings = FALSE, recursive = TRUE)
write_csv(agg, CAS_FEATURES_FILE)
cat(sprintf("Casualty features: %d rows x %d cols -> %s\n",
            nrow(agg), ncol(agg), CAS_FEATURES_FILE))

cat("\nFeature summary:\n")
cat(sprintf("  n_casualties            median=%d  max=%d\n",
            median(agg$n_casualties), max(agg$n_casualties)))
cat(sprintf("  n_pedestrian_casualties >0 in %d collisions (%.1f%%)\n",
            sum(agg$n_pedestrian_casualties > 0),
            100 * mean(agg$n_pedestrian_casualties > 0)))
cat(sprintf("  youngest_casualty_age   median=%.0f  NA=%d\n",
            median(agg$youngest_casualty_age, na.rm = TRUE),
            sum(is.na(agg$youngest_casualty_age))))
cat(sprintf("  oldest_casualty_age     median=%.0f  NA=%d\n",
            median(agg$oldest_casualty_age, na.rm = TRUE),
            sum(is.na(agg$oldest_casualty_age))))

# --- 2. join onto the final combined table -------------------------------------
combined <- read_csv(COMBINED_IN, show_col_types = FALSE)
cat(sprintf("\nCombined table: %d rows x %d cols\n", nrow(combined), ncol(combined)))

out <- left_join(combined, agg, by = "collision_index")

n_missing <- sum(is.na(out$n_casualties))
if (n_missing > 0) {
  stop(sprintf("%d collisions have no casualty rows -- unexpected (every STATS19 collision has >= 1 casualty).", n_missing))
}

dir.create(COMBINED_DIR, showWarnings = FALSE, recursive = TRUE)
write_csv(out, COMBINED_OUT)
cat(sprintf("Written %d rows x %d cols -> %s\n", nrow(out), ncol(out), COMBINED_OUT))

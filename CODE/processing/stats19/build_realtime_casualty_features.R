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
#   n_driver_cas             casualties with casualty_class == 1 (driver/rider)
#   n_passenger_cas          casualties with casualty_class == 2 (passenger)
#   n_child_cas              casualties aged 0-15
#   n_elderly_cas            casualties aged 70+
#   n_motorcyclist_cas       casualty_type in {2,3,4,5,23,97} (motorcyclist)
#   n_cyclist_cas            casualty_type == 1 (pedal cyclist)
#   n_female_cas             sex_of_casualty == 2
#
# (The 7 count features after oldest_casualty_age were added 2026-07-17: a
# 999 caller describes who is hurt -- "the motorcyclist", "a child", "my
# wife" -- so the class/age-band/type structure of the casualty list is
# dispatch-time information just like the original four aggregates.
# Measured on the binary normalized model: +0.002 valid macro F1 alone,
# the largest single contribution of the 2026-07-17 feature round.)
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
  PIPELINE_DATA_DIR <- "data/raw/stats19/stats19_20260706_193034"
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
                col_select = c(collision_index, casualty_class, age_of_casualty,
                               sex_of_casualty, casualty_type))
cat(sprintf("Loaded %d casualty rows (%d collisions) from %s\n",
            nrow(cas), n_distinct(cas$collision_index), INPUT_CASUALTY_CSV))

# Negative age codes mean "not recorded" in some spec years -- treat as NA.
cas$age_of_casualty[!is.na(cas$age_of_casualty) & cas$age_of_casualty < 0] <- NA_real_

# vehicle_type groupings mirror build_realtime_vehicle_features.R
MOTORCYCLE_CAS_TYPES <- c(2, 3, 4, 5, 23, 97)

agg <- cas |>
  group_by(collision_index) |>
  summarise(
    n_casualties            = n(),
    n_pedestrian_casualties = sum(casualty_class == 3, na.rm = TRUE),
    youngest_casualty_age   = ifelse(all(is.na(age_of_casualty)), NA_real_,
                                     min(age_of_casualty, na.rm = TRUE)),
    oldest_casualty_age     = ifelse(all(is.na(age_of_casualty)), NA_real_,
                                     max(age_of_casualty, na.rm = TRUE)),
    n_driver_cas            = sum(casualty_class == 1, na.rm = TRUE),
    n_passenger_cas         = sum(casualty_class == 2, na.rm = TRUE),
    n_child_cas             = sum(!is.na(age_of_casualty) & age_of_casualty < 16),
    n_elderly_cas           = sum(!is.na(age_of_casualty) & age_of_casualty >= 70),
    n_motorcyclist_cas      = sum(casualty_type %in% MOTORCYCLE_CAS_TYPES, na.rm = TRUE),
    n_cyclist_cas           = sum(casualty_type == 1, na.rm = TRUE),
    n_female_cas            = sum(sex_of_casualty == 2, na.rm = TRUE),
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

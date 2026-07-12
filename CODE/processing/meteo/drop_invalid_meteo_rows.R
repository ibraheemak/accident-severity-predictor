#!/usr/bin/env Rscript
# drop_invalid_meteo_rows.R
#
# Removes accident rows that have no usable lat/lon and therefore could never
# be matched to an Open-Meteo weather observation (the weather columns are
# structurally NA for these, not missing-by-chance). Same accidents that the
# OSM/AADF steps also had nothing to look up for.
#
# Usage:
#   Rscript drop_invalid_meteo_rows.R

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

INPUT_FILE <- "data/features/stats19+aadf+osm+meteo/stats19+aadf+osm+meteo_combined_realtime_features.csv"

df <- read_csv(INPUT_FILE, show_col_types = FALSE)
cat(sprintf("Loaded %d rows x %d cols from %s\n", nrow(df), ncol(df), INPUT_FILE))

invalid <- is.na(df$temperature_2m_c)
cat(sprintf("Dropping %d rows with no weather data (missing lat/lon/date/time):\n", sum(invalid)))
print(df$collision_index[invalid])

df_clean <- df[!invalid, ]

write_csv(df_clean, INPUT_FILE)
cat(sprintf("Written %d rows x %d cols to %s\n", nrow(df_clean), ncol(df_clean), INPUT_FILE))

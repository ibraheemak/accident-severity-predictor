#!/usr/bin/env Rscript
# normalize_splits.R
#
# Preprocessing pipeline (run before generate_masked_splits.R):
#   1. Cyclic encoding  -- day_of_week -> _sin/_cos; time -> hour_sin/hour_cos
#   2. log1p            -- applied to non-negative columns with skewness > 2
#   3. Z-score          -- mean/SD from train only, applied to all splits
#
# Outputs: data/splits/{train,valid,test}_norm.csv + normalization_stats.csv

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

SPLIT_DIR      <- "data/splits"
NORM_DIR       <- file.path(SPLIT_DIR, "normalized")
SPLITS         <- c("train", "valid", "test")
LOG1P_SKEW_THRESHOLD <- 2.0

# Columns to z-score (excludes binary flags, metadata, target, cyclic cols
# handled separately below, and nominal/categorical codes -- see
# NOMINAL_COLS -- which pass through as raw codes for XGBoost's native
# categorical split support instead of being treated as ordinal).
ZSCORE_COLS <- c(
  "number_of_vehicles", "first_road_number", "speed_limit",
  "second_road_number",
  "aadf_traffic_volume", "aadf_distance_m", "dist_to_speed_camera_m",
  "lane_count", "temperature_2m_c", "precipitation_mm", "wind_speed_10m_kmh",
  "snowfall_cm", "snow_depth_m",
  "hist_acc_rate_500m", "hist_severe_frac_500m", "hist_fatal_rate_500m",
  "dist_to_nearest_hist_acc_m",
  "hist_acc_rate_250m", "hist_severe_frac_250m",
  "hist_acc_rate_1500m", "hist_severe_frac_1500m", "dist_to_nearest_severe_m",
  "n_casualties", "n_pedestrian_casualties", "youngest_casualty_age",
  "oldest_casualty_age",
  "n_driver_cas", "n_passenger_cas", "n_child_cas", "n_elderly_cas",
  "n_motorcyclist_cas", "n_cyclist_cas", "n_female_cas"
)

# Nominal/unordered category codes: left as raw integers (NaNs preserved for
# genuine missingness and for masking downstream) rather than z-scored, so
# XGBoost can use native categorical splits instead of treating them as if
# they had a numeric order.
NOMINAL_COLS <- c(
  "first_road_class", "road_type", "junction_detail", "junction_control",
  "second_road_class", "pedestrian_crossing", "urban_or_rural_area",
  "light_conditions", "weather_conditions", "weather_code"
)

# --- helpers -----------------------------------------------------------------

skewness <- function(x) {
  x <- x[!is.na(x)]
  n <- length(x)
  m <- mean(x)
  s <- sd(x)
  (n / ((n - 1) * (n - 2))) * sum(((x - m) / s)^3)
}

cyclic_encode <- function(x, period) {
  angle <- 2 * pi * x / period
  list(sin = sin(angle), cos = cos(angle))
}

extract_hour <- function(time_str) {
  # Parses "HH:MM:SS" or "HH:MM"; returns integer hour 0-23.
  as.integer(sub("^(\\d{1,2}):.*", "\\1", time_str))
}

# --- load splits -------------------------------------------------------------

dfs <- lapply(SPLITS, function(s) {
  read_csv(file.path(SPLIT_DIR, paste0(s, ".csv")), show_col_types = FALSE)
})
names(dfs) <- SPLITS

stopifnot(all(ZSCORE_COLS %in% names(dfs$train)))
stopifnot(all(NOMINAL_COLS %in% names(dfs$train)))
stopifnot(all(c("day_of_week", "time") %in% names(dfs$train)))

# --- 1. cyclic encoding (applied uniformly, no train-fit needed) -------------

for (split in SPLITS) {
  df <- dfs[[split]]

  enc_dow  <- cyclic_encode(df$day_of_week - 1, 7)   # shift to 0-based
  enc_hour <- cyclic_encode(extract_hour(df$time), 24)

  df$day_of_week_sin <- enc_dow$sin
  df$day_of_week_cos <- enc_dow$cos
  df$hour_sin        <- enc_hour$sin
  df$hour_cos        <- enc_hour$cos

  df$day_of_week <- NULL
  df$time        <- NULL

  dfs[[split]] <- df
}

cat("Cyclic encoding done: day_of_week -> _sin/_cos, time -> hour_sin/hour_cos\n")

# --- 2. log1p for skewed non-negative columns (fit on train) -----------------

train_skew <- sapply(ZSCORE_COLS, function(col) skewness(dfs$train[[col]]))
train_min  <- sapply(ZSCORE_COLS, function(col) min(dfs$train[[col]], na.rm = TRUE))

log1p_cols <- ZSCORE_COLS[train_skew > LOG1P_SKEW_THRESHOLD & train_min >= 0]

for (split in SPLITS) {
  df <- dfs[[split]]
  for (col in log1p_cols) df[[col]] <- log1p(df[[col]])
  dfs[[split]] <- df
}

cat(sprintf("log1p applied to %d columns (skew > %.1f, min >= 0):\n  %s\n",
    length(log1p_cols), LOG1P_SKEW_THRESHOLD, paste(log1p_cols, collapse = ", ")))

# --- 3. z-score (mean/SD from train only) ------------------------------------

train_mean <- sapply(ZSCORE_COLS, function(col) mean(dfs$train[[col]], na.rm = TRUE))
train_sd   <- sapply(ZSCORE_COLS, function(col) sd(dfs$train[[col]],   na.rm = TRUE))

dir.create(NORM_DIR, showWarnings = FALSE, recursive = TRUE)

zero_sd_cols <- ZSCORE_COLS[train_sd == 0]
if (length(zero_sd_cols) > 0)
  stop(sprintf("Zero SD on train for: %s -- cannot z-score.", paste(zero_sd_cols, collapse = ", ")))

stats <- tibble(
  feature     = ZSCORE_COLS,
  log1p_first = ZSCORE_COLS %in% log1p_cols,
  mean        = train_mean,
  sd          = train_sd
)
write_csv(stats, file.path(NORM_DIR, "normalization_stats.csv"))

for (split in SPLITS) {
  df <- dfs[[split]]
  for (i in seq_along(ZSCORE_COLS)) {
    col <- ZSCORE_COLS[i]
    df[[col]] <- (df[[col]] - train_mean[i]) / train_sd[i]
  }
  out_path <- file.path(NORM_DIR, paste0(split, "_norm.csv"))
  write_csv(df, out_path)
  cat(sprintf("%s: written -> %s\n", split, out_path))
}

cat(sprintf("\nNormalization stats written to %s/normalization_stats.csv\n", NORM_DIR))

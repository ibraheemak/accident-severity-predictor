#!/usr/bin/env Rscript
# add_risk_density_features.R
#
# Adds spatial risk-density features: for each accident, how much accident
# history does this location have? Dangerous locations stay dangerous, and
# none of the existing features capture location-specific risk directly.
#
# Leakage design: for an accident in year Y, features are computed ONLY from
# accidents in years strictly < Y -- never the accident itself, never
# same-year or future accidents. Past accidents are public knowledge at
# 999-call time, so these features are legitimately available at inference.
# 2016 rows (no prior years in our data) get NA for all four columns.
#
# Known bias, accepted: the dataset is severity-sampled (all Fatal, capped
# Serious/Slight per year), so hist_severe_frac_500m is inflated relative to
# the true population fraction -- but the sampling fraction is constant
# across space within each year, so the *relative* variation between
# locations (what the model actually uses) is preserved.
#
# New columns:
#   hist_acc_rate_500m          prior-year accidents within 500m / n prior years
#   hist_severe_frac_500m       fraction of those that were Fatal or Serious
#                               (NA if none nearby)
#   hist_fatal_rate_500m        prior-year Fatal accidents within 500m / n prior years
#   dist_to_nearest_hist_acc_m  distance to nearest prior-year accident

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(sf)
})

INPUT_FILE  <- "data/features/stats19+aadf+osm+meteo/stats19+aadf+osm+meteo_combined_realtime_features.csv"
OUTPUT_DIR  <- "data/features/stats19+aadf+osm+meteo+risk"
OUTPUT_FILE <- file.path(OUTPUT_DIR, "stats19+aadf+osm+meteo+risk_combined_realtime_features.csv")
RADIUS_M    <- 500

df <- read_csv(INPUT_FILE, show_col_types = FALSE)
cat(sprintf("Loaded %d rows x %d cols from %s\n", nrow(df), ncol(df), INPUT_FILE))

df$hist_acc_rate_500m         <- NA_real_
df$hist_severe_frac_500m      <- NA_real_
df$hist_fatal_rate_500m       <- NA_real_
df$dist_to_nearest_hist_acc_m <- NA_real_

valid <- !is.na(df$longitude) & !is.na(df$latitude)
cat(sprintf("%d rows have valid coordinates (%d missing -> features stay NA)\n",
            sum(valid), sum(!valid)))

# Project once to metres (British National Grid), same as join_aadf.R.
pts <- st_as_sf(df[valid, c("longitude", "latitude")],
                coords = c("longitude", "latitude"), crs = 4326) |>
  st_transform(27700)
coords_year     <- df$collision_year[valid]
coords_severity <- df$collision_severity[valid]
row_idx         <- which(valid)   # position of each pts row in df

years <- sort(unique(coords_year))
first_year <- min(years)

for (y in years) {
  if (y == first_year) {
    cat(sprintf("[%d] first year -- no history, all features NA\n", y))
    next
  }
  cur   <- coords_year == y
  prior <- coords_year < y
  n_prior_years <- y - first_year

  cur_pts   <- pts[cur, ]
  prior_pts <- pts[prior, ]
  prior_sev <- coords_severity[prior]

  cat(sprintf("[%d] %d accidents vs %d prior accidents (%d prior years) ... ",
              y, nrow(cur_pts), nrow(prior_pts), n_prior_years))

  nbrs <- st_is_within_distance(cur_pts, prior_pts, dist = RADIUS_M)

  n_nearby      <- lengths(nbrs)
  n_severe      <- vapply(nbrs, function(i) sum(prior_sev[i] %in% c(1, 2)), integer(1))
  n_fatal       <- vapply(nbrs, function(i) sum(prior_sev[i] == 1), integer(1))

  nearest_idx  <- st_nearest_feature(cur_pts, prior_pts)
  nearest_dist <- as.numeric(st_distance(cur_pts, prior_pts[nearest_idx, ], by_element = TRUE))

  idx <- row_idx[cur]
  df$hist_acc_rate_500m[idx]         <- n_nearby / n_prior_years
  df$hist_severe_frac_500m[idx]      <- ifelse(n_nearby > 0, n_severe / n_nearby, NA_real_)
  df$hist_fatal_rate_500m[idx]       <- n_fatal / n_prior_years
  df$dist_to_nearest_hist_acc_m[idx] <- nearest_dist

  cat(sprintf("median rate=%.2f/yr, severe_frac=%.2f, nearest=%.0fm\n",
              median(n_nearby / n_prior_years),
              median(ifelse(n_nearby > 0, n_severe / n_nearby, NA_real_), na.rm = TRUE),
              median(nearest_dist)))
}

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
write_csv(df, OUTPUT_FILE)
cat(sprintf("\nWritten %d rows x %d cols to %s\n", nrow(df), ncol(df), OUTPUT_FILE))

# --- summary ------------------------------------------------------------------
cat("\nFeature summary (non-NA rows):\n")
for (col in c("hist_acc_rate_500m", "hist_severe_frac_500m",
              "hist_fatal_rate_500m", "dist_to_nearest_hist_acc_m")) {
  x <- df[[col]]
  cat(sprintf("  %-28s n=%d  min=%.2f  median=%.2f  mean=%.2f  max=%.2f  NA=%d\n",
              col, sum(!is.na(x)), min(x, na.rm = TRUE), median(x, na.rm = TRUE),
              mean(x, na.rm = TRUE), max(x, na.rm = TRUE), sum(is.na(x))))
}
urban <- df$urban_or_rural_area == 1 & !is.na(df$hist_acc_rate_500m)
rural <- df$urban_or_rural_area == 2 & !is.na(df$hist_acc_rate_500m)
cat(sprintf("\nSanity: median hist_acc_rate_500m urban=%.2f vs rural=%.2f (expect urban >> rural)\n",
            median(df$hist_acc_rate_500m[urban], na.rm = TRUE),
            median(df$hist_acc_rate_500m[rural], na.rm = TRUE)))

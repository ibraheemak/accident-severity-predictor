#!/usr/bin/env Rscript
# analyze_meteo_features.R
#
# Reports how the Open-Meteo weather features are distributed: summary
# statistics, quantiles, and a text histogram for the continuous variables,
# plus a frequency table for the categorical WMO weather_code.
#
# Usage:
#   Rscript analyze_meteo_features.R

suppressPackageStartupMessages({
  library(readr)
})

INPUT_FILE <- "data/features/stats19+aadf+osm+meteo/stats19+aadf+osm+meteo_combined_realtime_features.csv"

df <- read_csv(INPUT_FILE, show_col_types = FALSE)
n  <- nrow(df)
cat(sprintf("Loaded %d rows from %s\n\n", n, INPUT_FILE))

CONTINUOUS_VARS <- c(
  temperature_2m_c   = "Temperature at 2m (°C)",
  precipitation_mm   = "Precipitation (mm/hour)",
  wind_speed_10m_kmh = "Wind speed at 10m (km/h)",
  snowfall_cm        = "Snowfall (cm)",
  snow_depth_m       = "Snow depth (m)"
)

# WMO weather codes that actually appear in the data, named for readability.
WMO_LABELS <- c(
  `0` = "Clear sky", `1` = "Mainly clear", `2` = "Partly cloudy", `3` = "Overcast",
  `45` = "Fog", `48` = "Freezing fog",
  `51` = "Light drizzle", `53` = "Moderate drizzle", `55` = "Dense drizzle",
  `56` = "Light freezing drizzle", `57` = "Dense freezing drizzle",
  `61` = "Slight rain", `63` = "Moderate rain", `65` = "Heavy rain",
  `66` = "Light freezing rain", `67` = "Heavy freezing rain",
  `71` = "Slight snowfall", `73` = "Moderate snowfall", `75` = "Heavy snowfall",
  `77` = "Snow grains",
  `80` = "Slight rain showers", `81` = "Moderate rain showers", `82` = "Violent rain showers",
  `85` = "Slight snow showers", `86` = "Heavy snow showers",
  `95` = "Thunderstorm", `96` = "Thunderstorm + slight hail", `99` = "Thunderstorm + heavy hail"
)

# ── Text histogram for a continuous variable ───────────────────────────────────
print_histogram <- function(x, n_bins = 12, bar_width = 50) {
  x <- x[!is.na(x)]
  if (length(x) == 0) { cat("  (no observations)\n"); return(invisible(NULL)) }
  if (diff(range(x)) == 0) {
    cat(sprintf("  all values equal %.2f (n=%d)\n", x[1], length(x)))
    return(invisible(NULL))
  }
  breaks <- pretty(x, n = n_bins)
  bins   <- cut(x, breaks = breaks, include.lowest = TRUE)
  tbl    <- table(bins)
  max_count <- max(tbl)
  for (i in seq_along(tbl)) {
    bar_len <- round(bar_width * as.integer(tbl[i]) / max_count)
    cat(sprintf("  %-22s %6d  %s\n",
      names(tbl)[i], as.integer(tbl[i]), strrep("#", bar_len)))
  }
}

# ── Summary stats + histogram per continuous variable ──────────────────────────
for (col in names(CONTINUOUS_VARS)) {
  x <- df[[col]]
  k_missing <- sum(is.na(x))
  cat(sprintf("== %s (%s) ==\n", CONTINUOUS_VARS[[col]], col))
  cat(sprintf("  n=%d  missing=%d (%.1f%%)\n", n, k_missing, 100 * k_missing / n))
  xv <- x[!is.na(x)]
  if (length(xv) > 0) {
    q <- quantile(xv, probs = c(0, .25, .5, .75, .9, .99, 1))
    cat(sprintf("  min=%.2f  Q1=%.2f  median=%.2f  Q3=%.2f  P90=%.2f  P99=%.2f  max=%.2f\n",
      q[1], q[2], q[3], q[4], q[5], q[6], q[7]))
    cat(sprintf("  mean=%.2f  sd=%.2f\n", mean(xv), sd(xv)))
    cat(sprintf("  zero values: %d (%.1f%% of non-missing)\n",
      sum(xv == 0), 100 * mean(xv == 0)))
    cat("  distribution:\n")
    print_histogram(xv)
  }
  cat("\n")
}

# ── Categorical: weather_code frequency table ───────────────────────────────────
cat("== WMO weather_code (categorical) ==\n")
wc <- df$weather_code
k_missing <- sum(is.na(wc))
cat(sprintf("  n=%d  missing=%d (%.1f%%)\n", n, k_missing, 100 * k_missing / n))
tbl <- sort(table(wc), decreasing = TRUE)
for (code in names(tbl)) {
  label <- WMO_LABELS[[code]]
  if (is.null(label)) label <- "(unlabelled code)"
  count <- as.integer(tbl[[code]])
  cat(sprintf("  %-3s %-28s %6d  (%.1f%%)\n", code, label, count, 100 * count / n))
}
cat("\n")

# ── Correlations among continuous meteo features ───────────────────────────────
cat("== Pairwise correlations (continuous meteo features) ==\n")
cont_df <- df[, names(CONTINUOUS_VARS)]
cor_mat <- cor(cont_df, use = "pairwise.complete.obs")
print(round(cor_mat, 2))

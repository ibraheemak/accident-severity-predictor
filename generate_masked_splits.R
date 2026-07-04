#!/usr/bin/env Rscript
# generate_masked_splits.R
#
# Run normalize_splits.R first -- this script masks the already-normalized
# *_norm.csv splits, not the raw splits, so masked-out values don't carry
# unnormalized scale into the training data.
#
# For each split (train/valid/test), expands every row into:
#   - 1 unmasked copy   (copy_id = -1, mask_rate = 0.0)
#   - 10 masked copies  (copy_id = 0..9)
# Masked copies have a random subset of the "acquirable" STATS19 fields set
# to NA, simulating partial feature observation at 999-call time. The number
# of fields masked per copy (0..27) is sampled without replacement per row,
# so each row gets 10 copies with 10 distinct mask counts, and the specific
# fields masked are chosen independently for each copy.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

SEED      <- 42
SPLIT_DIR <- "data/splits"
NORM_DIR  <- file.path(SPLIT_DIR, "normalized")
MASK_DIR  <- file.path(SPLIT_DIR, "masked")
SPLITS    <- c("train", "valid", "test")

# Always available at call time -- never masked.
# day_of_week and time are replaced by their cyclic encodings from normalize_splits.R.
XBASE_COLS <- c(
  "collision_index", "collision_year", "longitude", "latitude", "date",
  "data_year", "collision_severity",
  "day_of_week_sin", "day_of_week_cos", "hour_sin", "hour_cos"
)

# STATS19 fields that may or may not be acquired -- candidates for masking.
ACQUIRABLE_COLS <- c(
  "number_of_vehicles", "first_road_class", "first_road_number", "road_type",
  "speed_limit", "junction_detail", "junction_control", "second_road_class",
  "second_road_number", "pedestrian_crossing", "urban_or_rural_area",
  "trunk_road_flag", "light_conditions", "has_motorcycle", "has_car",
  "has_hgv", "has_pedal_cycle", "has_minibus", "has_bus_coach",
  "has_ridden_horse", "has_agricultural", "has_tram", "has_van",
  "has_mobility_scooter", "has_other_vehicle", "has_unknown_vehicle",
  "has_towing", "has_escooter"
)
N_ACQUIRABLE <- length(ACQUIRABLE_COLS)

mask_split <- function(df) {
  n <- nrow(df)

  # Per row, 10 distinct mask counts (0..28) assigned to copy_id 0..9.
  counts <- t(sapply(seq_len(n), function(i) sample(0:N_ACQUIRABLE, 10)))

  # Expand: each row -> 11 copies, ordered (-1, 0, 1, ..., 9).
  expanded_idx <- rep(seq_len(n), each = 11)
  expanded <- df[expanded_idx, ]
  rownames(expanded) <- NULL

  copy_id <- rep(c(-1L, 0:9), times = n)
  mask_rate_mat <- cbind(0, counts / N_ACQUIRABLE)
  mask_rate <- as.vector(t(mask_rate_mat))

  # For each (row, copy 0..9) pick exactly `count` of the 28 fields to mask.
  kflat <- as.vector(t(counts))
  mask_mat_copies <- matrix(FALSE, nrow = length(kflat), ncol = N_ACQUIRABLE)
  for (j in seq_along(kflat)) {
    k <- kflat[j]
    if (k > 0) mask_mat_copies[j, sample.int(N_ACQUIRABLE, k)] <- TRUE
  }

  full_mask <- matrix(FALSE, nrow = nrow(expanded), ncol = N_ACQUIRABLE)
  full_mask[copy_id != -1L, ] <- mask_mat_copies

  for (j in seq_along(ACQUIRABLE_COLS)) {
    col <- ACQUIRABLE_COLS[j]
    expanded[[col]][full_mask[, j]] <- NA
  }

  expanded$copy_id <- copy_id
  expanded$mask_rate <- mask_rate

  list(expanded = expanded, mask_counts = kflat, mask_mat = mask_mat_copies)
}

mask_count_distribution <- function(mask_counts) {
  tibble(
    mask_count = 0:N_ACQUIRABLE,
    n_copies = as.integer(table(factor(mask_counts, levels = 0:N_ACQUIRABLE)))
  ) %>%
    mutate(
      pct_of_masked_copies = round(100 * n_copies / length(mask_counts), 3)
    )
}

feature_mask_frequency <- function(mask_mat) {
  tibble(
    feature = ACQUIRABLE_COLS,
    n_times_masked = colSums(mask_mat),
    n_masked_copies = nrow(mask_mat)
  ) %>%
    mutate(
      pct_of_masked_copies = round(100 * n_times_masked / n_masked_copies, 3)
    ) %>%
    arrange(desc(n_times_masked))
}

set.seed(SEED)

dir.create(MASK_DIR, showWarnings = FALSE, recursive = TRUE)
stats_dir <- file.path(MASK_DIR, "mask_stats")
dir.create(stats_dir, showWarnings = FALSE, recursive = TRUE)

all_mask_counts <- list()
all_mask_mats <- list()

for (split in SPLITS) {
  in_path <- file.path(NORM_DIR, paste0(split, "_norm.csv"))
  out_path <- file.path(MASK_DIR, paste0(split, "_masked.csv"))

  df <- read_csv(in_path, show_col_types = FALSE)
  stopifnot(all(c(XBASE_COLS, ACQUIRABLE_COLS) %in% names(df)))

  result <- mask_split(df)
  write_csv(result$expanded, out_path)

  cat(sprintf(
    "%s: %d rows -> %d rows (%d originals x 11) written to %s\n",
    split, nrow(df), nrow(result$expanded), nrow(df), out_path
  ))

  write_csv(
    mask_count_distribution(result$mask_counts),
    file.path(stats_dir, paste0(split, "_mask_count_distribution.csv"))
  )
  write_csv(
    feature_mask_frequency(result$mask_mat),
    file.path(stats_dir, paste0(split, "_feature_mask_frequency.csv"))
  )

  all_mask_counts[[split]] <- result$mask_counts
  all_mask_mats[[split]] <- result$mask_mat
}

# Combined stats across all splits.
combined_counts <- unlist(all_mask_counts, use.names = FALSE)
combined_mat <- do.call(rbind, all_mask_mats)

write_csv(
  mask_count_distribution(combined_counts),
  file.path(stats_dir, "combined_mask_count_distribution.csv")
)
write_csv(
  feature_mask_frequency(combined_mat),
  file.path(stats_dir, "combined_feature_mask_frequency.csv")
)

cat(sprintf("\nMasked splits written to %s/\n", MASK_DIR))
cat(sprintf("Mask statistics written to %s/\n", stats_dir))

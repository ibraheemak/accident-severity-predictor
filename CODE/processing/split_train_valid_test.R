#!/usr/bin/env Rscript
# split_train_valid_test.R
#
# Splits the final combined feature table into train/validation/test sets,
# stratified by collision_year x collision_severity so that every set
# contains data from every year and the severity mix is the same across
# sets (no set is skewed toward a particular severity or missing a year).

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

INPUT_FILE  <- "data/features/stats19+aadf+osm+meteo+risk+cas/stats19+aadf+osm+meteo+risk+cas_combined_realtime_features.csv"
OUTPUT_DIR  <- "data/splits"
TRAIN_FRAC  <- 0.70
VALID_FRAC  <- 0.15
TEST_FRAC   <- 0.15
SEED        <- 42

stopifnot(abs(TRAIN_FRAC + VALID_FRAC + TEST_FRAC - 1) < 1e-9)

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

df <- read_csv(INPUT_FILE, show_col_types = FALSE)

set.seed(SEED)

df$.row_id <- seq_len(nrow(df))
df$.split  <- NA_character_

strata <- split(df$.row_id, list(df$collision_year, df$collision_severity), drop = TRUE)

for (ids in strata) {
  ids <- sample(ids)
  n <- length(ids)
  n_train <- floor(n * TRAIN_FRAC)
  n_valid <- floor(n * VALID_FRAC)
  df$.split[ids[seq_len(n_train)]] <- "train"
  df$.split[ids[n_train + seq_len(n_valid)]] <- "valid"
  df$.split[ids[(n_train + n_valid + 1):n]] <- "test"
}

train <- select(filter(df, .split == "train"), -.row_id, -.split)
valid <- select(filter(df, .split == "valid"), -.row_id, -.split)
test  <- select(filter(df, .split == "test"),  -.row_id, -.split)

write_csv(train, file.path(OUTPUT_DIR, "train.csv"))
write_csv(valid, file.path(OUTPUT_DIR, "valid.csv"))
write_csv(test,  file.path(OUTPUT_DIR, "test.csv"))

cat(sprintf("Total rows: %d\n", nrow(df)))
cat(sprintf("train: %d (%.1f%%)\n", nrow(train), 100 * nrow(train) / nrow(df)))
cat(sprintf("valid: %d (%.1f%%)\n", nrow(valid), 100 * nrow(valid) / nrow(df)))
cat(sprintf("test:  %d (%.1f%%)\n", nrow(test),  100 * nrow(test)  / nrow(df)))

cat("\nRows per year x severity (train / valid / test):\n")
check <- bind_rows(
  mutate(train, .split = "train"),
  mutate(valid, .split = "valid"),
  mutate(test,  .split = "test")
)
print(table(check$collision_year, check$collision_severity, check$.split))

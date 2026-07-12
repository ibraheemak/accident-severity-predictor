# ── CONFIG ────────────────────────────────────────────────────────────────────
COLLISION_FEATURES_CSV <- "data/features/stats19/stats19_collision_realtime_features.csv"
VEHICLE_FEATURES_CSV   <- "data/features/stats19/stats19_vehicle_realtime_features.csv"

OUTPUT_DIR  <- "data/features/stats19"
OUTPUT_FILE <- "stats19_combined_realtime_features.csv"
# ──────────────────────────────────────────────────────────────────────────────

# ── Package setup ─────────────────────────────────────────────────────────────
for (pkg in c("readr", "dplyr")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  library(pkg, character.only = TRUE)
}

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
  for (f in c(COLLISION_FEATURES_CSV, VEHICLE_FEATURES_CSV)) {
    if (!file.exists(f)) stop("Input file not found: ", f)
  }

  message("Reading: ", COLLISION_FEATURES_CSV)
  col_df <- read_csv(COLLISION_FEATURES_CSV, show_col_types = FALSE)
  message(sprintf("  %d rows x %d cols.", nrow(col_df), ncol(col_df)))

  message("Reading: ", VEHICLE_FEATURES_CSV)
  veh_df <- read_csv(VEHICLE_FEATURES_CSV, show_col_types = FALSE)
  message(sprintf("  %d rows x %d cols.", nrow(veh_df), ncol(veh_df)))

  message("\nJoining on collision_index...")
  combined <- col_df |> left_join(veh_df, by = "collision_index")
  message(sprintf("  %d rows x %d cols.", nrow(combined), ncol(combined)))

  # Any collision with no matching vehicle row -> vehicle flags all NA
  unmatched <- sum(is.na(combined$has_car))
  if (unmatched > 0) {
    message(sprintf("  WARNING: %d collisions had no matching vehicle rows (flags set to NA).", unmatched))
  }

  # ── Save ─────────────────────────────────────────────────────────────────
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(OUTPUT_DIR, OUTPUT_FILE)

  if (file.exists(out_path)) {
    stop("Output already exists: ", out_path,
         "\nDelete it or change OUTPUT_FILE in the CONFIG block.")
  }

  message(sprintf("\nSaving to %s ...", out_path))
  write_csv(combined, out_path)
  message("Done.")
}

main()

# ── CONFIG ────────────────────────────────────────────────────────────────────
# PIPELINE_DATA_DIR may be set by run_pipeline.R; otherwise falls back to a fixed run.
if (!exists("PIPELINE_DATA_DIR")) {
  PIPELINE_DATA_DIR <- "data/raw/stats19/stats19_20260606_181720"
}
INPUT_VEHICLE_CSV <- file.path(
  PIPELINE_DATA_DIR, "codes",
  paste0(basename(PIPELINE_DATA_DIR), "_vehicles.csv")
)

OUTPUT_DIR  <- "data/features/stats19"
OUTPUT_FILE <- "stats19_vehicle_realtime_features.csv"
# ──────────────────────────────────────────────────────────────────────────────

# ── Realtime columns to keep (from docs/realtime features) ───────────────────
KEEP_COLS <- c(
  "collision_index",   # infrastructure — join key
  "vehicle_type",
  "towing_and_articulation"
)

# ── vehicle_type groupings ────────────────────────────────────────────────────
# The ultra-rare codes (tram, ridden_horse, unknown_vehicle, minibus,
# mobility_scooter, agricultural, other_vehicle, escooter -- all <1.5%
# prevalence) stay dropped: they showed a memorization signature (e.g.
# has_tram at 0.02% prevalence still appeared in 123 tree splits) rather than
# generalizable signal. has_hgv, has_bus_coach, and has_towing are kept
# despite low prevalence because each had a healthy split-to-prevalence ratio
# in the original 15-flag model (real signal, not memorization) -- has_hgv in
# particular had the highest gain of any of the 15, and dropping it cost real
# accuracy (test macro F1 0.697 -> 0.692 on the normalized binary variant).
VEHICLE_TYPE_GROUPS <- list(
  has_motorcycle  = c(2, 3, 4, 5, 23, 97),
  has_pedal_cycle = c(1),
  has_van         = c(19),
  has_car         = c(8, 9),
  has_hgv         = c(20, 21, 98),
  has_bus_coach   = c(11)
)

# ── Package setup ─────────────────────────────────────────────────────────────
for (pkg in c("readr", "dplyr")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  library(pkg, character.only = TRUE)
}

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
  if (!file.exists(INPUT_VEHICLE_CSV)) {
    stop("Input file not found: ", INPUT_VEHICLE_CSV,
         "\nUpdate INPUT_VEHICLE_CSV in the CONFIG block.")
  }

  message("Reading: ", INPUT_VEHICLE_CSV)
  veh_df <- read_csv(INPUT_VEHICLE_CSV, show_col_types = FALSE)
  message(sprintf("  Loaded %d rows x %d cols.", nrow(veh_df), ncol(veh_df)))

  missing <- setdiff(KEEP_COLS, names(veh_df))
  if (length(missing) > 0) stop("Missing columns: ", paste(missing, collapse = ", "))

  # ── Step 1: keep only realtime columns ───────────────────────────────────
  message("\nStep 1: Filtering to realtime columns...")
  veh_df <- veh_df[, KEEP_COLS]
  message(sprintf("  %d rows x %d cols.", nrow(veh_df), ncol(veh_df)))

  # ── Step 2: aggregate to one row per collision_index ──────────────────────
  message("\nStep 2: Aggregating to one row per collision_index...")

  # Build one multi-hot flag expression per entry in VEHICLE_TYPE_GROUPS
  vehicle_type_flags <- lapply(VEHICLE_TYPE_GROUPS, function(codes) {
    quo(as.integer(any(vehicle_type %in% !!codes, na.rm = TRUE)))
  })

  agg <- veh_df |>
    group_by(collision_index) |>
    summarise(
      !!!vehicle_type_flags,

      # ── towing: 1 if any vehicle had any tow/articulation (codes 1-5) ──
      has_towing = as.integer(any(towing_and_articulation %in% 1:5, na.rm = TRUE)),

      .groups = "drop"
    )

  message(sprintf("  Aggregated to %d rows x %d cols.", nrow(agg), ncol(agg)))

  # ── Summary ───────────────────────────────────────────────────────────────
  flag_cols <- setdiff(names(agg), "collision_index")
  message("\nFlag prevalence (% of collisions with at least one):")
  for (col in flag_cols) {
    pct <- round(mean(agg[[col]], na.rm = TRUE) * 100, 1)
    message(sprintf("  %-22s %5.1f%%", col, pct))
  }

  # ── Save ─────────────────────────────────────────────────────────────────
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  out_path <- file.path(OUTPUT_DIR, OUTPUT_FILE)

  if (file.exists(out_path)) {
    stop("Output already exists: ", out_path,
         "\nDelete it or change OUTPUT_FILE in the CONFIG block.")
  }

  message(sprintf("\nSaving to %s ...", out_path))
  write_csv(agg, out_path)
  message("Done.")
}

main()

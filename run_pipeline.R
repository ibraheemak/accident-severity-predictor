# ── CONFIG ────────────────────────────────────────────────────────────────────
# Which step to start the pipeline from:
#   1 = download & sample STATS19 data              (stats19_download.R)
#   2 = build collision realtime features            (build_realtime_collision_features.R)
#   3 = build vehicle realtime features              (build_realtime_vehicle_features.R)
#   4 = build combined feature table                 (build_combined_features.R)
#   5 = join AADF traffic volume onto combined table (join_aadf.R)
#   6 = join OpenStreetMap road infrastructure features (add_osm_features.R)
#
# All later steps are run too. If START_FROM > 1, the pipeline uses the most
# recent data/raw/stats19_* download as input.
#
# Note: step 6 (add_osm_features.R) downloads a ~2 GB Great Britain OSM PBF
# extract on first run and can take several minutes even with a warm cache.
START_FROM <- 2
# ──────────────────────────────────────────────────────────────────────────────

steps <- list(
  "1" = "stats19_download.R",
  "2" = "build_realtime_collision_features.R",
  "3" = "build_realtime_vehicle_features.R",
  "4" = "build_combined_features.R",
  "5" = "join_aadf.R",
  "6" = "add_osm_features.R"
)

if (!as.character(START_FROM) %in% names(steps)) {
  stop("START_FROM must be one of: ", paste(names(steps), collapse = ", "))
}

# Resolve PIPELINE_DATA_DIR (used by steps 2 and 3) before running anything.
if (START_FROM == 1) {
  message("=== Step 1: ", steps[["1"]], " ===")
  source(steps[["1"]])
  # stats19_download.R defines OUTPUT_DIR for the run it just produced.
  PIPELINE_DATA_DIR <- OUTPUT_DIR
} else {
  raw_dirs <- list.dirs("data/raw", recursive = FALSE)
  raw_dirs <- raw_dirs[grepl("^data/raw/stats19_", raw_dirs)]
  if (length(raw_dirs) == 0) {
    stop("No data/raw/stats19_* directories found. Run from step 1 first.")
  }
  PIPELINE_DATA_DIR <- sort(raw_dirs, decreasing = TRUE)[1]
  message("Using existing download: ", PIPELINE_DATA_DIR)
}

for (step in setdiff(names(steps), "1")) {
  if (as.integer(step) < as.integer(START_FROM)) next
  message("\n=== Step ", step, ": ", steps[[step]], " ===")
  source(steps[[step]])
}

message("\nPipeline complete.")

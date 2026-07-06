# ── CONFIG ────────────────────────────────────────────────────────────────────
RUN_TIMESTAMP <- format(Sys.time(), "%Y%m%d_%H%M%S")
YEARS         <- 2016:2025
OUTPUT_DIR    <- paste0("data/raw/stats19_", RUN_TIMESTAMP)
RANDOM_SEED   <- 42
# Sampling caps per year (keep all rows if actual count is below the cap):
#   - Fatal   : keep ALL rows (no cap)
#   - Serious : up to SERIOUS_CAP rows
#   - Slight  : up to SLIGHT_CAP rows
SERIOUS_CAP <- 2800L
SLIGHT_CAP  <- 5600L
# ──────────────────────────────────────────────────────────────────────────────

# ── Package setup ─────────────────────────────────────────────────────────────
required_packages <- c("stats19", "dplyr", "readr", "lubridate")

invisible(lapply(required_packages, function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing package: ", pkg)
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  library(pkg, character.only = TRUE)
}))
# ──────────────────────────────────────────────────────────────────────────────

# DfT renamed "accident" -> "collision" throughout the dataset (post-2016 package)
COLLISION_TYPE <- "collision"
CASUALTY_TYPE  <- "casualty"
VEHICLE_TYPE   <- "vehicle"
SEVERITY_COL   <- "collision_severity"
INDEX_COL      <- "collision_index"



# ── Helpers ───────────────────────────────────────────────────────────────────

#' Fixed-cap severity sample per year:
#'   Fatal   -> all rows
#'   Serious -> up to serious_cap rows (all if fewer available)
#'   Slight  -> up to slight_cap rows  (all if fewer available)
severity_sample <- function(df, severity_col, serious_cap, slight_cap, seed) {
  set.seed(seed)

  cap_stratum <- function(stratum, cap, label) {
    if (nrow(stratum) <= cap) {
      message(sprintf("  NOTE: %s has %d rows (<= cap %d); keeping all.",
                      label, nrow(stratum), cap))
      stratum
    } else {
      slice_sample(stratum, n = cap)
    }
  }

  fatal   <- filter(df, .data[[severity_col]] == "Fatal")
  serious <- cap_stratum(filter(df, .data[[severity_col]] == "Serious"), serious_cap, "Serious")
  slight  <- cap_stratum(filter(df, .data[[severity_col]] == "Slight"),  slight_cap,  "Slight")

  message(sprintf(
    "  Severity counts — Fatal: %d (all)  Serious: %d  Slight: %d",
    nrow(fatal), nrow(serious), nrow(slight)
  ))

  bind_rows(fatal, serious, slight)
}

#' Merge *_historic columns into their primary column then drop them.
#' Replicates the step format_collisions() does automatically, so the codes
#' version is consistent with the labels version.
#'
#' Pedestrian crossing special case:
#'   DfT already pre-populates `pedestrian_crossing` (new unified column) in
#'   all raw CSVs, so both _historic columns are simply dropped. The 684 NAs
#'   that remain are genuinely unrecoverable (historic columns also blank).
#'
#' All other *_historic columns:
#'   Values are backfilled into the primary column where NA, then dropped.
merge_historic_cols <- function(df) {
  # Drop pedestrian crossing _historic cols — pedestrian_crossing already correct
  df$pedestrian_crossing_human_control_historic      <- NULL
  df$pedestrian_crossing_physical_facilities_historic <- NULL

  # Generic backfill-and-drop for all remaining *_historic columns
  historic_cols <- names(df)[grepl("_historic$", names(df))]
  for (hcol in historic_cols) {
    primary_col <- gsub("_historic$", "", hcol)
    if (primary_col %in% names(df)) {
      df[[primary_col]] <- ifelse(is.na(df[[primary_col]]), df[[hcol]], df[[primary_col]])
      df[[hcol]] <- NULL
    }
  }
  df
}

#' Safely call get_stats19(); returns NULL and logs on any error.
#' format=TRUE  -> human-readable string labels (for EDA / reports)
#' format=FALSE -> raw numeric codes            (for ML pipeline)
safe_get <- function(year, type, format = TRUE) {
  tryCatch(
    {
      df <- get_stats19(year = year, type = type, ask = FALSE,
                        silent = TRUE, format = format)
      if (is.null(df) || nrow(df) == 0) {
        message(sprintf("  WARNING: %s %d returned empty — skipping.", type, year))
        return(NULL)
      }
      df
    },
    error = function(e) {
      message(sprintf("  WARNING: failed to download %s %d — %s",
                      type, year, conditionMessage(e)))
      NULL
    }
  )
}

#' Download, normalise names, and add data_year for one table.
#' Returns NULL (with a message) on failure.
fetch_table <- function(year, type, format) {
  df <- safe_get(year, type, format = format)
  if (is.null(df)) return(NULL)
  names(df) <- tolower(trimws(names(df)))
  mutate(df, data_year = year)
}

#' Process one year: download labeled + coded versions, sample once, filter both.
#' Returns list(labels = list(col, cas, veh), codes = list(col, cas, veh), info).
#' All tables are NULL if the year was skipped.
process_year <- function(year) {
  null_result <- function(reason) {
    info$skipped     <<- TRUE
    info$skip_reason <<- reason
    list(labels = list(col = NULL, cas = NULL, veh = NULL),
         codes  = list(col = NULL, cas = NULL, veh = NULL),
         info   = info)
  }

  info <- list(year = year, skipped = FALSE, skip_reason = NA_character_,
               raw_rows = 0L,
               fatal_rows = 0L,
               serious_sampled = 0L, serious_total = 0L,
               slight_sampled  = 0L, slight_total  = 0L,
               casualty_rows = 0L, vehicle_rows = 0L)

  # ── Collisions: labeled (used for sampling) ──
  message(sprintf("\n[%d] Downloading collisions (labels)...", year))
  col_lbl <- fetch_table(year, COLLISION_TYPE, format = TRUE)
  if (is.null(col_lbl)) return(null_result("collision labels empty or download failed"))

  if (!SEVERITY_COL %in% names(col_lbl)) {
    message(sprintf("  WARNING: column '%s' not found — skipping %d.", SEVERITY_COL, year))
    return(null_result(sprintf("column '%s' not found in collision table", SEVERITY_COL)))
  }

  info$raw_rows <- nrow(col_lbl)
  message(sprintf("  %d rows downloaded.", nrow(col_lbl)))

  # ── Sample on labeled version (severity_sample uses string labels) ──
  message("  Applying severity-aware sampling...")
  col_lbl_sampled <- severity_sample(col_lbl, SEVERITY_COL, SERIOUS_CAP, SLIGHT_CAP, RANDOM_SEED)

  info$fatal_rows      <- sum(col_lbl_sampled[[SEVERITY_COL]] == "Fatal")
  info$serious_sampled <- sum(col_lbl_sampled[[SEVERITY_COL]] == "Serious")
  info$serious_total   <- sum(col_lbl[[SEVERITY_COL]] == "Serious")
  info$slight_sampled  <- sum(col_lbl_sampled[[SEVERITY_COL]] == "Slight")
  info$slight_total    <- sum(col_lbl[[SEVERITY_COL]] == "Slight")
  message(sprintf("  Total sampled: %d rows.", nrow(col_lbl_sampled)))

  sampled_ids <- col_lbl_sampled[[INDEX_COL]]

  # ── Collisions: numeric codes — filter, merge historic cols, match row order ──
  message("  Downloading collisions (codes)...")
  col_cod <- fetch_table(year, COLLISION_TYPE, format = FALSE)
  if (is.null(col_cod)) return(null_result("collision codes empty or download failed"))
  col_cod <- merge_historic_cols(col_cod)
  col_cod_sampled <- col_lbl_sampled[, INDEX_COL, drop = FALSE] |>
    left_join(col_cod, by = INDEX_COL)

  # ── Casualties — filter both versions, merge historic cols, sort by index ──
  message("  Downloading casualties (labels)...")
  cas_lbl <- fetch_table(year, CASUALTY_TYPE, format = TRUE)
  if (is.null(cas_lbl)) return(null_result("casualty labels empty or download failed"))
  cas_lbl <- cas_lbl |> filter(.data[[INDEX_COL]] %in% sampled_ids) |>
    arrange(.data[[INDEX_COL]])

  message("  Downloading casualties (codes)...")
  cas_cod <- fetch_table(year, CASUALTY_TYPE, format = FALSE)
  if (is.null(cas_cod)) return(null_result("casualty codes empty or download failed"))
  cas_cod <- merge_historic_cols(cas_cod) |>
    filter(.data[[INDEX_COL]] %in% sampled_ids) |>
    arrange(.data[[INDEX_COL]])

  info$casualty_rows <- nrow(cas_lbl)
  message(sprintf("  %d casualty rows matched.", nrow(cas_lbl)))

  # ── Vehicles — filter both versions, merge historic cols, sort by index ──
  message("  Downloading vehicles (labels)...")
  veh_lbl <- fetch_table(year, VEHICLE_TYPE, format = TRUE)
  if (is.null(veh_lbl)) return(null_result("vehicle labels empty or download failed"))
  veh_lbl <- veh_lbl |> filter(.data[[INDEX_COL]] %in% sampled_ids) |>
    arrange(.data[[INDEX_COL]])

  message("  Downloading vehicles (codes)...")
  veh_cod <- fetch_table(year, VEHICLE_TYPE, format = FALSE)
  if (is.null(veh_cod)) return(null_result("vehicle codes empty or download failed"))
  veh_cod <- merge_historic_cols(veh_cod) |>
    filter(.data[[INDEX_COL]] %in% sampled_ids) |>
    arrange(.data[[INDEX_COL]])

  info$vehicle_rows <- nrow(veh_lbl)
  message(sprintf("  %d vehicle rows matched.", nrow(veh_lbl)))

  list(
    labels = list(col = col_lbl_sampled, cas = cas_lbl, veh = veh_lbl),
    codes  = list(col = col_cod_sampled, cas = cas_cod, veh = veh_cod),
    info   = info
  )
}

# ── Build metadata string ─────────────────────────────────────────────────────
build_metadata <- function(run_ts, year_infos, master_col, paths) {
  sev_counts <- master_col |>
    count(.data[[SEVERITY_COL]]) |>
    mutate(pct = round(n / sum(n) * 100, 1))

  per_year_lines <- sapply(year_infos, function(info) {
    if (info$skipped) {
      sprintf("  %d: SKIPPED — %s", info$year, info$skip_reason)
    } else {
      sprintf(
        "  %d: Fatal=%d (all)  Serious=%d (of %d)  Slight=%d (of %d)  casualties=%d  vehicles=%d",
        info$year,
        info$fatal_rows,
        info$serious_sampled, info$serious_total,
        info$slight_sampled,  info$slight_total,
        info$casualty_rows,   info$vehicle_rows
      )
    }
  })

  skipped_years <- Filter(function(i) i$skipped, year_infos)
  skipped_lines <- if (length(skipped_years) == 0) {
    "  None"
  } else {
    sapply(skipped_years, function(i) sprintf("  %d: %s", i$year, i$skip_reason))
  }

  sev_lines <- apply(sev_counts, 1, function(row) {
    sprintf("  %s: %s  (%.1f%%)", row[SEVERITY_COL], row["n"], as.numeric(row["pct"]))
  })

  paste(c(
    "STATS19 Download & Sample — Metadata",
    strrep("=", 50),
    sprintf("Run timestamp : %s", run_ts),
    sprintf("YEARS config  : %s", paste(YEARS, collapse = ", ")),
    sprintf("SERIOUS_CAP   : %d per year", SERIOUS_CAP),
    sprintf("SLIGHT_CAP    : %d per year", SLIGHT_CAP),
    sprintf("RANDOM_SEED   : %d", RANDOM_SEED),
    "",
    "Output files",
    strrep("-", 30),
    sprintf("  labels/collisions : %s", paths$lbl_col),
    sprintf("  labels/casualties : %s", paths$lbl_cas),
    sprintf("  labels/vehicles   : %s", paths$lbl_veh),
    sprintf("  codes/collisions  : %s", paths$cod_col),
    sprintf("  codes/casualties  : %s", paths$cod_cas),
    sprintf("  codes/vehicles    : %s", paths$cod_veh),
    "",
    "Per-year summary",
    strrep("-", 30),
    per_year_lines,
    "",
    "Collisions — final dataset",
    strrep("-", 30),
    sprintf("  Total rows : %d", nrow(master_col)),
    sprintf("  Columns    : %d", ncol(master_col)),
    "",
    sprintf("Severity distribution (%s)", SEVERITY_COL),
    strrep("-", 30),
    sev_lines,
    "",
    "Skipped years",
    strrep("-", 30),
    skipped_lines,
    ""
  ), collapse = "\n")
}

# ── Main ──────────────────────────────────────────────────────────────────────
main <- function() {
  stem     <- sprintf("stats19_%s", RUN_TIMESTAMP)
  dir_lbl  <- file.path(OUTPUT_DIR, "labels")
  dir_cod  <- file.path(OUTPUT_DIR, "codes")

  paths <- list(
    lbl_col = file.path(dir_lbl, sprintf("%s_collisions.csv", stem)),
    lbl_cas = file.path(dir_lbl, sprintf("%s_casualties.csv", stem)),
    lbl_veh = file.path(dir_lbl, sprintf("%s_vehicles.csv",   stem)),
    cod_col = file.path(dir_cod, sprintf("%s_collisions.csv", stem)),
    cod_cas = file.path(dir_cod, sprintf("%s_casualties.csv", stem)),
    cod_veh = file.path(dir_cod, sprintf("%s_vehicles.csv",   stem)),
    meta    = file.path(OUTPUT_DIR, sprintf("%s_meta.txt",    stem))
  )

  dir.create(dir_lbl, recursive = TRUE, showWarnings = FALSE)
  dir.create(dir_cod, recursive = TRUE, showWarnings = FALSE)
  for (path in unlist(paths)) {
    if (file.exists(path)) stop("Output file already exists: ", path)
  }

  lbl_cols <- list(); lbl_cass <- list(); lbl_vehs <- list()
  cod_cols <- list(); cod_cass <- list(); cod_vehs <- list()
  year_infos <- list()

  for (year in YEARS) {
    result <- process_year(year)
    year_infos[[length(year_infos) + 1]] <- result$info

    if (!is.null(result$labels$col)) {
      lbl_cols[[length(lbl_cols) + 1]] <- result$labels$col
      lbl_cass[[length(lbl_cass) + 1]] <- result$labels$cas
      lbl_vehs[[length(lbl_vehs) + 1]] <- result$labels$veh
      cod_cols[[length(cod_cols) + 1]] <- result$codes$col
      cod_cass[[length(cod_cass) + 1]] <- result$codes$cas
      cod_vehs[[length(cod_vehs) + 1]] <- result$codes$veh
    }
  }

  if (length(lbl_cols) == 0) {
    message("\nNo data collected — all years were skipped. Exiting.")
    return(invisible(NULL))
  }

  message("\nConcatenating all years...")
  master_lbl_col <- bind_rows(lbl_cols)
  master_lbl_cas <- bind_rows(lbl_cass)
  master_lbl_veh <- bind_rows(lbl_vehs)
  master_cod_col <- bind_rows(cod_cols)
  master_cod_cas <- bind_rows(cod_cass)
  master_cod_veh <- bind_rows(cod_vehs)

  message("  Labels — collisions: ", nrow(master_lbl_col), " rows  casualties: ",
          nrow(master_lbl_cas), "  vehicles: ", nrow(master_lbl_veh))
  message("  Codes  — collisions: ", nrow(master_cod_col), " rows  casualties: ",
          nrow(master_cod_cas), "  vehicles: ", nrow(master_cod_veh))

  saves <- list(
    list(path = paths$lbl_col, df = master_lbl_col, label = "collisions (labels)"),
    list(path = paths$lbl_cas, df = master_lbl_cas, label = "casualties (labels)"),
    list(path = paths$lbl_veh, df = master_lbl_veh, label = "vehicles   (labels)"),
    list(path = paths$cod_col, df = master_cod_col, label = "collisions (codes) "),
    list(path = paths$cod_cas, df = master_cod_cas, label = "casualties (codes) "),
    list(path = paths$cod_veh, df = master_cod_veh, label = "vehicles   (codes) ")
  )

  message("")
  for (s in saves) {
    message(sprintf("Saving %s -> %s ...", s$label, s$path))
    write_csv(s$df, s$path)
  }

  message(sprintf("\nWriting metadata to %s ...", paths$meta))
  meta_text <- build_metadata(RUN_TIMESTAMP, year_infos, master_lbl_col, paths)
  writeLines(meta_text, paths$meta)

  message("\nDone.")
  for (s in saves) message(sprintf("  %s : %s", s$label, s$path))
  message("  Meta              : ", paths$meta)
  message("\n", meta_text)
}

main()

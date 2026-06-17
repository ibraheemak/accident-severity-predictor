library(dplyr)
library(sf)
library(readr)

# ── Debug flags ────────────────────────────────────────────────────────────────
DEBUG        <- FALSE
DEBUG_YEAR   <- 2020
DEBUG_N_ACC  <- 50

# ── Debug log (terminal + file) ───────────────────────────────────────────────
here <- function(...) file.path(
  "/Users/ibraheem/Documents/GitHub/accident-severity-predictor", ...)

log_path <- here("data/features/stats19+aadf/join_aadf.log")

log <- function(..., level = "INFO") {
  msg <- paste0("[", level, "] ", paste0(...))
  message(msg)
  cat(msg, "\n", file = log_path, append = TRUE)   # always write to log
}
dbg <- function(...) if (DEBUG) log(..., level = "DEBUG")

# Clear/create log file at start of each run
cat(paste0("=== join_aadf log — ", Sys.time(),
           if (DEBUG) paste0("  [DEBUG: year=", DEBUG_YEAR, " n=", DEBUG_N_ACC, "]"),
           " ===\n\n"),
    file = log_path)

# ── Paths ──────────────────────────────────────────────────────────────────────
accidents_path <- here("data/features/stats19/stats19_combined_realtime_features.csv")
aadf_path      <- here("data/raw/aadf/dft_traffic_counts_aadf.csv")
out_path       <- here("data/features/stats19+aadf/stats19+aadf_combined_realtime_features.csv")

if (DEBUG) log("DEBUG MODE: year=", DEBUG_YEAR, ", n_acc=", DEBUG_N_ACC)

# ── Road-class code → letter prefix ───────────────────────────────────────────
# Numeric codes as found in Stats19: 1=Motorway, 2=A(M), 3=A, 4=B, 5=C, 6=Unclassified
road_class_prefix <- function(code) {
  dplyr::case_when(
    code == 1 ~ "M",
    code == 2 ~ "A",
    code == 3 ~ "A",
    code == 4 ~ "B",
    code == 5 ~ "C",
    code == 6 ~ "U",
    TRUE      ~ NA_character_
  )
}

# ── Load & prepare accidents ───────────────────────────────────────────────────
log("[1/6] Loading accidents from: ", accidents_path)
acc_raw <- read_csv(accidents_path, show_col_types = FALSE)
log("      Loaded ", nrow(acc_raw), " rows, ", ncol(acc_raw), " columns")

if (DEBUG) {
  acc_raw <- acc_raw |> filter(collision_year == DEBUG_YEAR) |> slice_head(n = DEBUG_N_ACC)
  log("      Kept ", nrow(acc_raw), " accidents (year=", DEBUG_YEAR, ")")
}

log("[2/6] Building road IDs for accidents …")
acc <- acc_raw |>
  mutate(
    .row = row_number(),
    first_road_class_n  = suppressWarnings(as.integer(first_road_class)),
    second_road_class_n = suppressWarnings(as.integer(second_road_class)),

    accident_road_id = {
      pfx <- road_class_prefix(first_road_class_n)
      num <- as.integer(first_road_number)
      ifelse(!is.na(pfx) & !is.na(num) & num != 0,
             paste0(pfx, num), NA_character_)
    },

    accident_road_id_2 = {
      pfx <- road_class_prefix(second_road_class_n)
      num <- as.integer(second_road_number)
      ifelse(!is.na(pfx) & !is.na(num) & num != 0,
             paste0(pfx, num), NA_character_)
    }
  )

dbg("Sample accident_road_id   : ", paste(head(acc$accident_road_id,   10), collapse = ", "))
dbg("Sample accident_road_id_2 : ", paste(head(acc$accident_road_id_2, 10), collapse = ", "))
dbg("NAs in accident_road_id   : ", sum(is.na(acc$accident_road_id)), " / ", nrow(acc))

log("      Converting accidents to sf (EPSG:4326) …")
acc_sf <- acc |>
  filter(!is.na(longitude), !is.na(latitude)) |>
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)
log("      ", nrow(acc_sf), " accidents with valid coordinates")

# ── Load & prepare AADF count points ──────────────────────────────────────────
log("[3/6] Loading AADF from: ", aadf_path)
aadf_raw <- read_csv(aadf_path, show_col_types = FALSE)
log("      Loaded ", nrow(aadf_raw), " rows")

aadf <- aadf_raw |>
  select(count_point_id, year, latitude, longitude,
         road_name, all_motor_vehicles) |>
  filter(!is.na(latitude), !is.na(longitude))

log("      ", nrow(aadf), " count points with valid coordinates")
dbg("AADF years available: ", paste(sort(unique(aadf$year)), collapse = ", "))

aadf_sf <- aadf |>
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)

# ── Matching function for one accident year ────────────────────────────────────
match_year <- function(yr) {
  log("[4/6] Matching year ", yr, " …")

  acc_yr <- acc_sf |> filter(collision_year == yr)
  log("      Accidents in year: ", nrow(acc_yr))
  if (nrow(acc_yr) == 0) { log("      No accidents — skipping"); return(tibble()) }

  cp_yr <- aadf_sf |> filter(year == yr)
  log("      Count points in year: ", nrow(cp_yr))
  if (nrow(cp_yr) == 0) {
    log("      No count points — all unmatched")
    return(acc_yr |>
             st_drop_geometry() |>
             select(.row) |>
             mutate(aadf_traffic_volume = NA_real_,
                    aadf_distance_m     = NA_real_,
                    aadf_match_type     = "unmatched"))
  }

  n_acc  <- nrow(acc_yr)
  result <- vector("list", n_acc)
  counts <- c(same_road_1 = 0L, same_road_2 = 0L, nearest_fallback = 0L)

  for (i in seq_len(n_acc)) {
    if (i %% 500 == 0 || i == n_acc)
      log("      processed ", i, " / ", n_acc, " accidents …")

    a     <- acc_yr[i, ]
    road1 <- a$accident_road_id
    road2 <- a$accident_road_id_2

    dbg(sprintf("acc %d | road1=%s road2=%s", i,
                ifelse(is.na(road1), "NA", road1),
                ifelse(is.na(road2), "NA", road2)))

    # ── Step 1: exact match on road1, nearest count point ────────────────────
    if (!is.na(road1)) {
      cand1 <- cp_yr |> filter(road_name == road1)
      if (nrow(cand1) > 0) {
        idx  <- st_nearest_feature(a, cand1)
        dist <- as.numeric(st_distance(a, cand1[idx, ]))
        dbg(sprintf("  step1 road1=%s matched %d pts, nearest=%.1f m", road1, nrow(cand1), dist))
        result[[i]] <- tibble(
          .row                = a$.row,
          aadf_traffic_volume = cand1$all_motor_vehicles[idx],
          aadf_distance_m     = dist,
          aadf_match_type     = "same_road"
        )
        counts["same_road_1"] <- counts["same_road_1"] + 1L
        next
      }
      dbg(sprintf("  step1 road1=%s — no count points in AADF", road1))
    }

    # ── Step 2: exact match on road2, nearest count point ────────────────────
    if (!is.na(road2)) {
      cand2 <- cp_yr |> filter(road_name == road2)
      if (nrow(cand2) > 0) {
        idx  <- st_nearest_feature(a, cand2)
        dist <- as.numeric(st_distance(a, cand2[idx, ]))
        dbg(sprintf("  step2 road2=%s matched %d pts, nearest=%.1f m", road2, nrow(cand2), dist))
        result[[i]] <- tibble(
          .row                = a$.row,
          aadf_traffic_volume = cand2$all_motor_vehicles[idx],
          aadf_distance_m     = dist,
          aadf_match_type     = "same_road"
        )
        counts["same_road_2"] <- counts["same_road_2"] + 1L
        next
      }
      dbg(sprintf("  step2 road2=%s — no count points in AADF", road2))
    }

    # ── Step 3: no exact road match — nearest count point overall (no threshold)
    dbg("  step3 no exact road match — nearest overall")
    idx  <- st_nearest_feature(a, cp_yr)
    dist <- as.numeric(st_distance(a, cp_yr[idx, ]))
    dbg(sprintf("  step3 nearest=%.1f m (road=%s)", dist, cp_yr$road_name[idx]))
    result[[i]] <- tibble(
      .row                = a$.row,
      aadf_traffic_volume = cp_yr$all_motor_vehicles[idx],
      aadf_distance_m     = dist,
      aadf_match_type     = "nearest_fallback"
    )
    counts["nearest_fallback"] <- counts["nearest_fallback"] + 1L
  }

  log(sprintf("      Done year %d — same_road(road1): %d | same_road(road2): %d | nearest_fallback: %d",
              yr, counts["same_road_1"], counts["same_road_2"], counts["nearest_fallback"]))

  bind_rows(result)
}

# ── Run per year and combine ───────────────────────────────────────────────────
years <- sort(unique(acc$collision_year))
log("\n[4/6] Years to process: ", paste(years, collapse = ", "))

matches <- bind_rows(lapply(years, match_year))
log("\n[5/6] Matching complete — ", nrow(matches), " rows returned")

# ── Join back onto original accidents ─────────────────────────────────────────
log("[6/6] Joining results back onto accidents …")
result <- acc_raw |>
  mutate(.row = row_number()) |>
  left_join(matches, by = ".row") |>
  mutate(
    aadf_match_type     = if_else(is.na(aadf_match_type), "nearest_fallback", aadf_match_type),
    aadf_traffic_volume = if_else(is.na(aadf_traffic_volume), NA_real_, aadf_traffic_volume)
  ) |>
  select(-.row)

# ── Write output ───────────────────────────────────────────────────────────────
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
log("Writing output to: ", out_path)
write_csv(result, out_path)
log("Done.")
if (DEBUG) log("(written from DEBUG mode — only ", nrow(result), " rows)")

# ── Summary ───────────────────────────────────────────────────────────────────
dist_summary <- function(distances, label) {
  breaks <- c(0, 100, 200, 300, 500, 1000, 2000, 5000, Inf)
  labels <- c("<100m", "100-200m", "200-300m", "300-500m",
              "500-1000m", "1-2km", "2-5km", ">5km")
  bins <- cut(distances[!is.na(distances)], breaks = breaks, labels = labels,
              right = FALSE, include.lowest = TRUE)
  tbl  <- table(bins)
  lines <- c(
    paste0("\nDistance distribution — ", label, ":"),
    sprintf("  %-12s  %6s  %6s", "bucket", "n", "%"),
    sprintf("  %-12s  %6s  %6s", "------", "-", "-"),
    sprintf("  %-12s  %6d  %5.1f%%",
            names(tbl), as.integer(tbl),
            100 * as.integer(tbl) / sum(tbl))
  )
  cat(paste(lines, collapse = "\n"), "\n")
  cat(paste(lines, collapse = "\n"), "\n", file = log_path, append = TRUE)
}

cat("\nMatch type distribution:\n")
print(table(result$aadf_match_type))

dist_summary(result$aadf_distance_m, "all matches")
dist_summary(result$aadf_distance_m[result$aadf_match_type == "same_road"],
             "same_road only")
dist_summary(result$aadf_distance_m[result$aadf_match_type == "nearest_fallback"],
             "nearest_fallback only")

cat(sprintf("\nTotal accidents: %d\nOutput columns: %d\n",
            nrow(result), ncol(result)))

if (DEBUG) {
  cat("\nSample output (first 5 rows of new columns):\n")
  print(result |> select(collision_index,
                          aadf_traffic_volume, aadf_distance_m, aadf_match_type) |>
          head(5))
  cat("\nFull debug log written to:", log_path, "\n")
}

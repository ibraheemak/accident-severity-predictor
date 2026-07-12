#!/usr/bin/env Rscript
# add_osm_features.R
#
# Adds OpenStreetMap road infrastructure features to the STATS19+AADF combined
# real-time feature table.
#
# LIMITATION: OSM data is a current-era snapshot (downloaded at run time).
# Accident years span 2020-2024, but road infrastructure tags (lanes, etc.)
# reflect the network as it exists today.  Treat as a proxy, not
# a historically accurate record.
#
# Usage:
#   Rscript add_osm_features.R            # full run
#   Rscript add_osm_features.R --debug    # 50-row smoke test + verbose logging

suppressPackageStartupMessages({
  library(osmdata)
  library(sf)
  library(dplyr)
  library(readr)
})

# osmextract is used for road_ways (national-scale; Overpass times out).
# Install automatically if missing.
if (!requireNamespace("osmextract", quietly = TRUE)) {
  message("Installing osmextract …")
  install.packages("osmextract", repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages(library(osmextract))

# ── Debug flag ─────────────────────────────────────────────────────────────────
args  <- commandArgs(trailingOnly = TRUE)
DEBUG <- "--debug" %in% args
DEBUG_N          <- 50L    # accidents to process in debug mode
DEBUG_BBOX_PAD_M <- 15000  # bbox padded this many metres around the cluster centroid

# Geofabrik PBF URL for the osmextract full-run path.
# Geofabrik does not publish a separate England extract; only great-britain-latest
# is available.  Debug mode bypasses osmextract entirely and uses Overpass instead
# (the debug bbox is small enough that Overpass handles it without timeouts).
GEOFABRIK_URL <- "https://download.geofabrik.de/europe/great-britain-latest.osm.pbf"

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE <- "data/features/stats19+aadf/stats19+aadf_combined_realtime_features.csv"

if (DEBUG) {
  OUTPUT_DIR  <- "data/features/stats19+aadf+osm"
  OUTPUT_FILE <- file.path(OUTPUT_DIR, "stats19+aadf+osm_combined_realtime_features_debug.csv")
  CACHE_DIR   <- "data/raw/osm/cache/debug"
  LOG_FILE    <- file.path(OUTPUT_DIR, "add_osm_features_debug.log")
} else {
  OUTPUT_DIR  <- "data/features/stats19+aadf+osm"
  OUTPUT_FILE <- file.path(OUTPUT_DIR, "stats19+aadf+osm_combined_realtime_features.csv")
  CACHE_DIR   <- "data/raw/osm/cache"
  LOG_FILE    <- file.path(OUTPUT_DIR, "add_osm_features.log")
}

SPEED_CAM_RADIUS_M      <- 500   # search radius for speed camera nodes
TRAFFIC_CALM_RADIUS_M   <-  50   # search radius for traffic calming nodes
TRAFFIC_SIGNAL_RADIUS_M <-  50   # search radius for traffic signal nodes (junction-local, like calming)
WAY_RADIUS_M            <-  50   # search radius for road way matching
MEDIAN_BARRIER_RADIUS   <-  15   # barrier=* way must be within this of accident
POINT_WAY_RADIUS_M      <- 100   # max dist to assign a way's ref to a point node

BNG_CRS <- 27700  # British National Grid — metres, used for all distances

# STATS19 first_road_class codes → OSM ref prefix
ROAD_CLASS_PREFIX <- c(
  `1` = "M",   # Motorway
  `2` = "A",   # A(M) road (motorway standard)
  `3` = "A",   # A road
  `4` = "B",   # B road
  `5` = "C",   # C road
  `6` = ""     # Unclassified — no meaningful ref
)

# ── 0. Setup: directories + logging ───────────────────────────────────────────
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(CACHE_DIR,  showWarnings = FALSE, recursive = TRUE)

log_con <- file(LOG_FILE, open = "wt")

log_msg <- function(...) {
  ts  <- format(Sys.time(), "[%Y-%m-%d %H:%M:%S]")
  txt <- paste0(ts, " ", paste(..., sep = ""))
  message(txt)
  writeLines(txt, log_con)
}

dbg <- function(...) {
  if (!DEBUG) return(invisible(NULL))
  log_msg("[DBG] ", ...)
}

step <- local({
  t0 <- NULL
  function(label) {
    now <- Sys.time()
    if (!is.null(t0))
      log_msg(sprintf("  → done in %.1fs", as.numeric(now - t0, units = "secs")))
    t0 <<- now
    log_msg("━━ ", label, " ━━")
  }
})

on.exit({
  log_msg("Log closed.")
  close(log_con)
}, add = TRUE)

log_msg(if (DEBUG) "=== DEBUG RUN (n=50) ===" else "=== FULL RUN ===")
log_msg("Output:    ", OUTPUT_FILE)
log_msg("Cache dir: ", CACHE_DIR)
log_msg("Log file:  ", LOG_FILE)

# ── 1. Load input ──────────────────────────────────────────────────────────────
step("1. Load input")
accidents_all <- read_csv(INPUT_FILE, show_col_types = FALSE)
log_msg(sprintf("  Loaded %d rows × %d cols from %s",
  nrow(accidents_all), ncol(accidents_all), INPUT_FILE))

has_coords_all <- !is.na(accidents_all$longitude) & !is.na(accidents_all$latitude)
log_msg(sprintf("  %d rows have valid coordinates (%d missing)",
  sum(has_coords_all), sum(!has_coords_all)))

# In debug mode: pick DEBUG_N accidents clustered around the median lat/lon
# so the bounding box stays small and Overpass queries are fast.
# We sort candidates by distance from the median point and take the nearest DEBUG_N.
if (DEBUG) {
  spatial_idx <- which(has_coords_all)
  med_lon <- median(accidents_all$longitude[spatial_idx], na.rm = TRUE)
  med_lat <- median(accidents_all$latitude[spatial_idx],  na.rm = TRUE)
  dist_from_med <- sqrt(
    (accidents_all$longitude[spatial_idx] - med_lon)^2 +
    (accidents_all$latitude[spatial_idx]  - med_lat)^2
  )
  debug_idx <- spatial_idx[order(dist_from_med)[seq_len(min(DEBUG_N, length(spatial_idx)))]]
  accidents  <- accidents_all[debug_idx, ]
  log_msg(sprintf(
    "  [debug] %d accidents clustered around median (%.4f, %.4f)",
    nrow(accidents), med_lon, med_lat))
  dbg("Sample collision_index values: ",
    paste(head(accidents$collision_index, 5), collapse = ", "))
} else {
  accidents <- accidents_all
}

has_coords <- !is.na(accidents$longitude) & !is.na(accidents$latitude)
accidents$.row_id <- seq_len(nrow(accidents))

accidents_spatial <- accidents[has_coords, ]
accidents_sf <- st_as_sf(
  accidents_spatial,
  coords = c("longitude", "latitude"),
  crs    = 4326
) |>
  st_transform(BNG_CRS)

log_msg(sprintf("  %d rows entering spatial pipeline", nrow(accidents_sf)))

# ── 2. Construct full road ref for OSM matching ────────────────────────────────
step("2. Build road refs")
# STATS19 stores road class and number separately; OSM uses combined ref "A1".

normalise_ref <- function(x) {
  x <- toupper(trimws(as.character(x)))
  x <- gsub("\\s+", "", x)
  ifelse(x %in% c("", "NA", "0"), NA_character_, x)
}

build_acc_ref <- function(road_class, road_number) {
  prefix  <- ROAD_CLASS_PREFIX[as.character(road_class)]
  prefix  <- ifelse(is.na(prefix), "", prefix)
  num_str <- normalise_ref(road_number)
  ifelse(!is.na(num_str) & num_str != "0" & prefix != "",
         paste0(prefix, num_str),
         NA_character_)
}

accidents_sf$acc_ref <- build_acc_ref(
  accidents_sf$first_road_class,
  accidents_sf$first_road_number
)

n_with_ref <- sum(!is.na(accidents_sf$acc_ref))
log_msg(sprintf("  %d/%d accidents have a matchable road ref (%.0f%%)",
  n_with_ref, nrow(accidents_sf),
  100 * n_with_ref / nrow(accidents_sf)))

dbg("Sample acc_ref values (first 10):")
dbg(paste(
  sprintf("  row %d: class=%s num=%s → ref=%s",
    accidents_sf$.row_id[1:min(10, nrow(accidents_sf))],
    accidents_sf$first_road_class[1:min(10, nrow(accidents_sf))],
    accidents_sf$first_road_number[1:min(10, nrow(accidents_sf))],
    accidents_sf$acc_ref[1:min(10, nrow(accidents_sf))]),
  collapse = "\n"))

# ── 3. Bounding box for Overpass (WGS84) ──────────────────────────────────────
step("3. Compute bounding box")

if (DEBUG) {
  # Clamp debug bbox to a fixed radius around the cluster centroid.
  # Even after clustering, the 50 accidents' extent can span tens of km;
  # a hard cap keeps the Overpass road_ways query fast and avoids 504 timeouts.
  pad_deg_lon <- DEBUG_BBOX_PAD_M / (111000 * cos(med_lat * pi / 180))
  pad_deg_lat <- DEBUG_BBOX_PAD_M / 111000
  bbox_wgs84 <- c(
    xmin = med_lon - pad_deg_lon,
    ymin = med_lat - pad_deg_lat,
    xmax = med_lon + pad_deg_lon,
    ymax = med_lat + pad_deg_lat
  )
  log_msg(sprintf("  [debug] Bbox clamped to %.0fm radius around centroid", DEBUG_BBOX_PAD_M))
} else {
  bbox_wgs84 <- c(
    xmin = min(accidents$longitude, na.rm = TRUE),
    ymin = min(accidents$latitude,  na.rm = TRUE),
    xmax = max(accidents$longitude, na.rm = TRUE),
    ymax = max(accidents$latitude,  na.rm = TRUE)
  )
}

log_msg(sprintf("  lon [%.4f, %.4f]  lat [%.4f, %.4f]",
  bbox_wgs84["xmin"], bbox_wgs84["xmax"],
  bbox_wgs84["ymin"], bbox_wgs84["ymax"]))

bbox_area_km2 <- (bbox_wgs84["xmax"] - bbox_wgs84["xmin"]) *
                 (bbox_wgs84["ymax"] - bbox_wgs84["ymin"]) * 111^2
log_msg(sprintf("  Approx bbox area: %.0f km²", bbox_area_km2))

# ── 4. OSM download helpers ────────────────────────────────────────────────────
# Three public Overpass mirrors — rotated on each retry so a rate-limited or
# overloaded server doesn't block all attempts.
OVERPASS_SERVERS <- c(
  "https://overpass-api.de/api/interpreter",
  "https://overpass.kumi.systems/api/interpreter",
  "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
)

fetch_or_cache <- function(key, download_fn, max_retries = 3L) {
  path <- file.path(CACHE_DIR, paste0(key, ".rds"))
  if (file.exists(path)) {
    log_msg("  [cache hit] ", key, " (", path, ")")
    return(readRDS(path))
  }

  for (attempt in seq_len(max_retries)) {
    server <- OVERPASS_SERVERS[((attempt - 1L) %% length(OVERPASS_SERVERS)) + 1L]
    # osmdata 0.4.x reads from options("osmdata.base_url"), NOT "osmdata_overpass_url".
    # set_overpass_url() also pings the server before setting it, which can block
    # for minutes. Setting the option directly bypasses both problems.
    options(osmdata.base_url = server)
    log_msg(sprintf("  [Overpass] %s — attempt %d/%d via %s …",
      key, attempt, max_retries, server))

    t1     <- Sys.time()
    result <- tryCatch(download_fn(), error = function(e) e)
    elapsed <- as.numeric(Sys.time() - t1, units = "secs")

    if (!inherits(result, "error")) {
      saveRDS(result, path)
      log_msg(sprintf("  [Overpass] %s done in %.1fs — saved to %s", key, elapsed, path))
      return(result)
    }

    msg <- conditionMessage(result)
    log_msg(sprintf("  [Overpass] %s failed on attempt %d (%.1fs): %s",
      key, attempt, elapsed, msg))

    if (attempt < max_retries) {
      # 429 = rate-limited — wait much longer before hitting a fresh server.
      # Other errors (504, timeout) only need a short pause.
      wait <- if (grepl("429", msg)) 120L else 15L
      log_msg(sprintf("  [Overpass] waiting %ds before next attempt …", wait))
      Sys.sleep(wait)
    } else {
      stop(sprintf("Overpass query '%s' failed after %d attempts: %s",
        key, max_retries, msg))
    }
  }
}

# ── 5. Bulk OSM downloads ──────────────────────────────────────────────────────
step("5. OSM downloads")

# highway=* without a value filter returns ALL osm ways including footpaths,
# cycleways, steps etc. — that's >2 GB for a UK-wide bbox and will OOM Overpass.
# We restrict to driveable types that can appear in STATS19 accident records.
DRIVEABLE_HIGHWAY_TYPES <- c(
  "motorway",    "motorway_link",
  "trunk",       "trunk_link",
  "primary",     "primary_link",
  "secondary",   "secondary_link",
  "tertiary",    "tertiary_link",
  "residential", "unclassified",
  "road",        "living_street",
  "service"
)

# 2 GB Overpass memory budget (server default is 512 MB)
OVERPASS_MEMSIZE <- 2 * 1024^3

speed_cam_osm <- fetch_or_cache("speed_cameras", function() {
  opq(bbox = bbox_wgs84, timeout = 300, memsize = OVERPASS_MEMSIZE) |>
    add_osm_feature(key = "highway", value = "speed_camera") |>
    osmdata_sf()
})

traffic_calm_osm <- fetch_or_cache("traffic_calming", function() {
  opq(bbox = bbox_wgs84, timeout = 300, memsize = OVERPASS_MEMSIZE) |>
    add_osm_feature(key = "traffic_calming") |>
    osmdata_sf()
})

traffic_signal_osm <- fetch_or_cache("traffic_signals", function() {
  opq(bbox = bbox_wgs84, timeout = 300, memsize = OVERPASS_MEMSIZE) |>
    add_osm_feature(key = "highway", value = "traffic_signals") |>
    osmdata_sf()
})

# road_ways + barrier_ways strategy:
#   Debug mode  — Overpass with the small clamped bbox (works fine, no timeouts).
#   Full run    — osmextract/Geofabrik PBF (national scale exceeds Overpass limits).
road_ways_osm    <- NULL   # populated below for debug mode
barrier_ways_osm <- NULL   # populated below for debug mode

if (DEBUG) {
  road_ways_osm <- fetch_or_cache("road_ways", function() {
    opq(bbox = bbox_wgs84, timeout = 300, memsize = OVERPASS_MEMSIZE) |>
      add_osm_feature(key = "highway", value = DRIVEABLE_HIGHWAY_TYPES) |>
      osmdata_sf()
  })

  barrier_ways_osm <- fetch_or_cache("barrier_ways", function() {
    opq(bbox = bbox_wgs84, timeout = 300, memsize = OVERPASS_MEMSIZE) |>
      add_osm_feature(key = "barrier") |>
      osmdata_sf()
  })
}

# ── 6. Prepare sf objects ──────────────────────────────────────────────────────
step("6. Prepare OSM sf objects")

select_existing <- function(sf_obj, wanted) {
  sf_obj[, intersect(wanted, names(sf_obj)), drop = FALSE]
}

prepare_ways <- function(osm_obj) {
  ln <- osm_obj$osm_lines
  if (is.null(ln) || nrow(ln) == 0) return(NULL)
  # osmdata sometimes stores the OSM id only as row names, not as a column
  if (!"osm_id" %in% names(ln)) ln$osm_id <- rownames(ln)
  ln <- select_existing(ln, c(
    "osm_id", "ref", "highway",
    "lanes", "divider", "barrier"
  ))
  ln |> st_transform(BNG_CRS)
}

prepare_nodes <- function(osm_obj) {
  pt <- osm_obj$osm_points
  if (is.null(pt) || nrow(pt) == 0) return(NULL)
  if (!"osm_id" %in% names(pt)) pt$osm_id <- rownames(pt)
  pt <- select_existing(pt, c("osm_id", "highway", "traffic_calming", "ref"))
  pt |> st_transform(BNG_CRS)
}

speed_cam_nodes      <- prepare_nodes(speed_cam_osm)
traffic_calm_nodes   <- prepare_nodes(traffic_calm_osm)
traffic_signal_nodes <- prepare_nodes(traffic_signal_osm)

# ── HSTORE parser for osmextract other_tags column ────────────────────────────
# osmextract stores non-default tags in an OGR HSTORE field:
#   "key1"=>"val1","key2"=>"val2",...
# Using extra_tags in oe_read() modifies osmconf.ini and can break vectortranslate.
# Parsing other_tags in R is more robust and version-agnostic.
hstore_get <- function(hstore_vec, key) {
  pat  <- paste0('"', key, '"=>"([^"]*)"')
  m    <- regexpr(pat, hstore_vec, perl = TRUE)
  out  <- rep(NA_character_, length(hstore_vec))
  ok   <- !is.na(m) & m > 0
  if (any(ok)) {
    starts  <- attr(m, "capture.start")[ok]
    lengths <- attr(m, "capture.length")[ok]
    out[ok] <- substr(hstore_vec[ok], starts, starts + lengths - 1L)
  }
  out
}

# ── osmextract helper (full run only; debug uses Overpass above) ───────────────
# Reads a layer from the Geofabrik PBF via oe_read(), caches result as RDS.
# No extra_tags: we rely on the default other_tags HSTORE column to get ref,
# lanes, surface, etc., which avoids osmconf.ini modification and vectortranslate
# failures caused by unrecognised extra_tags in some osmextract/GDAL versions.
osmextract_layer <- function(key, layer, sql, target_crs) {
  cache_path <- file.path(CACHE_DIR, paste0(key, ".rds"))
  if (file.exists(cache_path)) {
    log_msg("  [cache hit] ", key)
    return(readRDS(cache_path))
  }
  log_msg(sprintf("  [osmextract] %s — downloading/converting PBF (first run only) …", key))
  t1 <- Sys.time()
  result <- oe_read(
    file_path          = GEOFABRIK_URL,
    layer              = layer,
    extra_tags         = character(0),
    download_directory = CACHE_DIR,
    quiet              = FALSE,
    query              = sql
  ) |> st_transform(target_crs)
  if (!"osm_id" %in% names(result)) result$osm_id <- rownames(result)
  elapsed <- as.numeric(Sys.time() - t1, units = "secs")
  log_msg(sprintf("  [osmextract] %s: %d features in %.1fs", key, nrow(result), elapsed))
  saveRDS(result, cache_path)
  result
}

# ── road_ways ─────────────────────────────────────────────────────────────────
if (DEBUG) {
  road_ways <- prepare_ways(road_ways_osm)
} else {
  road_ways_raw <- osmextract_layer(
    key        = "road_ways_raw",
    layer      = "lines",
    sql        = sprintf(
      "SELECT osm_id, highway, other_tags, geometry FROM lines WHERE highway IN ('%s')",
      paste(DRIVEABLE_HIGHWAY_TYPES, collapse = "', '")
    ),
    target_crs = BNG_CRS
  )
  if (!is.null(road_ways_raw) && nrow(road_ways_raw) > 0) {
    ot <- if ("other_tags" %in% names(road_ways_raw)) road_ways_raw$other_tags else NA_character_
    road_ways <- road_ways_raw |>
      mutate(
        ref     = hstore_get(ot, "ref"),
        lanes   = hstore_get(ot, "lanes"),
        divider = hstore_get(ot, "divider")
      )
    if ("other_tags" %in% names(road_ways)) road_ways$other_tags <- NULL
  } else {
    road_ways <- road_ways_raw
  }
  road_ways <- select_existing(road_ways,
    c("osm_id", "ref", "highway", "lanes", "divider"))
}

# ── barrier_ways ──────────────────────────────────────────────────────────────
if (DEBUG) {
  barrier_ways <- prepare_ways(barrier_ways_osm)
} else {
  barrier_ways <- osmextract_layer(
    key        = "barrier_ways",
    layer      = "lines",
    sql        = "SELECT osm_id, barrier, geometry FROM lines WHERE barrier IS NOT NULL",
    target_crs = BNG_CRS
  )
}

log_msg("  OSM object counts:")
for (item in list(
  list("road_ways",            road_ways),
  list("barrier_ways",         barrier_ways),
  list("speed_cam_nodes",      speed_cam_nodes),
  list("traffic_calm_nodes",   traffic_calm_nodes),
  list("traffic_signal_nodes", traffic_signal_nodes)
)) {
  n <- if (is.null(item[[2]])) 0L else nrow(item[[2]])
  log_msg(sprintf("    %-25s %d", item[[1]], n))
}

# Attach normalised ref to road ways
if (!is.null(road_ways)) {
  road_ways$way_ref <- normalise_ref(
    if ("ref" %in% names(road_ways)) road_ways$ref else NA_character_
  )
  n_way_ref <- sum(!is.na(road_ways$way_ref))
  log_msg(sprintf("  %d/%d road ways carry a ref tag (%.0f%%)",
    n_way_ref, nrow(road_ways), 100 * n_way_ref / nrow(road_ways)))

  dbg("road_ways columns present: ", paste(names(road_ways), collapse = ", "))
  dbg("Sample way refs: ", paste(head(na.omit(road_ways$way_ref), 10), collapse = ", "))
}

# ── 7. Core: two-stage (ref-first, then nearest) way matching ─────────────────
#
# For each accident, find all road ways within `radius_m`.
# Stage 1 — ref match: keep only ways whose ref matches the accident's road ref;
#            pick the nearest among those.
# Stage 2 — fallback: if no ref match (or acc_ref is NA), pick the globally
#            nearest way within radius instead.
#
# Returns a data frame: .row_id | matched_osm_id | matched_dist_m

match_ways <- function(acc_sf, ways_sf, radius_m) {
  empty <- tibble(.row_id        = acc_sf$.row_id,
                  matched_osm_id = NA_character_,
                  matched_dist_m = NA_real_)
  if (is.null(ways_sf) || nrow(ways_sf) == 0) return(empty)

  acc_buf <- st_buffer(acc_sf[c(".row_id", "acc_ref")], radius_m)
  cands   <- st_join(acc_buf, ways_sf[c("osm_id", "way_ref")],
                     join = st_intersects) |>
    st_drop_geometry() |>
    filter(!is.na(osm_id))

  if (nrow(cands) == 0) return(empty)

  acc_idx      <- match(cands$.row_id, acc_sf$.row_id)
  way_idx      <- match(cands$osm_id,  ways_sf$osm_id)
  cands$dist_m <- as.numeric(
    st_distance(acc_sf[acc_idx, ], ways_sf[way_idx, ], by_element = TRUE)
  )

  # Handle semicolon-separated OSM refs ("A1;A2")
  cands$ref_match <- mapply(function(way_ref, acc_ref) {
    if (is.na(way_ref) || is.na(acc_ref)) return(FALSE)
    parts <- normalise_ref(strsplit(way_ref, ";", fixed = TRUE)[[1]])
    acc_ref %in% parts
  }, cands$way_ref, cands$acc_ref, USE.NAMES = FALSE)

  cands |>
    group_by(.row_id) |>
    mutate(any_ref_match = any(ref_match)) |>
    filter(!any_ref_match | ref_match) |>   # Stage 1 if possible, else all
    slice_min(dist_m, n = 1, with_ties = FALSE) |>
    ungroup() |>
    select(.row_id, matched_osm_id = osm_id, matched_dist_m = dist_m)
}

# ── 8. Way-level features ──────────────────────────────────────────────────────
step("8. Way-level features (lanes/divider)")
log_msg(sprintf("  Matching %d accidents to road ways within %dm …",
  nrow(accidents_sf), WAY_RADIUS_M))

way_matches <- match_ways(accidents_sf, road_ways, WAY_RADIUS_M)

n_matched  <- sum(!is.na(way_matches$matched_osm_id))
n_unmatched <- sum(is.na(way_matches$matched_osm_id))
log_msg(sprintf("  Way matches: %d matched, %d unmatched (%.1f%% coverage)",
  n_matched, n_unmatched, 100 * n_matched / nrow(way_matches)))

dbg("Matched way distance summary (metres):")
dbg(sprintf("  min=%.1f  median=%.1f  mean=%.1f  max=%.1f",
  min(way_matches$matched_dist_m, na.rm = TRUE),
  median(way_matches$matched_dist_m, na.rm = TRUE),
  mean(way_matches$matched_dist_m, na.rm = TRUE),
  max(way_matches$matched_dist_m, na.rm = TRUE)))

parse_lanes <- function(x) {
  x <- sub(";.*", "", as.character(x))   # "2;1" → "2"
  suppressWarnings(as.integer(x))
}

if (!is.null(road_ways)) {
  tag_cols <- intersect(
    c("osm_id", "lanes", "divider"),
    names(road_ways)
  )
  dbg("Way tag columns available: ", paste(tag_cols, collapse = ", "))

  way_tags <- st_drop_geometry(road_ways)[, tag_cols, drop = FALSE] |>
    rename(matched_osm_id = osm_id)

  way_raw <- way_matches |>
    left_join(way_tags, by = "matched_osm_id")

  # Ensure every expected tag column exists (OSM may not return all tags)
  for (.col in c("lanes", "divider")) {
    if (!.col %in% names(way_raw)) {
      way_raw[[.col]] <- NA_character_
      dbg("  tag '", .col, "' absent from OSM download — filled with NA")
    }
  }

  way_features <- way_raw |>
    transmute(
      .row_id,
      lane_count      = parse_lanes(lanes),
      has_divider_tag = !is.na(divider)
    )
} else {
  log_msg("  No road ways downloaded — all way-level features will be NA")
  way_features <- tibble(
    .row_id         = accidents_sf$.row_id,
    lane_count      = NA_integer_,
    has_divider_tag = FALSE
  )
}

dbg("Way feature fill rates:")
for (.c in c("lane_count")) {
  pct <- 100 * mean(!is.na(way_features[[.c]]))
  dbg(sprintf("  %-25s %.1f%%", .c, pct))
}

# ── 9. has_median ─────────────────────────────────────────────────────────────
# A median is inferred from either:
#   (a) the matched road way carries a `divider` tag (any value), OR
#   (b) a separate barrier=* way passes within MEDIAN_BARRIER_RADIUS of the
#       accident point (e.g. barrier=separator mapped as a central reservation).
step("9. Median / barrier check")

if (!is.null(barrier_ways) && nrow(barrier_ways) > 0) {
  log_msg(sprintf("  Checking %d barrier ways within %dm of each accident …",
    nrow(barrier_ways), MEDIAN_BARRIER_RADIUS))
  acc_buf_m   <- st_buffer(accidents_sf[".row_id"], MEDIAN_BARRIER_RADIUS)
  barrier_hits <- st_join(acc_buf_m,
                           barrier_ways["osm_id"],
                           join = st_intersects) |>
    st_drop_geometry() |>
    filter(!is.na(osm_id)) |>
    distinct(.row_id) |>
    mutate(has_barrier_nearby = TRUE)
  log_msg(sprintf("  %d accidents have a nearby barrier way", nrow(barrier_hits)))
} else {
  log_msg("  No barrier ways found — has_median will rely on divider tag only")
  barrier_hits <- tibble(.row_id = integer(0), has_barrier_nearby = logical(0))
}

way_features <- way_features |>
  left_join(barrier_hits, by = ".row_id") |>
  mutate(
    has_median         = as.integer(has_divider_tag | (!is.na(has_barrier_nearby) & has_barrier_nearby)),
    has_divider_tag    = NULL,
    has_barrier_nearby = NULL
  )

log_msg(sprintf("  has_median = 1 for %d/%d accidents (%.1f%%)",
  sum(way_features$has_median, na.rm = TRUE),
  nrow(way_features),
  100 * mean(way_features$has_median, na.rm = TRUE)))

# ── 10. Point-to-way ref indirection ──────────────────────────────────────────
# Speed cameras and traffic calming nodes rarely carry a `ref` tag themselves.
# We assign each point the `ref` of its nearest road way (within POINT_WAY_RADIUS_M).
# This is used as the point's "road identity" in Stage 1 matching.

assign_point_road_ref <- function(pt_sf, ways_sf, max_dist_m) {
  if (is.null(pt_sf) || nrow(pt_sf) == 0 || is.null(ways_sf) || nrow(ways_sf) == 0) {
    pt_sf$point_road_ref <- NA_character_
    return(pt_sf)
  }
  nearest_idx  <- st_nearest_feature(pt_sf, ways_sf)
  nearest_dist <- as.numeric(
    st_distance(pt_sf, ways_sf[nearest_idx, ], by_element = TRUE)
  )
  pt_sf$point_road_ref <- ifelse(
    nearest_dist <= max_dist_m,
    ways_sf$way_ref[nearest_idx],
    NA_character_
  )
  n_assigned <- sum(!is.na(pt_sf$point_road_ref))
  log_msg(sprintf("  %d/%d nodes assigned a road ref via nearest way (within %dm)",
    n_assigned, nrow(pt_sf), max_dist_m))
  pt_sf
}

# ── 11. Two-stage point matching ───────────────────────────────────────────────
# Returns: .row_id | has_feature (0/1) | best_dist_m

match_points <- function(acc_sf, pt_sf, radius_m, label = "point") {
  empty <- tibble(.row_id     = acc_sf$.row_id,
                  has_feature = 0L,
                  best_dist_m = NA_real_)
  if (is.null(pt_sf) || nrow(pt_sf) == 0) {
    log_msg(sprintf("  No %s nodes available — all set to 0", label))
    return(empty)
  }

  acc_buf <- st_buffer(acc_sf[c(".row_id", "acc_ref")], radius_m)
  cands   <- st_join(acc_buf,
                     pt_sf[c("osm_id", "point_road_ref")],
                     join = st_intersects) |>
    st_drop_geometry() |>
    filter(!is.na(osm_id))

  log_msg(sprintf("  %d candidate (accident, %s) pairs within %dm",
    nrow(cands), label, radius_m))

  if (nrow(cands) == 0) return(empty)

  acc_idx      <- match(cands$.row_id, acc_sf$.row_id)
  pt_idx       <- match(cands$osm_id,  pt_sf$osm_id)
  cands$dist_m <- as.numeric(
    st_distance(acc_sf[acc_idx, ], pt_sf[pt_idx, ], by_element = TRUE)
  )

  cands$ref_match <- mapply(function(pt_ref, acc_ref) {
    if (is.na(pt_ref) || is.na(acc_ref)) return(FALSE)
    parts <- normalise_ref(strsplit(pt_ref, ";", fixed = TRUE)[[1]])
    acc_ref %in% parts
  }, cands$point_road_ref, cands$acc_ref, USE.NAMES = FALSE)

  n_ref_match <- sum(cands$ref_match)
  log_msg(sprintf("  %d/%d candidate pairs pass ref filter (%.0f%% use fallback)",
    n_ref_match, nrow(cands),
    100 * (1 - n_ref_match / nrow(cands))))

  result <- cands |>
    group_by(.row_id) |>
    mutate(any_ref_match = any(ref_match)) |>
    filter(!any_ref_match | ref_match) |>
    slice_min(dist_m, n = 1, with_ties = FALSE) |>
    ungroup() |>
    transmute(.row_id, has_feature = 1L, best_dist_m = dist_m)

  log_msg(sprintf("  %d/%d accidents have at least one %s within %dm",
    nrow(result), nrow(acc_sf), label, radius_m))

  empty |>
    filter(!.row_id %in% result$.row_id) |>
    bind_rows(result)
}

# ── 12. Speed camera features ──────────────────────────────────────────────────
step("12. Speed camera features")
speed_cam_nodes <- assign_point_road_ref(speed_cam_nodes, road_ways, POINT_WAY_RADIUS_M)
cam_result      <- match_points(accidents_sf, speed_cam_nodes, SPEED_CAM_RADIUS_M,
                                label = "speed_camera")
cam_features <- cam_result |>
  transmute(
    .row_id,
    has_speed_camera       = has_feature,
    dist_to_speed_camera_m = best_dist_m
  )

dbg("Speed camera distance summary (metres, among accidents with a camera):")
cam_dists <- cam_features$dist_to_speed_camera_m[!is.na(cam_features$dist_to_speed_camera_m)]
if (length(cam_dists) > 0)
  dbg(sprintf("  min=%.1f  median=%.1f  max=%.1f",
    min(cam_dists), median(cam_dists), max(cam_dists)))

# ── 13. Traffic calming features ───────────────────────────────────────────────
step("13. Traffic calming features")
traffic_calm_nodes <- assign_point_road_ref(traffic_calm_nodes, road_ways, POINT_WAY_RADIUS_M)
tc_result          <- match_points(accidents_sf, traffic_calm_nodes, TRAFFIC_CALM_RADIUS_M,
                                   label = "traffic_calming")
tc_features <- tc_result |>
  transmute(
    .row_id,
    has_traffic_calming = has_feature
  )

# ── 14. Traffic signal features ────────────────────────────────────────────────
step("14. Traffic signal features")
traffic_signal_nodes <- assign_point_road_ref(traffic_signal_nodes, road_ways, POINT_WAY_RADIUS_M)
signal_result         <- match_points(accidents_sf, traffic_signal_nodes, TRAFFIC_SIGNAL_RADIUS_M,
                                      label = "traffic_signal")
signal_features <- signal_result |>
  transmute(
    .row_id,
    has_traffic_signals = has_feature
  )

# ── 15. Assemble all OSM features ─────────────────────────────────────────────
step("15. Assemble output")

osm_features <- accidents_sf |>
  st_drop_geometry() |>
  select(.row_id) |>
  left_join(cam_features,    by = ".row_id") |>
  left_join(tc_features,     by = ".row_id") |>
  left_join(signal_features, by = ".row_id") |>
  left_join(way_features,    by = ".row_id")

# For rows with no valid coordinates, all OSM columns will be NA (left join fills NA)
output <- accidents |>
  left_join(osm_features, by = ".row_id") |>
  select(-.row_id)

log_msg(sprintf("  Output table: %d rows × %d cols", nrow(output), ncol(output)))

# ── 16. Feature statistics ────────────────────────────────────────────────────
new_cols <- c(
  "has_speed_camera", "dist_to_speed_camera_m", "lane_count",
  "has_median", "has_traffic_calming", "has_traffic_signals"
)
n <- nrow(output)

log_msg(sprintf("  ── OSM feature statistics (n = %d) ──", n))

# helper: count and percentage of non-NA values
fill_str <- function(x) {
  k <- sum(!is.na(x))
  sprintf("%d / %d (%.1f%%)", k, n, 100 * k / n)
}

# ── binary features ────────────────────────────────────────────────────────────
for (col in c("has_speed_camera", "has_median", "has_traffic_calming",
              "has_traffic_signals")) {
  if (!col %in% names(output)) next
  x   <- output[[col]]
  yes <- sum(x == 1L, na.rm = TRUE)
  no  <- sum(x == 0L, na.rm = TRUE)
  log_msg(sprintf("  %-28s  yes=%d (%.1f%%)  no=%d (%.1f%%)  NA=%d",
    col,
    yes, 100 * yes / n,
    no,  100 * no  / n,
    sum(is.na(x))))
}

# ── speed camera distance (continuous, only where has_speed_camera = 1) ────────
if ("dist_to_speed_camera_m" %in% names(output)) {
  d <- output$dist_to_speed_camera_m[!is.na(output$dist_to_speed_camera_m)]
  if (length(d) > 0) {
    log_msg(sprintf("  %-28s  n=%d  min=%.0fm  median=%.0fm  mean=%.0fm  max=%.0fm",
      "dist_to_speed_camera_m",
      length(d), min(d), median(d), mean(d), max(d)))
  } else {
    log_msg("  dist_to_speed_camera_m   no observations")
  }
}

# ── lane count (numeric — distribution of common values) ──────────────────────
if ("lane_count" %in% names(output)) {
  lc <- output$lane_count[!is.na(output$lane_count)]
  log_msg(sprintf("  %-28s  filled=%s", "lane_count", fill_str(output$lane_count)))
  if (length(lc) > 0) {
    tbl <- sort(table(lc), decreasing = TRUE)
    top <- head(tbl, 6)
    log_msg(sprintf("    distribution: %s",
      paste(sprintf("%s lanes: %d", names(top), as.integer(top)), collapse = "  |  ")))
    log_msg(sprintf("    median=%.0f  mean=%.1f  range=[%d, %d]",
      median(lc), mean(lc), min(lc), max(lc)))
  }
}

if (DEBUG) {
  log_msg("  [debug] Sample output rows (first 5):")
  sample_rows <- output |>
    select(collision_index, any_of(c("first_road_class", "first_road_number")),
           any_of(new_cols)) |>
    head(5)
  for (i in seq_len(nrow(sample_rows))) {
    log_msg("    row ", i, ": ",
      paste(names(sample_rows), as.character(sample_rows[i, ]), sep = "=", collapse = "  "))
  }
}

# ── 17. Write output ───────────────────────────────────────────────────────────
step("17. Write output")
write_csv(output, OUTPUT_FILE)
log_msg(sprintf("  Written: %s", OUTPUT_FILE))
log_msg(sprintf("  Done. %d rows × %d columns.", nrow(output), ncol(output)))
if (DEBUG) log_msg("  [debug] Re-run without --debug for full dataset.")

# Final elapsed
log_msg(sprintf("  → done in %.1fs",
  as.numeric(Sys.time() - get("t0", envir = environment(step)), units = "secs")))

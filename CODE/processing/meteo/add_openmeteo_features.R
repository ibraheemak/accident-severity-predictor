#!/usr/bin/env Rscript
# add_openmeteo_features.R
#
# Adds Open-Meteo historical weather features to the STATS19+AADF+OSM combined
# real-time feature table.
#
# STATS19 records a subjective, officer-assessed weather category at the
# scene. Open-Meteo's Historical Weather API (ERA5/ERA5-Land reanalysis)
# provides objective, hourly weather at the exact accident lat/lon and hour —
# this replaces that subjective field with continuous measurements.
#
# Source: https://open-meteo.com/en/docs/historical-weather-api
#
# NOTE: visibility is deliberately omitted — the archive endpoint (ERA5/
# ERA5-Land reanalysis) does not model it and returns null for every hour.
#
# Usage:
#   Rscript add_openmeteo_features.R            # full run
#   Rscript add_openmeteo_features.R --debug    # 50-row smoke test + verbose logging

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(httr2)
})

# ── Debug flag ─────────────────────────────────────────────────────────────────
args  <- commandArgs(trailingOnly = TRUE)
DEBUG <- "--debug" %in% args
DEBUG_N <- 50L   # accidents to process in debug mode

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_FILE <- "data/features/stats19+aadf+osm/stats19+aadf+osm_combined_realtime_features.csv"

if (DEBUG) {
  OUTPUT_DIR  <- "data/features/stats19+aadf+osm+meteo"
  OUTPUT_FILE <- file.path(OUTPUT_DIR, "stats19+aadf+osm+meteo_combined_realtime_features_debug.csv")
  CACHE_DIR   <- "data/raw/openmeteo/cache/debug"
  LOG_FILE    <- file.path(OUTPUT_DIR, "add_openmeteo_features_debug.log")
} else {
  OUTPUT_DIR  <- "data/features/stats19+aadf+osm+meteo"
  OUTPUT_FILE <- file.path(OUTPUT_DIR, "stats19+aadf+osm+meteo_combined_realtime_features.csv")
  CACHE_DIR   <- "data/raw/openmeteo/cache"
  LOG_FILE    <- file.path(OUTPUT_DIR, "add_openmeteo_features.log")
}

# Free Historical Weather API endpoint. Set OPENMETEO_API_KEY (and optionally
# OPENMETEO_BASE_URL) for the Professional plan, which raises rate limits and
# unlocks a commercial-use license.
OPENMETEO_BASE_URL <- Sys.getenv("OPENMETEO_BASE_URL", "https://archive-api.open-meteo.com/v1/archive")
OPENMETEO_API_KEY  <- Sys.getenv("OPENMETEO_API_KEY", "")

HOURLY_VARS <- c(
  "temperature_2m", "precipitation", "wind_speed_10m",
  "weather_code", "snowfall", "snow_depth"
)

MAX_BATCH_SIZE <- 100L   # locations per API request (per distinct date)
MAX_RETRIES    <- 3L

# Self-throttle: free/anonymous tier is best-effort and shared by IP. Without
# pacing, this loop fires the next request the instant the previous one
# returns, which is what tips it into a sustained 429 — and once tripped, a
# short per-call backoff isn't enough to clear it. MIN_REQUEST_INTERVAL_SEC
# enforces a floor between consecutive *actual* network calls (cache hits are
# exempt). CIRCUIT_BREAKER_COOLDOWN is how long a rate-limited batch waits
# before retrying — a 429 is a transient throttle, not a real data error, so
# the same batch is retried (not skipped) until it succeeds or hits the cap.
MIN_REQUEST_INTERVAL_SEC <- 2
CIRCUIT_BREAKER_COOLDOWN <- 600L   # 10 minutes

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
      log_msg(sprintf("  -> done in %.1fs", as.numeric(now - t0, units = "secs")))
    t0 <<- now
    log_msg("== ", label, " ==")
  }
})

# NOTE: no top-level on.exit() here — when this script is source()d (e.g. by
# run_pipeline.R), a top-level on.exit fires as soon as its own expression
# finishes evaluating, closing log_con and crashing the next log_msg call.
# The log is closed explicitly at the end of the script instead; on error the
# process/session teardown reclaims the connection.

log_msg(if (DEBUG) "=== DEBUG RUN (n=50) ===" else "=== FULL RUN ===")
log_msg("Endpoint:  ", OPENMETEO_BASE_URL, if (nzchar(OPENMETEO_API_KEY)) " (API key set)" else " (free tier)")
log_msg("Output:    ", OUTPUT_FILE)
log_msg("Cache dir: ", CACHE_DIR)
log_msg("Log file:  ", LOG_FILE)

# ── 1. Load input ──────────────────────────────────────────────────────────────
step("1. Load input")
accidents_all <- read_csv(INPUT_FILE, show_col_types = FALSE)
log_msg(sprintf("  Loaded %d rows x %d cols from %s",
  nrow(accidents_all), ncol(accidents_all), INPUT_FILE))

if (DEBUG) {
  accidents <- accidents_all |> slice_head(n = DEBUG_N)
  log_msg(sprintf("  [debug] kept first %d rows", nrow(accidents)))
} else {
  accidents <- accidents_all
}

accidents$.row_id <- seq_len(nrow(accidents))

# ── 2. Parse date/time into request keys ───────────────────────────────────────
step("2. Parse date/time")
accidents <- accidents |>
  mutate(
    .date = as.Date(date, format = "%d/%m/%Y"),
    .hour = suppressWarnings(as.integer(substr(time, 1, 2)))
  )

has_join_keys <- with(accidents,
  !is.na(longitude) & !is.na(latitude) & !is.na(.date) & !is.na(.hour))
log_msg(sprintf("  %d/%d rows have valid lat/lon/date/time (%d missing)",
  sum(has_join_keys), nrow(accidents), sum(!has_join_keys)))

joinable    <- accidents[has_join_keys, ]
unjoinable  <- accidents[!has_join_keys, ]

# ── 3. Open-Meteo fetch helper ──────────────────────────────────────────────────
# One request per (date, batch-of-locations). Cached to disk by date+batch
# index so reruns are free. Locations within a request share a single
# start_date/end_date day; the response returns the full 24h series per point.
.last_request_time <- NULL

throttle <- function() {
  if (!is.null(.last_request_time)) {
    elapsed <- as.numeric(Sys.time() - .last_request_time, units = "secs")
    if (elapsed < MIN_REQUEST_INTERVAL_SEC) Sys.sleep(MIN_REQUEST_INTERVAL_SEC - elapsed)
  }
  .last_request_time <<- Sys.time()
}

# Returns list(body = ..., rate_limited = TRUE/FALSE) so the caller can drive
# the circuit breaker. body is NULL on failure.
fetch_or_cache <- function(key, request_fn, max_retries = MAX_RETRIES) {
  path <- file.path(CACHE_DIR, paste0(key, ".rds"))
  if (file.exists(path)) {
    dbg("[cache hit] ", key)
    return(list(body = readRDS(path), rate_limited = FALSE))
  }

  backoff <- c(60, 180, 300)   # escalating cooldown on consecutive 429s
  for (attempt in seq_len(max_retries)) {
    throttle()
    result <- tryCatch(request_fn(), error = function(e) e)
    if (!inherits(result, "error")) {
      saveRDS(result, path)
      return(list(body = result, rate_limited = FALSE))
    }
    msg         <- conditionMessage(result)
    is_429      <- grepl("429", msg)
    log_msg(sprintf("  [Open-Meteo] %s failed on attempt %d: %s", key, attempt, msg))
    if (attempt < max_retries) {
      wait <- if (is_429) backoff[min(attempt, length(backoff))] else 5L
      log_msg(sprintf("  [Open-Meteo] waiting %ds before retry …", wait))
      Sys.sleep(wait)
    } else {
      log_msg(sprintf("  [Open-Meteo] %s gave up after %d attempts — leaving NA", key, max_retries))
      return(list(body = NULL, rate_limited = is_429))
    }
  }
}

call_openmeteo <- function(lats, lons, date_str) {
  req <- request(OPENMETEO_BASE_URL) |>
    req_url_query(
      latitude   = paste(lats, collapse = ","),
      longitude  = paste(lons, collapse = ","),
      start_date = date_str,
      end_date   = date_str,
      hourly     = paste(HOURLY_VARS, collapse = ","),
      timezone   = "Europe/London",
      .multi     = "comma"
    )
  if (nzchar(OPENMETEO_API_KEY)) req <- req_url_query(req, apikey = OPENMETEO_API_KEY)

  resp <- req_perform(req)
  resp_body_json(resp, simplifyVector = FALSE)
}

# Normalises a single-location response (named list) into the same shape as
# one element of a multi-location response (Open-Meteo drops the array
# wrapper when only one location is requested).
as_location_list <- function(body, n_locations) {
  if (n_locations == 1L) list(body) else body
}

# ── 4. Batch accidents by date ──────────────────────────────────────────────────
step("4. Batch accidents by date and call Open-Meteo")

dates <- sort(unique(joinable$.date))
log_msg(sprintf("  %d distinct dates across %d joinable rows", length(dates), nrow(joinable)))

weather_rows <- vector("list", length(dates))
MAX_COOLDOWN_CYCLES <- 20L   # ~200 min cap before a rate-limited batch finally gives up

for (di in seq_along(dates)) {
  d        <- dates[di]
  date_str <- format(d, "%Y-%m-%d")
  day_rows <- joinable[joinable$.date == d, ]

  if (di %% 100 == 0 || di == length(dates))
    log_msg(sprintf("  date %d/%d (%s) — %d accidents", di, length(dates), date_str, nrow(day_rows)))

  n_batches <- ceiling(nrow(day_rows) / MAX_BATCH_SIZE)
  day_weather <- vector("list", n_batches)

  for (bi in seq_len(n_batches)) {
    idx       <- ((bi - 1L) * MAX_BATCH_SIZE + 1L):min(bi * MAX_BATCH_SIZE, nrow(day_rows))
    batch     <- day_rows[idx, ]
    # Key includes batch size + coordinate checksums: cached responses are
    # matched back to batch rows by position, so an entry must never be
    # served to a batch with different membership (e.g. after re-sampling
    # the STATS19 download) — that would silently attach wrong-location
    # weather. With these components in the key, a changed sample is a
    # clean cache miss instead of corruption.
    cache_key <- sprintf("%s_b%d_n%d_%.4f_%.4f", date_str, bi,
                         nrow(batch), sum(batch$latitude), sum(batch$longitude))

    # A 429 is a transient global throttle, not a real data error — retrying
    # the same batch after a long cooldown almost always succeeds. Only a
    # non-429 failure (e.g. malformed response) gives up and writes NA.
    body <- NULL
    for (cycle in seq_len(MAX_COOLDOWN_CYCLES)) {
      fetched <- fetch_or_cache(cache_key, function() {
        call_openmeteo(batch$latitude, batch$longitude, date_str)
      })
      if (!fetched$rate_limited) { body <- fetched$body; break }
      log_msg(sprintf(
        "  [circuit breaker] %s still rate-limited (cycle %d/%d) — cooling down %ds",
        cache_key, cycle, MAX_COOLDOWN_CYCLES, CIRCUIT_BREAKER_COOLDOWN))
      Sys.sleep(CIRCUIT_BREAKER_COOLDOWN)
    }

    if (is.null(body)) {
      day_weather[[bi]] <- tibble(.row_id = batch$.row_id) |>
        bind_cols(as_tibble(setNames(rep(list(NA_real_), length(HOURLY_VARS)), HOURLY_VARS)))
      next
    }

    locs <- as_location_list(body, nrow(batch))

    extract_one <- function(loc, target_hour) {
      hourly <- loc$hourly
      if (is.null(hourly) || is.null(hourly$time)) {
        return(setNames(as.list(rep(NA_real_, length(HOURLY_VARS))), HOURLY_VARS))
      }
      hour_idx <- target_hour + 1L   # 24 hourly entries, 1-indexed in R
      vals <- lapply(HOURLY_VARS, function(v) {
        col <- hourly[[v]]
        if (is.null(col) || hour_idx > length(col)) return(NA_real_)
        val <- col[[hour_idx]]
        if (is.null(val)) NA_real_ else as.numeric(val)
      })
      setNames(vals, HOURLY_VARS)
    }

    extracted <- lapply(seq_len(nrow(batch)), function(i) extract_one(locs[[i]], batch$.hour[i]))
    day_weather[[bi]] <- bind_cols(
      tibble(.row_id = batch$.row_id),
      bind_rows(extracted)
    )
  }

  weather_rows[[di]] <- bind_rows(day_weather)
}

weather <- bind_rows(weather_rows)
log_msg(sprintf("  Collected weather for %d/%d joinable rows", nrow(weather), nrow(joinable)))

# ── 5. Assemble output ─────────────────────────────────────────────────────────
step("5. Assemble output")

na_weather <- as_tibble(setNames(rep(list(NA_real_), length(HOURLY_VARS)), HOURLY_VARS))

weather_all <- bind_rows(
  weather,
  bind_cols(tibble(.row_id = unjoinable$.row_id), na_weather[rep(1, nrow(unjoinable)), , drop = FALSE])
)

output <- accidents |>
  left_join(weather_all, by = ".row_id") |>
  select(-.row_id, -.date, -.hour) |>
  rename(
    temperature_2m_c     = temperature_2m,
    precipitation_mm     = precipitation,
    wind_speed_10m_kmh   = wind_speed_10m,
    weather_code         = weather_code,
    snowfall_cm          = snowfall,
    snow_depth_m         = snow_depth
  )

log_msg(sprintf("  Output table: %d rows x %d cols", nrow(output), ncol(output)))

# ── 6. Feature statistics ──────────────────────────────────────────────────────
new_cols <- c("temperature_2m_c", "precipitation_mm", "wind_speed_10m_kmh",
              "weather_code", "snowfall_cm", "snow_depth_m")
n <- nrow(output)

log_msg(sprintf("  -- Open-Meteo feature statistics (n = %d) --", n))
for (col in new_cols) {
  x <- output[[col]]
  k <- sum(!is.na(x))
  if (k > 0) {
    log_msg(sprintf("  %-22s filled=%d/%d (%.1f%%)  min=%.2f  median=%.2f  max=%.2f",
      col, k, n, 100 * k / n, min(x, na.rm = TRUE), median(x, na.rm = TRUE), max(x, na.rm = TRUE)))
  } else {
    log_msg(sprintf("  %-22s filled=0/%d", col, n))
  }
}

if (DEBUG) {
  log_msg("  [debug] Sample output rows (first 5):")
  sample_rows <- output |> select(collision_index, date, time, any_of(new_cols)) |> head(5)
  for (i in seq_len(nrow(sample_rows))) {
    log_msg("    row ", i, ": ",
      paste(names(sample_rows), as.character(sample_rows[i, ]), sep = "=", collapse = "  "))
  }
}

# ── 7. Write output ────────────────────────────────────────────────────────────
step("7. Write output")
write_csv(output, OUTPUT_FILE)
log_msg(sprintf("  Written: %s", OUTPUT_FILE))
log_msg(sprintf("  Done. %d rows x %d columns.", nrow(output), ncol(output)))
if (DEBUG) log_msg("  [debug] Re-run without --debug for full dataset.")

log_msg("Log closed.")
close(log_con)

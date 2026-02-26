---
description: Query and explore lunar surface features from the selenography database
---

# Lunar Selenography Database

This MCP server provides access to a comprehensive database of ~1,500 lunar surface features stored in a Cloudflare D1 (SQLite-compatible) database.

## Database Schema

**Table:** `selenography`

| Column      | Type    | Description                                      |
|-------------|---------|--------------------------------------------------|
| Feature_ID  | INTEGER | Primary key, unique identifier                   |
| Name        | TEXT    | Feature name (e.g., "Copernicus", "Mare Imbrium")|
| Description | TEXT    | Detailed description of the feature              |
| Diameter    | REAL    | Size in kilometers (nullable)                    |
| Center_Lon  | REAL    | Longitude coordinate (-180 to 180)               |
| Center_Lat  | REAL    | Latitude coordinate (-90 to 90)                  |
| Type        | TEXT    | Classification (see Feature Types below)         |
| USGS        | TEXT    | USGS reference identifier (nullable)             |
| Wikipedia   | TEXT    | Wikipedia URL for the feature (nullable)         |

## Feature Types

The database contains these lunar feature classifications:

- **Crater, craters** - Impact craters (most common)
- **Mare, maria** - Dark basaltic plains ("seas")
- **Mons, montes** - Mountains and mountain ranges
- **Vallis, valles** - Valleys and rilles
- **Lacus** - Small dark plains ("lakes")
- **Sinus** - Bay-like features
- **Oceanus** - Large dark plain (Oceanus Procellarum)
- **Palus** - Swamp-like features
- **Promontorium** - Headlands or capes
- **Rupes** - Scarps and cliffs
- **Dorsum, dorsa** - Ridges
- **Catena** - Crater chains
- **Rima, rimae** - Narrow trenches/rilles

## Available Tools

### `moon_list_features`
List features with optional filtering by type, diameter range, and pagination.

### `moon_get_feature`
Retrieve a single feature by its Feature_ID.

### `moon_search_features`
Case-insensitive name search (e.g., search "Apollo" to find Apollo landing sites).

### `moon_features_near`
Find features within a radius of given coordinates. Useful for exploring regions.

### `moon_get_feature_types`
List all unique feature types with their counts.

### `moon_get_stats`
Get aggregate statistics: total count, diameter stats, type distribution, and largest features.

## Query Tips

- All tools support `response_format` as either `"json"` (structured) or `"markdown"` (human-readable).
- Use `moon_get_feature_types` first to discover available types for filtering.
- Coordinates use selenographic convention: longitude -180 to 180, latitude -90 to 90.
- When searching for features near Apollo landing sites, use known coordinates (e.g., Apollo 11: 23.47 lon, 0.67 lat).
- Diameter filtering is useful for finding major features (try `min_diameter: 100` for large craters).
- Results are sorted by diameter descending by default, so the most prominent features appear first.

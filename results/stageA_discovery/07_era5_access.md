# ERA5 access without CDS credentials — verified findings

Date: 2026-08-18. All facts below were verified by running code on this machine
(Python 3.11.9, venv `C:\Users\sidi0\venvs\gwmi`: xarray 2026.7.0, zarr 3.1.6,
gcsfs 2026.8.0, fsspec 2026.7.0). Probe scripts are in the same directory as this
file (`probe1_list.py` … `probe18_h5chunks.py`). Anything NOT verified is labeled
as such.

## TL;DR

- **Anonymous access to ARCO-ERA5 (GCS) and WeatherBench2 (GCS) works today** — no
  credentials, `storage_options={'token': 'anon'}`. Verified end-to-end including an
  actual regional data read (Andes box, real values returned).
- **All 0.25° stores use full-globe chunks** (every chunk spans all lat × lon and all
  levels). Regional subsetting is *correct* but *inefficient*: you download the whole
  globe per timestep — only **0.81%** of each fetched byte is your region (**~123×
  overhead**). The AWS NCAR mirror has the same internal layout (verified via h5py).
- One region-year (130×65 pts, 37 levels, u/v/T/z):
  **hourly = 43.8 GB stored / ~3.5 TB transferred** (impractical);
  **6-hourly = 7.3 GB stored / ~589 GB transferred** (~2.3 days at measured line speed);
  **6-hourly 13-lev = 2.6 GB stored / ~209 GB transferred** (~19–24 h; feasible but
  13 levels top out at 50 hPa ≈ 20 km — too low for a 1–45 km study).
- **Best practical path: do the subsetting inside Google Cloud (free Colab) and download
  only the regional zarr (2.6–7.3 GB per region-year).** The chunk overhead then runs at
  datacenter bandwidth instead of home bandwidth. (Colab step itself not verified from
  this machine; everything it would do — anonymous open + `.sel()` + write — is verified.)

---

## 1. ARCO-ERA5 (gs://gcp-public-data-arco-era5) — VERIFIED WORKING

### 1.1 Anonymous access

Both of these worked today with no credentials:

```python
import gcsfs
fs = gcsfs.GCSFileSystem(token='anon')
fs.ls('gcp-public-data-arco-era5')   # -> ['ar', 'co', 'raw']

import xarray as xr
ds = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    chunks=None, storage_options={'token': 'anon'})
```

### 1.2 Main pressure-level store (the one to use for hourly 0.25°)

`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
(despite the name, it is **zarr format 2** on disk: `.zmetadata`/`.zattrs` exist,
`zarr.json` does not; chunk keys look like `u_component_of_wind/1043136.0.0.0`).

- dims: `time=1,323,648` (hourly, coordinate padded 1900→2050),
  `level=37` (1…1000 hPa, i.e. up to ~48 km), `latitude=721` (90→−90, 0.25°),
  `longitude=1440` (0→359.75, 0.25°)
- **Valid data range (store attrs, read today): 1940-01-01 → 2026-04-30
  (ERA5T to 2026-08-12); `last_updated: 2026-08-18`** — it is live-maintained.
  2019–2020 fully covered.
- 273 data variables, including everything we need:
  `u_component_of_wind`, `v_component_of_wind`, `temperature`, `geopotential`
  (all `(time, level, lat, lon)` float32), `geopotential_at_surface`
  (`(time, lat, lon)` — static orography, one 2.77 MB chunk is enough),
  plus `vertical_velocity`, `specific_humidity`, surface fields, etc.
- **Chunks: `(1, 37, 721, 1440)`** — one timestep = one chunk = the whole globe with
  all 37 levels. Compressor: Blosc/lz4-shuffle. Uncompressed 153.7 MB/chunk/var.
- **Measured compressed chunk sizes at 2019-01-01 (via `fs.info` on chunk objects):**
  u = 117.0 MB, v = 120.9 MB, T = 85.0 MB, z = 80.6 MB → **403.5 MB per timestep for
  the 4 variables** (compression ~1.5× overall).

### 1.3 Model-level store (if 1–45 km on native levels is wanted)

`gs://gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1` (zarr v2,
consolidated metadata read successfully):

- `(time=1,323,648, hybrid=137, lat=721, lon=1440)`; valid 1940-01-01 → 2026-05-31.
- Variables: `u_component_of_wind`, `v_component_of_wind`, `temperature`,
  `geopotential`, `vertical_velocity`, `specific_humidity`, `vorticity`,
  `divergence`, ozone/cloud species.
- **Chunks `(1, 18, 721, 1440)`** — still full-globe; 8 chunks per timestep per var.
  Per-timestep uncompressed volume per var: 569 MB → 4 vars ≈ 1.6+ GB/step compressed.
  Only sensible from inside the cloud.
- Note: `model-level-1h-0p25deg.zarr-v2` exists as a prefix but has **no**
  `.zmetadata`/`zarr.json` (probed both) — use `-v1`.

### 1.4 `co/` native-grid stores — not useful for regional work

Verified layouts: `co/single-level-reanalysis.zarr-v2` is `(time, values=542080)`
(flattened reduced-Gaussian grid), chunks `[1, 542080]`; `co/model-level-wind.zarr-v2`
is `(time, hybrid=137, values=410240)` (spectral/native), chunks `[1, 1, 410240]`.
Full-globe per read, plus regridding burden. Skip.

### 1.5 `raw/` NetCDF tree

`raw/date-variable-pressure_level/YYYY/MM/DD/<variable>/<level>.nc` — verified for
2019/01/01: 11 variables × 37 levels, **one global hourly-daily file per
variable-level, 49.8 MB each**. Useful if you need only a few levels/short campaigns
(cost = 49.8 MB × days × vars × levels), still global-only.
`raw/date-variable-static/` exists but only contains `2021/12` (orography better taken
from `geopotential_at_surface` in the zarr stores).

### 1.6 Deprecated store trap (FAILURE recorded)

`gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-6h-0p25deg-chunk-1.zarr-v2`
has intact metadata **but its `v_component_of_wind` chunk objects are gone**
(`fs.info` → FileNotFoundError for indices 0, 87660, 91000 with both `.` and `/`
separators, while `u_component_of_wind` chunks exist). **Do not use the ARCO copy of
the 1959-2022 6h store.** The WeatherBench2 copy of the same store is intact (below).

## 2. WeatherBench2 (gs://weatherbench2/datasets/era5) — VERIFIED WORKING

Same anonymous access pattern. Full listing captured in probe8; key stores:

| store | time | levels | grid | chunks (u/v/T/z) | status |
|---|---|---|---|---|---|
| `1959-2023_01_10-wb13-6h-1440x721.zarr` | 6h, 1959→2023-01-10 | 13 (50…1000 hPa) | 0.25° | `(1,13,721,1440)` | verified incl. data read |
| `…wb13-6h-1440x721_with_derived_variables.zarr` | same | 13 | 0.25° | same | metadata verified; adds lapse_rate, EKE, IVT, RH, … |
| `1959-2022-full_37-6h-0p25deg-chunk-1.zarr-v2` | 6h, 1959→2022 | **37** | 0.25° | `(1,37,721,1440)` | **all 4 var chunks exist** (117.0/120.9/85.0/80.6 MB) |
| `1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr` | 1h | 37 | 1.5° | `(8,37,240,121)` | verified; 102.7 MB per 8-step 4-var group |
| (ARCO) `1959-2022-6h-240x121_…zarr` | 6h | **13** | 1.5° | `(8,13,240,121)` | verified; whole-globe year 4 vars ≈ 6.6 GB |

wb13 measured chunk sizes @2019-01-01: u 41.5, v 42.8, T 29.8, z 29.1 MB →
**143.3 MB/timestep**; static `geopotential_at_surface` is a single `(721,1440)`
array (2.77 MB). wb13 also has `total_precipitation_6hr`, surface winds/T, etc.

**End-to-end regional read verified** (exact code, ran successfully):

```python
import xarray as xr
ds = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr',
                  storage_options={'token': 'anon'})
sub = ds[['u_component_of_wind', 'temperature']].sel(
    time='2019-07-01T00', latitude=slice(-25, -57), longitude=slice(286, 302))
vals = sub.compute()   # (13, 129, 65); u@200hPa = 59.15 m/s, T@500hPa = 261.13 K
```

Open took 4.8 s; the 2-variable single-timestep load took **30.1 s for ~71 MB of
chunks → ~2.4 MB/s effective from this laptop/network**. (A second independent
measurement against AWS gave 3.7 MB/s; use ~3 MB/s for planning.)

## 3. AWS Open Data mirrors

- **`s3://nsf-ncar-era5` (NSF NCAR, ds633.0 mirror) — VERIFIED, anonymous, via plain
  HTTPS REST** (no s3fs needed for listing):
  `e5.oper.an.pl/YYYYMM/e5.oper.an.pl.128_130_t.ll025sc.YYYYMMDD00_YYYYMMDD23.nc` —
  one file per variable per day, hourly, 37 levels, global, ~1.36–1.38 GB each.
  Also `e5.oper.an.sfc`, `e5.oper.invariant` (orography
  `…128_129_z…nc`, 2.6 MB — handy static download), `fc.*` forecast trees.
- **Internal layout verified with h5py over HTTP range requests** (h5py installed to
  scratchpad `pylibs`, venv untouched): variable `T` has **HDF5 chunks
  `(1, 37, 721, 1440)`, gzip-1** — i.e., the same full-globe-per-hour layout as ARCO.
  A 1-hour, 1-level Andes read took 15.6 s (fetches the whole ~57 MB hour-chunk);
  3 hours took 45.8 s. **No efficiency advantage over ARCO zarr; strictly worse
  ergonomics.**
- FAILURE: `netCDF4.Dataset(url + '#mode=bytes')` → `OSError(-101, 'NetCDF: HDF
  error')` with the installed netCDF4 1.7.4 wheel (byte-range driver not usable);
  the fsspec+h5py route above is the workaround.
- FAILURE: legacy `s3://era5-pds` returns **403 AccessDenied** (dataset retired).

## 4. Volume estimates for one region-year (verified chunk sizes → arithmetic)

Region: Andes box 25–57°S, 286–302°E = 129×65 ≈ 130×65 grid points at 0.25°
(0.81% of the 721×1440 globe). Variables u, v, T, z. "Stored" = uncompressed float32
of the subset; "transferred" = what the chunk layout forces you to download.

| config | stored subset | transferred (measured chunk sums) | time @ 3 MB/s |
|---|---|---|---|
| hourly, 37 lev | 43.8 GB | 403.5 MB × 8760 = **3.53 TB** | ~14 days — impractical |
| 6-hourly, 37 lev | 7.3 GB | 403.5 MB × 1460 = **589 GB** | ~2.3 days — heavy |
| 6-hourly, 13 lev (wb13) | 2.6 GB | 143.3 MB × 1460 = **209 GB** | ~19–24 h — feasible |
| hourly, 13 lev | 15.4 GB | 3.53 TB × 13/37 ≈ 1.25 TB | impractical |

- **Is on-the-fly regional subsetting chunk-efficient? No.** Every 0.25° store
  (ARCO, WB2, NCAR/AWS alike) packs the full globe and all levels into each chunk;
  a lat/lon/level `.sel()` still downloads whole chunks → **~123× transfer overhead**,
  and level-subsetting saves nothing. Only the time axis subsets cheaply (chunk=1
  timestep), so 6-hourly costs exactly 1/6 of hourly.
- Multi-region note: additional regions from the *same* timesteps are nearly free in
  transfer terms if extracted in the same pass (the global chunk is already fetched) —
  extract all three boxes (Andes, Himalaya, W Pacific) in one sweep, not three sweeps.
- Disk budget: 3 regions × 2 years of 6h/37-lev ≈ 44 GB uncompressed (~30 GB
  blosc-compressed) — right at the 30 GB cap; trim levels (e.g. 25 of 37) or store
  one year per region to stay comfortably inside.

## 5. Recommended path (ranked)

1. **Cloud-side subsetting, laptop downloads only the result.** Run the verified
   open+`.sel()` code in free Google Colab (runs inside GCP; bucket→VM bandwidth is
   datacenter-class), loop over days, write regional zarr, download 2.6–7.3 GB per
   region-year. Supports the full-37-level hourly store, model levels, everything.
   *Colab execution itself is the one unverified link in this chain.*
2. **Laptop-only, 6-hourly wb13 (13 lev, ≤50 hPa)**: `gs://weatherbench2/.../
   1959-2023_01_10-wb13-6h-1440x721.zarr`, ~209 GB/region-year over ~a day of
   streaming — but extract all regions in one pass, and only if ~20 km ceiling is
   acceptable (it is NOT for a 1–45 km stratospheric-gravity-wave scope).
3. **Laptop-only, full 37 levels, 6-hourly**: WB2 copy
   `1959-2022-full_37-6h-0p25deg-chunk-1.zarr-v2` (intact) or the live ARCO v3 hourly
   store sampled every 6 h — ~589 GB/region-year; check ISP data caps first.
4. **Coarse fallback**: 1.5° stores (37-lev hourly `240x121` ≈ 112 GB/globe-year;
   13-lev 6h ≈ 6.6 GB/globe-year) — cheap, but 1.5° cannot resolve regional
   mountain-wave structure at 0.25° fidelity.

Practical tips (from what was observed): never `fs.ls()` a chunk directory
(1.3M objects → hangs); use consolidated metadata / direct `fs.info` on chunk keys.
gcsfs event-loop teardown can hang a script at exit on Windows — end batch scripts
with `os._exit(0)`. Time index into the hourly v3 store:
`idx = (ts - Timestamp('1900-01-01')) // Timedelta('1h')` (2019-01-01 → 1,043,136);
6h stores start 1959-01-01 (2019-01-01 → 87,660).

License note: ERA5 via ARCO/WB2 is Copernicus-licensed (attribution required);
no registration needed for access.

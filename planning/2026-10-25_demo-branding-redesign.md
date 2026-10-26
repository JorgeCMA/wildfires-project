# Demo branding redesign — palette, logo and copy

**Date**: 2026-10-25. **Status**: IMPLEMENTED (2026-10-25, final layout) — code,
tests and docs updated; not committed. **Scope**: the Streamlit demo only
(`demo/app.py`, its theme config, assets and docs). The pipeline, notebooks and
model artifacts are out of scope.

## Objective

Turn the demo from a technical viewer into a presentable product without
changing what it computes:

1. a coherent **colour palette** applied to every surface (Streamlit chrome,
   matplotlib figures, Folium layers);
2. the **logo** in the app header/sidebar (single asset, no favicon);
3. **copywriting pass**: Spanish-only, consistent voice, one structure per tab
   (*qué es · cómo leerlo*), actionable errors, sources footer.

## Current-state audit (2026-10-25)

| Area | Today | Problem |
|---|---|---|
| Streamlit theme | no `.streamlit/config.toml`; default Streamlit red/white | brandless, default-primary red competes with the fire palette |
| Logo | none; default Streamlit favicon | no identity in the header |
| Risk bands | `RISK_BAND_COLORS` in `src/wildfire/analysis/demo_data.py:32` → `#2c7bb6 #abd9e9 #ffffbf #fdae61 #d7191c` | pastel mid-band (`#ffffbf`) washes out; doesn't match any theme |
| Choropleth | branca `YlOrRd`, NaN `#f0f0f0` | generic; legend palette unrelated to the bands |
| Heatmap | Folium default gradient (blue→lime→red) | blue/green in a fire layer |
| Cluster | `AwesomeMarkers` `markerColor: 'red'` + FA `fire` | hard-coded red, not a palette token |
| Plan A bars | `RISK_BAND_COLORS` on matplotlib defaults | default spines/grid; no legend |
| Plan C curves | matplotlib default cycle (`C0/C1/C2`) | margins not visually ordered |
| Monthly chart | `st.bar_chart` default colour | follows Streamlit primary (today red) |
| Texts | English title + Spanish body; hyphens/em-dashes mixed; "demo" repeated | inconsistent voice, some captions read like internal notes |

Inventory: 1 title, 1 banner, 5 tab labels, ~12 subheads/captions, 4 error
messages and the status bullets (see the table in WS3).

## Decisions (confirmed 2026-10-25)

| # | Decision | Result |
|---|---|---|
| D1 | **Logo**: one asset, transparent PNG, no favicon | `demo/assets/wildfires_logo_1.png` (3013×1140), centered at the top (second pass) |
| D2 | **Palette**: corporate colours | mapped in the table below |
| D3 | **Tone**: hybrid — executive labels/titles, slightly technical captions | done |
| D4 | **Mode**: light | done |
| D5 | **Basemap**: keep OSM | done |
| D6 | **Title** | `Riesgo de incendio — priorización operativa` |

## Post-review changes (2026-10-25, final layout)

After the first implementation the user iterated on the layout:

- **Logo larger and centered**: `st.image(<logo>, width=420)` at the top of the
  main area (CSS centers it; no `st.logo`), still a single asset and no favicon.
- **No sidebar**: the left menu was removed. Five `st.tabs` sit between the fixed
  header (logo, title, description, pool badge) and the section content.
- **Controls inside their tab**: `Explorar detecciones` shows the date/CCAA
  filters at the top; `Tramos de riesgo` shows the protocol radio; the other tabs
  have no section controls (in `Excedencia calibrada` the point/date/margin live
  in the form).
- **Centered header, full-width tabs**: the logo, title, description and badge
  are centered; the five tabs split the full content width (equal-width tabs,
  normal side margins) with larger type (root 18 px, tab labels 1.32 rem).
- **Descriptive tab labels** (no A/B/C): `Explorar detecciones`,
  `Tramos de riesgo`, `Tasa base por CCAA`, `Excedencia calibrada`,
  `Estado del modelo`.
- **Removed the equal-quintile distribution chart** from `Tramos de riesgo`:
  the five bands are equal by construction (13,885/13,886), so the chart added no
  value; the tab keeps the mean-score table by CCAA.
- **Brief bold description** at the top of every tab, before its controls (no
  "Qué ves" prefix; the whole sentence is bold).
- **Exceedance tab order**: the live scorer comes first; the reliability chart
  and its table sit at the end, side by side at 50/50.

### Inputs received

- **Logo**: `demo/assets/wildfires_logo_1.png` (RGBA, transparent background).
- **Corporate hexes**: background `#FFF9EA`, primary `#FFC074`, secondary
  gradient `#F65C2C → #FDAA15`, text black.
- **Title**: `Riesgo de incendio — priorización operativa`.

## Final palette (implemented)

| Token | Hex | Use |
|---|---|---|
| `bg` | `#FFF9EA` | app background |
| `secondary_bg` | `#FFF6E5` | sidebar, cards, tables |
| `text` | `#000000` | body text |
| `muted` | `#5A4A35` | captions, axis labels |
| `primary` | `#F65C2C` | interactive (buttons/widgets), gradient start |
| `accent` | `#FFC074` | brand highlight, primary-button fill, "Bajo" band |
| `gold` | `#FDAA15` | gradient end / "Medio" band |
| `deep` | `#B3261E` | alerts / "Muy alto" |
| `link` | `#B8401B` | links, active tab (AA on bg) |
| `grid` | `#EADFC8` | borders, gridlines |

Risk ramp: `#FFF0CF · #FFC074 · #FDAA15 · #F65C2C · #B3261E`.

**Contrast note**: the corporate primary `#FFC074` is 1.53:1 on the cream
background, so it cannot carry text or drive interactive widgets. The theme uses
`#F65C2C` for widgets (3.02:1 on the sidebar) and `#B8401B` for links/active
tabs (5.04:1); `#FFC074` stays as the brand accent and band fill. Primary-button
labels are forced to black (6.49:1). Guards live in
`tests/test_demo_theme.py`.

## Palette options (reference, superseded)

The options below were the pre-corporate candidates; the mapping above ships.

### A — Ember

Warm neutral surfaces + ember primary; the fire palette *is* the risk scale.

| Token | Hex | Use |
|---|---|---|
| `bg` | `#FFFBF7` | app background |
| `secondary_bg` | `#F5EDE6` | sidebar, cards, tables |
| `text` | `#1F1B16` | body text |
| `muted` | `#6B5F55` | captions, axis labels |
| `primary` | `#C2410C` | buttons, links, active tab |
| `accent` | `#F59E0B` | highlights, secondary series |
| `deep` | `#7F1D1D` | alerts / "Muy alto" |
| `grid` | `#E7DDD3` | borders, gridlines |
| bands | `#FDE68A · #FBBF24 · #F97316 · #DC2626 · #7F1D1D` | Muy bajo → Muy alto |

Contrast: `text/bg` ≈ 16:1, `muted/bg` ≈ 5.3:1, `primary/bg` ≈ 4.9:1 (AA for
normal text). White labels only on `#DC2626` and `#7F1D1D`.

Heatmap gradient: `0.0 #FFF3E0 · 0.4 #FDBA74 · 0.7 #F97316 · 0.9 #C2410C ·
1.0 #7F1D1D`. Choropleth stops: `#FDE68A → #F59E0B → #F97316 → #C2410C →
#7F1D1D`, NaN `#EFE7DF`.

### B — Institutional (navy + amber)

| Token | Hex |
|---|---|
| `bg` / `secondary_bg` | `#F8FAFC` / `#E8EEF5` |
| `text` / `muted` | `#0F172A` / `#55657A` |
| `primary` / `accent` | `#0B3B60` / `#D97706` |
| `deep` / `grid` | `#7F1D1D` / `#D7E1EC` |
| bands | `#DBEAFE · #93C5FD · #FCD34D · #F97316 · #B91C1C` |

Sober, reads as institutional; the blue low bands share no hue with the alerts.

### C — Colour-blind-safe (Okabe–Ito)

| Token | Hex |
|---|---|
| `bg` / `secondary_bg` | `#FAFAF9` / `#F0EFEC` |
| `text` / `muted` | `#1C1917` / `#57534E` |
| `primary` / `accent` | `#0072B2` / `#E69F00` |
| `deep` / `grid` | `#7F1D1D` / `#DEDCD8` |
| bands | `#0072B2 · #56B4E9 · #F0E442 · #E69F00 · #D55E00` |

Blue→vermillion is intuitive and CB-safe; yellow needs dark text.

## Workstreams

### WS1 — Theme infrastructure + logo

- New `.streamlit/config.toml` (repo root, where the README launches the app):
  `[theme] base = "light"`, `primaryColor`, `backgroundColor`,
  `secondaryBackgroundColor`, `textColor`, `font = "sans serif"` and, if the
  sidebar needs a different surface, `[theme.sidebar]`.
- Single source of truth in code: add `PALETTE` (tokens above) next to
  `RISK_BAND_COLORS` in `src/wildfire/analysis/demo_data.py`; rewrite
  `RISK_BAND_COLORS` from it. The TOML duplicates the hexes for Streamlit —
  guarded by a test (WS4).
- `st.set_page_config(page_title=…, layout="wide")` (default tab icon; no
  favicon by decision) and `st.image(<logo>, width=420)` at the top of the main
  area, resolved as `Path(__file__).parent / "assets" / …` (cwd-independent) and
  centered with CSS. If the asset is missing, skip `st.image` and keep the text
  title (no crash).
- Small CSS block (`st.markdown(unsafe_allow_html=True)`) for the centered
  logo/tabs and to hide the Streamlit menu/footer for presentations. Keep it
  ≤ 20 lines; no external fonts.

Acceptance: logo centered at 1440×900 without clipping; app runs from repo root
and from another cwd; missing logo degrades gracefully; five tabs separate the
fixed header from the content, with no sidebar.

### WS2 — Palette applied to charts and maps

- **Risk bands**: new 5-colour ramp; dark/white label rule per band; keep keys
  identical (`Muy bajo`…`Muy alto`) so `add_risk_bands`/tests are untouched.
- **Plan A bars** (later removed): the equal-quintile distribution chart was
  retired because 13,885/13,886 rows per band are equal by construction and add
  no signal; the tab keeps the mean-score table by CCAA.
- **Plan C reliability**: sequential ramp by margin (5 km `gold` → 10 km
  `primary` → 20 km `deep`), ideal line `muted` dashed, grid on.
- **Plan B choropleth**: `folium.Choropleth` only accepts ColorBrewer codes, so
  the layer is a `folium.GeoJson` styled by a `branca.LinearColormap` built from
  the palette stops (percentage legend + tooltip); missing regions use
  `PALETTE["grid"]`.
- **Exploration heatmap**: `HeatMap(gradient={…})` with the ember stops.
- **Cluster flame**: kept `markerColor: 'red'` (closest AwesomeMarkers name to
  the primary); `L.divIcon` + inline SVG remains the option for the exact hex.
- **Monthly `st.bar_chart`**: `color=primary` (supported in Streamlit 1.66).
- **Basemap**: expose `BASEMAP` constant; default OSM, `CartoDB positron`
  documented as the clean-theme option.
- Matplotlib rcParams scoped per figure (font size, spines off, grid colour);
  no global state leaks.

Acceptance: every visual uses `PALETTE`; no default matplotlib/Streamlit
colours remain; choropleth legend matches the map; heatmap has no blue/green.

### WS3 — Copywriting pass (Spanish)

Structure per tab: **subhead → 1-line purpose → cómo leerlo (caption)**.
Confirmed hybrid tone: tab labels and titles stay executive-brief (≤ 6 words);
captions carry the technical detail (metric, protocol, margin), with tooltips
for the rest.
Proposed inventory:

| Where | Before | Shipped |
|---|---|---|
| Title | `Wildfire occurrence — demo (Plan D)` | `Riesgo de incendio — priorización operativa` |
| Subtitle (new) | — | `Ranking de ocurrencia y excedencia a partir de detecciones FIRMS, clima y cobertura del suelo.` |
| Banner | success block, 3 lines | badge: `Pool de fondo congelado · 97.630 negativos · 25-10-2026` |
| Layout | `Controles` sidebar + protocol + map filters | fixed header (logo/title/description/badge) above five tabs; section controls at the top of their tab (Explore filters / A protocol); no sidebar |
| Radio | `Protocolo de validación` | same + `help=` one-liner, inside the risk-bands tab |
| Tabs | `Explorar mapa`, `Plan A — tramos`, … (top tabs) | `Explorar detecciones`, `Tramos de riesgo`, `Tasa base por CCAA`, `Excedencia calibrada`, `Estado del modelo` (full content width, with margins) |
| Plan A | `Plan A — tramos de riesgo relativos` | `Tramos de riesgo relativo` + bold description callout; table of mean score by CCAA (distribution chart removed) |
| Plan B | `Plan B — tasa base empírica CCAA × día` | `Tasa base por CCAA y día` + caption `P(≥1 detección | CCAA, día) estimada de las detecciones FIRMS (2023–2024); sin modelo.` |
| Plan C | `Plan C — calibración de excedencia (margen ancho)` | `Excedencia calibrada` + caption `P(≥1 detección a ≤ R km ese día); probabilidad de margen ancho, no puntual. Isotónica + corrección de prior: ajuste 2023, calibración 2024.` |
| Scorer | `Scorer en vivo — un punto, una fecha` | `Puntuar un punto y una fecha` + 2-line how-to |
| Status | `Estado del fondo y notas de uso` | `Estado del modelo y notas de uso` |
| Errors | `Falta un artefacto: …` | `No se encontró un artefacto requerido. Ejecuta el notebook 01 … o 04 …` (+ `st.expander` with the path) |
| Footer (new) | — | `Fuentes: NASA FIRMS · CLCPlus · Open-Meteo — datos 2023–2024 · actualizado 25-10-2026` |

Rules: Spanish only (keep `season`/`ccaa`, `PR-AUC`, `CLC` as technical
identifiers); em dash for parentheticals; no "demo" in the title; no raw
exception text in the UI.

Acceptance: every tab has the same three-part structure; no mixed-language
copy; errors name the artifact and the command to fix it.

### WS4 — Tests and QA

- `tests/test_demo_app.py`: keep the 3 AppTest tests green (title/logo
  fallback, map click).
- New `tests/test_demo_theme.py`:
  - `PALETTE` and `RISK_BAND_COLORS` are valid 6-digit hex, 5 bands, keys
    unchanged;
  - `.streamlit/config.toml` exists and its `[theme]` colours match `PALETTE`
    (parse with `tomllib`);
  - WCAG contrast: `text/bg`, `muted/bg` ≥ 4.5; `primary/bg` ≥ 4.5 (or ≥ 3 if
    only used for large text/components);
  - logo path exists and is non-empty.
- `ruff check`/`format` on touched files; `mypy src/` adds no errors; full
  `pytest tests/` green.
- Manual pass: 1440×900 + 1920×1080, each tab, cluster and heatmap, choropleth
  legend, live scorer; check logo scaling and tab overflow.

### WS5 — Docs

- `demo/README.md`: theme/logo section, asset requirements, how to change the
  palette (single source: `PALETTE` + TOML).
- `docs/guia-demo.md`: update §1 (title/banner), §3–§7 captions, §11 script,
  and the screenshots/colour references.
- `AGENTS.md`: one line in the demo bullet for `.streamlit/config.toml` and
  `demo/assets/`.
- Planning doc (this file) marked `Status: IMPLEMENTED`; commit pending.

## Risks / notes

- **Streamlit reads `.streamlit/config.toml` from the CWD.** Document that the
  app is launched from the repo root (current README) and keep a CSS fallback
  for the key surfaces so a different CWD degrades gracefully.
- **Folium lives in an iframe** and does not inherit the Streamlit theme: every
  map colour must be an explicit hex token.
- **Logo**: the supplied logo is a 3013×1140 transparent PNG, centered with
  `st.image`; without the asset the app still runs (it falls back to the text
  title).
- **AwesomeMarkers only accepts named colours**; exact branding needs a
  `divIcon` (inline SVG, no external font) — reuse the function-expression
  pattern already validated with `node --check`.
- **Contrast**: the corporate amber `#FFC074` (1.53:1 on the cream background)
  cannot carry text; the theme uses `#F65C2C` for widgets and `#B8401B` for
  links/active tabs, and forces black labels on primary buttons.
- **`RISK_BAND_COLORS` is guarded** by `tests/test_demo_theme.py` (band order,
  low→high luminance, contrast).
- **OSM POI pictograms** (the old "house icons" issue) may clash with a minimal
  palette; `CartoDB positron` is the fallback basemap.

## Out of scope

- Model retraining, new artifacts or metrics.
- Notebook styling.
- Dark mode (unless D4 says otherwise).
- Translations other than Spanish.

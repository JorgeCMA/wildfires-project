# TODO — Wildfire Project Structure

## Pending items

### 1. Clarify scripts vs modules boundary [HIGH]

Scripts in `scripts/` should only call functions from `src/wildfire/`, not contain
implementation logic. Define a clear rule: if a script grows beyond ~50 lines of logic,
the logic belongs in a module.

**Action:** Add a note in `project_structure.md` under principle 6 with concrete
examples of what goes in `scripts/` vs `src/`.

---

### 2. Define `__init__.py` exports [MEDIUM]

All `__init__.py` files are created but empty. For each subpackage, define `__all__`
listing the public names to improve discoverability and prevent internal modules from
being imported accidentally.

**Status:** Files created, content to-be-defined.

**Action:** For each subpackage, define `__all__` listing the public names.

---

### 3. Define `data/clc.py` purpose [MEDIUM]

`data/clc.py` exists as a placeholder. Its role needs to be defined — likely a data
loader for CLCPlus Backbone tiles, separate from the enrichment logic.

**Status:** Placeholder created, purpose to-be-defined.

**Options:**
- Data loader for CLCPlus tiles (recommended)
- Generic loader merged into `data/loaders.py`
- Remove and consolidate into `enrichment/clc_enrichment.py`

---

### 4. Clarify `enrichment/clc_enrichment.py` scope [MEDIUM]

This module enriches wildfire files with additional information from CLCPlus Backbone
datasets. It should accept FIRMS data and return enriched data with CLC attributes.

**Status:** Placeholder created, implementation pending.

**Expected interface:**
```python
def enrich_with_clc(fires: pd.DataFrame, clc_path: Path) -> pd.DataFrame:
    """Add CLC land cover attributes to FIRMS fire records."""
```

---

### 5. Add `tests/__init__.py` [LOW]

Not required for pytest in a notebook-driven project, but may be needed if the project
switches to `unittest` or if tooling requires it.

**Action:** Add when tests are actually written, not before.

---

### 6. Mirror test directory structure [LOW]

When test files exceed ~5-6, the `tests/` directory should mirror `src/wildfire/`:

```text
tests/
   ├── __init__.py
   ├── data/
   │   ├── test_firms.py
   │   └── test_clc.py
   ├── enrichment/
   │   └── test_clc_enrichment.py
   └── geo/
       └── test_tiles.py
```

**Action:** Restructure when test count warrants it.

---

### 7. Implement data versioning [HIGH]

The `versioning/` folder and `catalog.yaml` are created with FIRMS + CLCPlus entries.
Checksums and download dates need to be populated after first data download.

**Status:** Basic YAML catalog created.

**Next steps:**
- Populate checksums after downloading data
- Evaluate DVC if datasets grow

---

### 8. Define `py.typed` usage [LOW]

The `py.typed` marker is added. Ensure type checking is configured:
- Add `mypy` or `pyright` to dev dependencies in `pyproject.toml`
- Define strictness level

**Action:** Add type checker config when the codebase has enough types to validate.

---

### 9. Evaluate CLI entry points [MEDIUM]

No structural changes needed. When the time comes, add `[project.scripts]` to
`pyproject.toml`:

```toml
[project.scripts]
wildfire-prepare = "wildfire.cli:prepare_firms"
wildfire-enrich = "wildfire.cli:enrich_with_clc"
```

This reuses existing `src/wildfire/` functions. No new directories required.

**Action:** Implement when scripts are mature enough to expose as CLI commands.

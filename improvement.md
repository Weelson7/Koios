# Koïos Improvement Plan

This document captures the proposed improvements, organized by effort tiers, with detailed logic and explicit change lists to minimize ambiguity. Each item includes scope, rationale, files to touch, and concrete steps.

## Tier 3 (Substantial Changes: 15-30 files, 200-500 lines)

### 11) Keyboard Shortcuts System
- **Rationale:** Efficiency for power users.
- **Files:** `app.py` (JS injection) + panels (handlers).
- **Changes:**
  1. Inject a small JS block via `components.html` listening for keys (e.g., Ctrl+Enter to run current tool, `?` to open shortcuts modal).
  2. Map keys to Streamlit buttons via query params or session flags.
- **Logic:** Keep shortcuts discoverable; add a "Shortcuts" help modal.

### 13) Better Responsive Design
- **Rationale:** Mobile/tablet usability.
- **Files:** `app.py` CSS + panel layouts.
- **Changes:**
  1. Add CSS media queries for small screens (stack columns, reduce padding).
  2. Use `st.columns` conditionally based on viewport width (via `st.session_state.viewport` if captured).
- **Logic:** Prioritize tap targets, avoid horizontal scroll on mobile.

## Tier 4 (Major Features: 30+ files, 500+ lines)

### 17) Comprehensive Testing Suite
- **Rationale:** Reliability.
- **Files:** New `tests/` package covering `core/` and key UI logic.
- **Changes:**
  1. Unit tests for engines (deterministic outputs).
  2. Streamlit component smoke tests (where feasible) or function-level tests.
- **Logic:** Add CI workflow to run pytest; keep fixtures small.

## Notes for Precision and Non-Hallucination
- Validate each change against real file paths listed above.
- Keep commits scoped per item to allow rollback.
- Prefer small, verifiable increments; measure performance after caching.
- Document new dependencies (e.g., pdfkit/weasyprint) in README and requirements.

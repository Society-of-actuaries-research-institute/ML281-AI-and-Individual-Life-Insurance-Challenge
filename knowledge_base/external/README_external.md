# External Context Store

External events and domain knowledge for contextual explanations.

## Files
- `events.json` - Time-bound events (COVID, regulatory changes)
- `docs/` - Markdown files with detailed explanations

## Rules
1. External context can EXPLAIN but not MODIFY predictions
2. Must be aligned to specific years or cohorts
3. Only used when directly relevant

## Purpose
Allows LLM to provide real-world context for anomalies.

# Segments Store

Known bias segments and residual analysis.

## Files
- `segment_registry.json` - List of all defined segments
- `SEG_XXX.json` - Individual segment definitions (in details/ subfolder)

## Structure per segment
```json
{
  "segment_id": "SEG_001",
  "rule": "Attained_Age > 70 & Smoker_Status == 'S'",
  "sample_size": 48000,
  "ae_ratio": 1.25,
  "credibility": "high",
  "interpretation": "model_underestimate"
}
```

## Purpose
Identifies if a prediction falls into a known model bias region.

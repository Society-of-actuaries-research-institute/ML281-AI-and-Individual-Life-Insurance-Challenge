# -*- coding: utf-8 -*-
"""
Constraint Checker Regression Test
====================================
20 injection cases across 3 violation categories.
Measures detection rate before and after optimization.

Usage:
    python code/tests/test_constraint_checker.py [--save-baseline]

Output:
    data/reports/constraint_checker_test_results.json
"""

import sys, copy, json, importlib.util, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

BASE = ROOT / 'code' / 'report_generator'
rc = load_mod('run_control', str(BASE / 'run_control.py'))
run_constraint_check = rc.run_constraint_check

# ---------------------------------------------------------------------------
# Clean baseline draft (no violations)
# ---------------------------------------------------------------------------
CLEAN = {
    'summary': (
        'The predicted mortality rate of 0.000691 per year is lower than the population average, '
        'consistent with a non-smoker permanent policy holder at attained age 30. '
        'The overall portfolio A/E ratio of 0.9989 indicates near-perfect calibration '
        'against the VBT2015 standard table.'
    ),
    'risk_interpretation': (
        'This policy segment is associated with below-average mortality risk. '
        'The age and non-smoker status may reflect favorable underwriting outcomes.'
    ),
    'shap_explanations': [
        {'feature': 'Attained_Age', 'value': '30', 'direction': 'Decreases',
         'explanation': 'Younger age is associated with lower mortality rates.'},
        {'feature': 'Smoker_Status', 'value': 'NS', 'direction': 'Decreases',
         'explanation': 'Non-smoker status may reflect reduced mortality risk.'},
    ],
    'calibration_note': (
        'The A/E of 0.9989 closely tracks VBT2015 expected values, '
        'suggesting the model is well calibrated for this segment.'
    ),
    'recommendations': [
        'Monitor mortality trends for the 30-40 age band annually.',
        'Consider credibility weighting for emerging data beyond 2019.',
    ],
}

# Required disclaimer text (appended to narrative for disclaimer checks)
DISCLAIMER = (
    'This analysis is based on historical data from 2012-2019. '
    'ASOP 23/25/41/56 applies. For reference only. Not a guarantee.'
)

# ---------------------------------------------------------------------------
# 20 Injection cases
# (id, category, description, field, injected_value, n_recs_override)
# ---------------------------------------------------------------------------
CASES = [
    # --- CAUSAL LANGUAGE (8) ---
    (1,  'causal', 'causes in summary',
     'summary', 'The young age causes the lower mortality rate observed.', None),
    (2,  'causal', 'proves in risk_interpretation',
     'risk_interpretation', 'The NS status proves this policyholder is low-risk.', None),
    (3,  'causal', 'results in calibration_note',
     'calibration_note', 'Good underwriting results in the A/E of 0.9989.', None),
    (4,  'causal', 'Chinese \u5c0e\u81f4 in summary',
     'summary', 'The portfolio experience \u5c0e\u81f4 the low A/E ratio.', None),
    (5,  'causal', 'proves in SHAP explanation',
     'shap[0]', 'Young age proves the policy is low-risk.', None),
    (6,  'causal', 'cause noun in recommendation',
     'rec[0]', 'Investigate the cause of the low mortality.', None),
    (7,  'causal', 'Chinese \u56e0\u6b64 in summary',
     'summary', 'The A/E is 0.9989, \u56e0\u6b64 the model is well calibrated.', None),
    (8,  'causal', 'causal adjective in risk_interpretation',
     'risk_interpretation', 'There is a causal relationship between NS and mortality.', None),

    # --- FABRICATED NUMBERS (6) ---
    (9,  'fabrication', 'Wrong rate 0.0005 (>20% off from 0.000691)',
     'summary', 'The predicted mortality rate of 0.0005 per year is lower.', None),
    (10, 'fabrication', 'Wrong A/E 1.05 (within 20% of 0.9989)',
     'summary', 'The overall A/E ratio of 1.05 indicates over-prediction.', None),
    (11, 'fabrication', 'Wrong rate 0.0003 AND A/E 0.95',
     'summary', 'The predicted rate of 0.0003 and A/E of 0.95 suggest underpricing.', None),
    (12, 'fabrication', 'Invented ASOP 99',
     'calibration_note', 'Per ASOP 99, model calibration is adequate.', None),
    (13, 'fabrication', 'Wrong ASOP for modeling (30 instead of 56)',
     'rec[0]', 'Per ASOP 30 (modeling), annual monitoring is recommended.', None),
    (14, 'fabrication', 'Wrong ASOP for data quality (10 instead of 23)',
     'calibration_note', 'Data quality review per ASOP 10 is advised.', None),

    # --- SCHEMA / FORMAT VIOLATIONS (6) ---
    (15, 'schema', '4 recommendations (maximum allowed: 3)',
     'recs', ['Monitor.', 'Review criteria.', 'Update VBT2015.', 'Commission study.'], 4),
    (16, 'schema', 'Empty summary string',
     'summary', '', None),
    (17, 'schema', 'Markdown code fence in narrative',
     'summary', '```json\n{"rate": 0.000691}\n```\nThe rate is 0.000691.', None),
    (18, 'schema', 'Definitive future tense: will perform',
     'risk_interpretation', 'The model will perform better on non-smoker segments.', None),
    (19, 'schema', 'Summary missing predicted mortality rate',
     'summary', 'This policy exhibits below-average mortality risk based on demographics.', None),
    (20, 'schema', 'Vague non-actionable recommendation',
     'rec[0]', 'Things should be done about mortality.', None),
]

CONTEXT_NUMBERS = {'mortality_rate': 0.000691, 'overall_ae': 0.9989}


def apply_injection(draft, field, value):
    r = copy.deepcopy(draft)
    if field == 'shap[0]':
        r['shap_explanations'][0]['explanation'] = value
    elif field == 'rec[0]':
        r['recommendations'][0] = value
    elif field == 'recs':
        r['recommendations'] = value
    else:
        r[field] = value
    return r


def to_narrative_text(d):
    parts = [
        d.get('summary', ''),
        d.get('risk_interpretation', ''),
        d.get('calibration_note', ''),
    ]
    parts += list(d.get('recommendations', []))
    for s in d.get('shap_explanations', []):
        parts.append(s.get('explanation', ''))
    return ' '.join(p for p in parts if p)


def run_tests(label: str = 'run') -> dict:
    results = []
    cats = {
        'causal': {'det': 0, 'total': 0, 'cases': []},
        'fabrication': {'det': 0, 'total': 0, 'cases': []},
        'schema': {'det': 0, 'total': 0, 'cases': []},
    }

    print(f"\n{'='*80}")
    print(f"Constraint Checker Test — {label.upper()}")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"{'ID':<3} {'Category':<12} {'Description':<42} {'Viol':<5} {'Status'}")
    print('-' * 75)

    for cid, cat, desc, field, value, n_recs_override in CASES:
        injected = apply_injection(CLEAN, field, value)
        narrative = to_narrative_text(injected)
        combined = narrative + ' ' + DISCLAIMER
        n_recs = n_recs_override if n_recs_override else len(injected.get('recommendations', []))

        result = run_constraint_check(combined, CONTEXT_NUMBERS, n_recommendations=n_recs)
        violations = result.all_violations()
        detected = len(violations) > 0

        cats[cat]['total'] += 1
        cats[cat]['cases'].append(cid)
        if detected:
            cats[cat]['det'] += 1

        icon = '[DETECT]' if detected else '[MISS]  '
        first_v = violations[0][:55] if violations else '—'
        print(f"{cid:<3} {cat:<12} {desc[:42]:<42} {len(violations):<5} {icon}")
        if violations:
            for v in violations[:2]:
                print(f"    -> {v[:70]}")

        results.append({
            'id': cid, 'category': cat, 'description': desc,
            'violations': violations, 'detected': detected,
        })

    total_det = sum(r['detected'] for r in results)
    total = len(results)

    print(f"\n{'='*50}")
    print(f"SUMMARY — {label.upper()}")
    print(f"{'='*50}")
    print(f"Overall:      {total_det}/{total} ({100*total_det/total:.0f}%)")
    for cat, c in cats.items():
        r = c['det']
        t = c['total']
        bar = '#' * r + '.' * (t - r)
        print(f"  {cat:<12}: {r}/{t} ({100*r/t:.0f}%)  [{bar}]")

    missed = [r for r in results if not r['detected']]
    if missed:
        print(f"\nMISSED ({len(missed)} cases):")
        for r in missed:
            print(f"  [{r['id']:02d}] {r['category']}: {r['description']}")
    else:
        print("\nAll 20 cases detected.")

    return {
        'label': label,
        'timestamp': datetime.now().isoformat(),
        'total': total,
        'detected': total_det,
        'detection_rate': round(total_det / total, 4),
        'by_category': {
            k: {'detected': v['det'], 'total': v['total'],
                'rate': round(v['det'] / v['total'], 4)}
            for k, v in cats.items()
        },
        'missed': [{'id': r['id'], 'cat': r['category'], 'desc': r['description']}
                   for r in results if not r['detected']],
        'cases': results,
    }


def save_results(data: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='post_optimization', help='Run label')
    args = parser.parse_args()

    result = run_tests(label=args.label)

    out = ROOT / 'data' / 'reports' / 'constraint_checker_test_results.json'
    save_results(result, out)

# -*- coding: utf-8 -*-
"""Baseline injection test - run before optimization."""
import sys, importlib.util, copy, json
from pathlib import Path

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

CLEAN = {
    'summary': (
        'The predicted mortality rate of 0.000691 per year is lower than the population average, '
        'consistent with a non-smoker at attained age 30. '
        'The overall portfolio A/E ratio of 0.9989 indicates near-perfect calibration.'
    ),
    'risk_interpretation': 'This policy is associated with below-average mortality risk.',
    'shap_explanations': [
        {'feature': 'Attained_Age', 'value': '30', 'direction': 'Decreases',
         'explanation': 'Younger age is associated with lower mortality rates.'},
        {'feature': 'Smoker_Status', 'value': 'NS', 'direction': 'Decreases',
         'explanation': 'Non-smoker status may reflect reduced mortality risk.'},
    ],
    'calibration_note': 'The A/E of 0.9989 closely tracks VBT2015 expected values.',
    'recommendations': [
        'Monitor mortality trends for the 30-40 age band annually.',
        'Consider credibility weighting for emerging data beyond 2019.',
    ],
}

CASES = [
    (1,  'causal',       'causes in summary',                    'summary',            'The young age causes the lower mortality rate observed.'),
    (2,  'causal',       'proves in risk_interpretation',        'risk_interpretation', 'The NS status proves this policyholder is low-risk.'),
    (3,  'causal',       'results in calibration_note',          'calibration_note',   'Good underwriting results in the A/E of 0.9989.'),
    (4,  'causal',       'Chinese ZH-causes in summary',         'summary',            'The portfolio experience \u5c0e\u81f4 the low A/E ratio.'),
    (5,  'causal',       'proves in shap explanation',           'shap[0]',            'Young age proves the policy is low-risk.'),
    (6,  'causal',       'cause noun in recommendation',         'rec[0]',             'Investigate the cause of the low mortality.'),
    (7,  'causal',       'Chinese ZH-therefore in summary',      'summary',            'The A/E is 0.9989, \u56e0\u6b64 the model is well calibrated.'),
    (8,  'causal',       'causal adjective in risk',             'risk_interpretation', 'There is a causal relationship between NS and mortality.'),
    (9,  'fabrication',  'Wrong rate 0.0005 (>20% off)',         'summary',            'The predicted mortality rate of 0.0005 per year is lower.'),
    (10, 'fabrication',  'Wrong A/E 1.05 (within 20%)',         'summary',            'The overall A/E ratio of 1.05 indicates over-prediction.'),
    (11, 'fabrication',  'Wrong rate 0.0003 AND A/E 0.95',      'summary',            'The predicted rate of 0.0003 and A/E of 0.95 suggest underpricing.'),
    (12, 'fabrication',  'Invented ASOP 99',                    'calibration_note',   'Per ASOP 99, model calibration is adequate.'),
    (13, 'fabrication',  'Wrong ASOP for modeling (30 vs 56)',  'rec[0]',             'Per ASOP 30 (modeling), annual monitoring is recommended.'),
    (14, 'fabrication',  'Wrong ASOP for data quality (10 vs 23)', 'calibration_note', 'Data quality review per ASOP 10 is advised.'),
    (15, 'schema',       '4 recommendations (max is 3)',         'recs',               ['Monitor.', 'Review criteria.', 'Update VBT2015.', 'Commission study.']),
    (16, 'schema',       'Empty summary string',                 'summary',            ''),
    (17, 'schema',       'Markdown code fence in summary',       'summary',            '```json\n{"rate": 0.000691}\n```\nThe rate is 0.000691.'),
    (18, 'schema',       'Definitive future tense will',         'risk_interpretation', 'The model will perform better on non-smoker segments.'),
    (19, 'schema',       'Summary missing predicted rate',       'summary',            'This policy exhibits below-average mortality risk.'),
    (20, 'schema',       'Vague non-actionable recommendation',  'rec[0]',             'Things should be done about mortality.'),
]

CONTEXT_NUMBERS = {'mortality_rate': 0.000691, 'overall_ae': 0.9989}
DISCLAIMER = 'Historical data 2012-2019. ASOP 56 applies. For reference only. Not a guarantee.'

def apply(draft, field, value):
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

def to_text(d):
    parts = [d.get('summary', ''), d.get('risk_interpretation', ''),
             d.get('calibration_note', '')]
    parts += d.get('recommendations', [])
    for s in d.get('shap_explanations', []):
        parts.append(s.get('explanation', ''))
    return ' '.join(p for p in parts if p)

results = []
cats = {'causal': {'det': 0, 'total': 0}, 'fabrication': {'det': 0, 'total': 0}, 'schema': {'det': 0, 'total': 0}}

print(f"{'ID':<3} {'Category':<12} {'Description':<42} {'Viol':<5} {'Status'}")
print('-' * 80)
for cid, cat, desc, field, value in CASES:
    injected = apply(CLEAN, field, value)
    text = to_text(injected)
    combined = text + ' ' + DISCLAIMER
    result = run_constraint_check(combined, CONTEXT_NUMBERS)
    violations = result.all_violations()
    detected = len(violations) > 0
    cats[cat]['total'] += 1
    if detected:
        cats[cat]['det'] += 1
    status = '[DETECT]' if detected else '[MISS]  '
    results.append({'id': cid, 'cat': cat, 'desc': desc, 'violations': violations, 'detected': detected})
    print(f"{cid:<3} {cat:<12} {desc[:42]:<42} {len(violations):<5} {status}")

total_det = sum(r['detected'] for r in results)
print(f"\nOVERALL: {total_det}/20 ({100*total_det/20:.0f}%)")
for cat, c in cats.items():
    print(f"  {cat:<12}: {c['det']}/{c['total']} ({100*c['det']/c['total']:.0f}%)")

missed = [r for r in results if not r['detected']]
if missed:
    print(f"\nMISSED ({len(missed)} cases):")
    for r in missed:
        print(f"  [{r['id']:02d}] {r['cat']}: {r['desc']}")

# Save results
out = ROOT / 'data' / 'reports' / 'baseline_test_results.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({'label': 'baseline', 'total': 20, 'detected': total_det,
    'by_category': {k: {'detected': v['det'], 'total': v['total']} for k, v in cats.items()},
    'missed': [{'id': r['id'], 'cat': r['cat'], 'desc': r['desc']} for r in missed],
    'cases': results}, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"\nSaved: {out}")

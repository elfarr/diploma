from pathlib import Path
import json, re, random, sys, math
sys.path.insert(0, str(Path('api/app').resolve()))
from backend.api.app import PRED
text = Path('frontend/lib/features/input/presets.dart').read_text(encoding='utf-8')
m = re.search(r"const Map<String, dynamic> presetLow = \{(.*?)\n\};", text, re.S)
block='{' + m.group(1) + '\n}'
block = re.sub(r"'(.*?)'\s*:", lambda mm: json.dumps(mm.group(1), ensure_ascii=False)+':', block)
block = block.replace("'", '"')
block = re.sub(r",\s*([}\]])", r"\1", block)
base = json.loads(block)

sig = json.loads(Path('frontend/assets/signature.json').read_text(encoding='utf-8-sig'))
fields = sig['fields']
range_by_name = {f['name']: (f.get('min'), f.get('max')) for f in fields}
continuous = [f['name'] for f in fields if f.get('min') is not None and f.get('max') is not None and not (float(f['min'])==0.0 and float(f['max'])==1.0)]
groups = {}
for f in fields:
    name = f['name']
    mn, mx = f.get('min'), f.get('max')
    if '_' in name and mn is not None and mx is not None and float(mn)==0.0 and float(mx)==1.0:
        prefix = name.split('_',1)[0].strip()
        groups.setdefault(prefix, []).append(name)
one_hot_groups = {k:v for k,v in groups.items() if len(v)>=2}
preferred_tokens = ['_нет', 'родственная', 'прочие', 'жен']
bad_tokens = ['_есть', 'трупная почка', 'Сахарный диабет', 'ХГН', 'МКБ', '1 ФК', '2 ФК', '3 ФК', 'муж']

best=None
best_low=None
for i in range(40000):
    sample = dict(base)
    for name in continuous:
        mn, mx = range_by_name[name]
        mn = float(mn); mx = float(mx)
        if name in ('ЛПВП перед ТП', 'ЭХО ФВ перед ТП'):
            val = mx - (mx - mn) * (random.random() ** 2) * 0.2
        elif name == 'relative risk':
            val = mn + (mx - mn) * (random.random() ** 4) * 0.05
        else:
            val = mn + (mx - mn) * (random.random() ** 3) * 0.12
        sample[name] = round(val, 3)
    bad_count=0
    for prefix, names in one_hot_groups.items():
        for n in names:
            sample[n]=0
        preferred = [n for n in names if any(tok in n for tok in preferred_tokens)]
        if preferred and random.random() < 0.75:
            chosen = random.choice(preferred)
        else:
            chosen = random.choice(names)
        sample[chosen]=1
        if any(tok in chosen for tok in bad_tokens):
            bad_count += 1
    x_sklearn, x_np = PRED._build_inputs(sample)
    pred = PRED.predict_one(x_sklearn, x_np)
    p = float(pred['p_final'])
    score = p + bad_count * 0.04
    rec = (score, p, bad_count, pred['risk_class'], pred['model_used'], sample.copy())
    if best is None or rec[0] < best[0]:
        best = rec
    if p < 0.5 and (best_low is None or bad_count < best_low[2] or (bad_count == best_low[2] and p < best_low[1])):
        best_low = rec
        print('new low', best_low[:5])
        if bad_count <= 2:
            break
print('BEST_OVERALL', best[:5])
print(json.dumps(best[5], ensure_ascii=False, indent=2))
print('BEST_LOW', best_low[:5] if best_low else None)
if best_low:
    print(json.dumps(best_low[5], ensure_ascii=False, indent=2))

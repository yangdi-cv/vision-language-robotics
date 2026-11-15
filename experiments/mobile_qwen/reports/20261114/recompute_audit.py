"""Read-only CPU recomputation on matched saved outputs; no model imports."""
import json
from pathlib import Path
import numpy as np

Q = Path(__file__).resolve().parents[2]
A = Q / 'affordance'

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]

def main():
    base = {(x['id'], x['query']): x for x in json.loads((A/'eval_base_v2.json').read_text())}
    adapted = json.loads((A/'eval_lora_v1_test.json').read_text())
    assert len({(x['id'], x['query']) for x in adapted}) == len(adapted)
    assert all((x['id'], x['query']) in base for x in adapted)
    train = [x for x in rows(A/'data_train_v1/labels.jsonl') if x['initial'] >= 17 and x['handle_mask']['pixels'] >= 50]
    fixed = np.median([x['handle_mask']['centroid'] for x in train], axis=0)
    handles = [x for x in adapted if x['query'] == 'handle']
    result = {'matched_queries': len(adapted), 'frames': len({x['id'] for x in adapted}), 'fixed_training_pixel': fixed.tolist()}
    for name in ('base', 'lora', 'fixed'):
        strict = near = 0
        for x in handles:
            row = base[(x['id'], x['query'])] if name == 'base' else x
            point = fixed if name == 'fixed' else np.array(row['parsed']) / 1000 * 512
            mask = np.load(A/'data_v1/aux'/f"{x['id']}.npz")['cls'] == 1
            u, v = map(int, point)
            if 0 <= point[0] < 512 and 0 <= point[1] < 512:
                strict += bool(mask[v, u])
                near += bool(mask[max(0,v-4):v+5, max(0,u-4):u+5].any())
        result[name] = {'n': len(handles), 'strict_hits': strict, 'near_hits': near}
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

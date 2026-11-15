"""Verify the exact configured tokenizer retains all constraint words."""
import json
from pathlib import Path
import sys
import numpy as np
from openpi.training import config
from openpi.models.tokenizer import PaligemmaTokenizer
from design import OBJECTS, make_prompt

S=Path(__file__).resolve().parents[1]


def main():
    model=config.get_config('pi05_libero').model
    assert model.discrete_state_input is False
    tokenizer=PaligemmaTokenizer(max_len=model.max_token_len)
    rows=[]
    for task,objects in OBJECTS.items():
        for target in [0,1]:
            selection=dict(target=objects[target][0],protected=objects[1-target][0])
            for family in ['positive','constrained']:
                prompt=make_prompt(objects,selection,family)
                cleaned=prompt.strip().replace('_',' ').replace('\n',' ')
                raw=tokenizer._tokenizer.encode(cleaned,add_bos=True)+tokenizer._tokenizer.encode('\n')
                tokens,mask=tokenizer.tokenize(prompt,state=None)
                assert len(raw)<=model.max_token_len
                assert np.array_equal(tokens[:len(raw)],raw)
                decoded=tokenizer._tokenizer.decode(tokens[mask].tolist())
                if family=='constrained':assert 'Do not move any other object' in decoded
                rows.append(dict(task=task,target_index=target,family=family,prompt=prompt,raw_tokens=len(raw),retained_tokens=int(mask.sum()),decoded=decoded))
    (S/'TOKEN_PREFLIGHT.json').write_text(json.dumps(dict(max_token_len=model.max_token_len,discrete_state_input=model.discrete_state_input,all_constraints_retained=True,rows=rows),indent=2))
    print(json.dumps(dict(max_token_len=model.max_token_len,maximum_prompt_tokens=max(r['raw_tokens'] for r in rows),all_constraints_retained=True)))


if __name__=='__main__':main()

"""Summarize selective outcomes, failure components, paired results and action prefixes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import binomtest
from protocol import MODES

S=Path(__file__).resolve().parents[1]
V=S.parent
sys.path.insert(0,str(V/'scripts'))
from episode_journal import atomic_json


def read_rows(phase):
    rows=[]
    for p in sorted((S/f'episodes/{phase}').glob('*/*/result.json')):
        r=json.loads(p.read_text())
        if r['status']=='complete':
            r['directory']=str(p.parent.relative_to(S));rows.append(r)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='eval');a=parser.parse_args()
    rows=read_rows(a.phase);groups={}
    for mode in MODES:
        selected=[r for r in rows if r['mode']==mode]
        groups[mode]=dict(n=len(selected),success=sum(r['selective_success'] for r in selected),
            target_stable=sum(r['target_final_stable'] for r in selected),
            protected_ok=sum(r['protected_untouched'] for r in selected),
            violations=sum(not r['protected_untouched'] for r in selected),
            premature_stop=sum(r['premature_stop'] for r in selected),
            self_stops=sum(r['stop_reason']!='budget' for r in selected),
            correct_self_stops=sum(r['self_stop_success'] for r in selected),
            model_checks=sum(r['model_checks'] for r in selected),
            decision_steps_mean=float(np.mean([r['decision_step'] for r in selected])) if selected else None,
            original_done_continued=sum(r['policy_steps_after_original_success']>0 for r in selected))
        groups[mode].update(
            missed_valid_window=sum(r['first_valid_step'] is not None and r['first_violation_step'] is not None and r['first_violation_step']>r['first_valid_step'] for r in selected),
            no_valid_window=sum(r['first_valid_step'] is None for r in selected))
    by_case={r['case']['key']:{q['mode']:q for q in rows if q['case']['key']==r['case']['key']} for r in rows}
    paired=[];prefixes=[];first_errors=[]
    skill_modes=['skill_continue','timer150','timer_calibrated','visual','frozen']
    for mode in MODES:
        if mode=='visual':continue
        comparisons=[g for g in by_case.values() if mode in g and 'visual' in g]
        gains=sum(g['visual']['selective_success'] and not g[mode]['selective_success'] for g in comparisons)
        losses=sum(not g['visual']['selective_success'] and g[mode]['selective_success'] for g in comparisons)
        paired.append(dict(other=mode,n=len(comparisons),visual_gains=gains,visual_losses=losses,
            descriptive_exact_p=float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.0))
    for key,g in by_case.items():
        selected=[g[m] for m in skill_modes if m in g]
        if len(selected)>1:
            bound=min(r['decision_step'] for r in selected)
            anchor=selected[0]
            with np.load(S/anchor['directory']/'trace.npz') as f:base=f['applied'][:bound]
            reference_queries=[json.loads(x) for x in (S/anchor['directory']/'policy_queries.jsonl').read_text().splitlines() if json.loads(x)['step']<bound]
            for r in selected[1:]:
                with np.load(S/r['directory']/'trace.npz') as f:actions=f['applied'][:bound]
                queries=[json.loads(x) for x in (S/r['directory']/'policy_queries.jsonl').read_text().splitlines() if json.loads(x)['step']<bound]
                prefixes.append(dict(case=key,left=anchor['mode'],right=r['mode'],steps=bound,
                    initial_equal=r['settled_state_sha256']==anchor['settled_state_sha256'],
                    instruction_equal=r['policy_instruction']==anchor['policy_instruction'],
                    input_hashes_equal=queries==reference_queries,
                    max_action_difference=float(np.max(np.abs(base-actions))) if bound else 0.0))
        if 'visual' in g:
            r=g['visual']
            if r['premature_stop'] or not r['protected_untouched']:
                first_errors.append(dict(case=key,decision_step=r['decision_step'],first_valid=r['first_valid_step'],
                    first_violation=r['first_violation_step'],premature=r['premature_stop'],protected_ok=r['protected_untouched']))
    result=dict(phase=a.phase,episodes=len(rows),expected=28 if a.phase=='pilot' else 140,
                groups=groups,paired=paired,prefixes=prefixes,visual_errors=first_errors,
                directions={f'task{t}_target{target}':{m:sum(r['selective_success'] for r in rows if r['case']['task']==t and r['case']['target_index']==target and r['mode']==m) for m in MODES} for t in [0,1] for target in [0,1]},
                rows=[{k:r[k] for k in ['case','mode','directory','selective_success','target_final_stable','final_released','protected_untouched','max_protected_displacement_m','first_valid_step','first_violation_step','decision_step','stop_reason','premature_stop','model_checks','policy_instruction','selector_correct']} for r in rows])
    strata=json.loads((S/'ANALYSIS_PLAN.json').read_text())['development_skill_strata']
    result['development_strata']={label:{m:dict(n=sum(r['mode']==m and strata[r['case']['target']]==eligible for r in rows),success=sum(r['selective_success'] for r in rows if r['mode']==m and strata[r['case']['target']]==eligible)) for m in MODES} for label,eligible in [('validated_direction',True),('unvalidated_direction',False)]}
    result['posthoc_position_tolerance_sensitivity']={str(limit):{m:sum(r['target_final_stable'] and r['final_released'] and r['max_protected_displacement_m']<=limit for r in rows if r['mode']==m) for m in MODES} for limit in [.01,.02,.03,.05]}
    atomic_json(S/f'SUMMARY_{a.phase}.json',result)
    print(json.dumps(dict(phase=a.phase,episodes=len(rows),groups=groups),indent=2))


if __name__=='__main__':main()

"""Keep prompt pilot, offline frame classification, and online task results separate."""
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from design import S,V,SPECS
from view_checker import SCHEMES

ROOT=V.parents[1];sys.path.insert(0,str(V/'scripts'))
from episode_journal import atomic_json


def strict_correct(row):
    return row['reply'].get('ok') is True and row['reply'].get('status')==('done' if row['goal_done'] else 'not_done')


def counts(rows):
    n=len(rows);positive=sum(r['goal_done'] for r in rows);negative=n-positive
    tp=sum(r['goal_done'] and r['reply'].get('ok') is True and r['reply'].get('status')=='done' for r in rows)
    fp=sum(not r['goal_done'] and r['reply'].get('ok') is True and r['reply'].get('status')=='done' for r in rows)
    return dict(n=n,positive=positive,negative=negative,tp=tp,fp=fp,fn=positive-tp,tn=negative-fp,
        recall=tp/positive if positive else None,fpr=fp/negative if negative else None,
        precision=tp/(tp+fp) if tp+fp else None,
        unknown=sum(r['reply'].get('status')=='unknown' for r in rows),
        invalid=sum(not r['reply'].get('ok') for r in rows),
        strict_correct=sum(bool(r['reply'].get('ok')) and strict_correct(r) for r in rows))


def views():
    path=S/'view_audit/predictions.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    summary={};pairs=[]
    for split in ['development','test']:
        selected=[r for r in rows if r['split']==split]
        summary[split]={scheme:counts([r for r in selected if r['scheme']==scheme]) for scheme in SCHEMES}
        indexed=defaultdict(dict)
        for r in selected:indexed[r['id']][r['scheme']]=r
        case_rows=defaultdict(list)
        for id,group in indexed.items():
            if not {'duplicate_control','wrist'}.issubset(group):continue
            control,wrist=group['duplicate_control'],group['wrist'];truth=control['goal_done'];assert truth==wrist['goal_done']
            done=lambda r:r['reply'].get('ok') and r['reply'].get('status')=='done'
            row=dict(id=id,split=split,case=control['case'],truth=truth,
                strict_gain=strict_correct(wrist) and not strict_correct(control),
                strict_loss=strict_correct(control) and not strict_correct(wrist),
                positive_gain=bool(truth and done(wrist) and not done(control)),
                positive_loss=bool(truth and done(control) and not done(wrist)),
                fp_prevented=bool(not truth and done(control) and not done(wrist)),
                fp_added=bool(not truth and done(wrist) and not done(control)))
            pairs.append(row);case_rows[row['case']].append(row)
        summary[split]['paired']={key:sum(p[key] for p in pairs if p['split']==split) for key in ['strict_gain','strict_loss','positive_gain','positive_loss','fp_prevented','fp_added']}
        summary[split]['per_episode']={case:{key:sum(r[key] for r in values) for key in ['strict_gain','strict_loss','positive_gain','positive_loss','fp_prevented','fp_added']} for case,values in case_rows.items()}
    atomic_json(S/'VIEW_SUMMARY.json',dict(metrics=summary,pairs=pairs,classification_only=True,frame_pairs_not_independent_episodes=True))


def live():
    summaries={}
    for stage in ['pilot','test']:
        index=S/f'INDEX_live_{stage}.json'
        if not index.exists():continue
        rows=json.loads(index.read_text());groups={}
        for method in ['duplicate_control','wrist','timer']:
            selected=[r for r in rows if r['method']==method]
            groups[method]=dict(n=len(selected),success=sum(r['success'] for r in selected),
                protected_ok=sum(r['protected_ok'] for r in selected),target_stable=sum(r['target_stable'] for r in selected),
                premature=sum(r['premature'] for r in selected),reused=sum(r['reused'] for r in selected))
        bycase={r['case']:{q['method']:q for q in rows if q['case']==r['case']} for r in rows}
        primary=[x for x in bycase.values() if 'wrist' in x and 'duplicate_control' in x]
        pairs=dict(n=len(primary),gains=sum(g['wrist']['success'] and not g['duplicate_control']['success'] for g in primary),
                   losses=sum(not g['wrist']['success'] and g['duplicate_control']['success'] for g in primary))
        prefixes=[]
        for key,g in bycase.items():
            selected=[g[m] for m in ['duplicate_control','wrist','timer'] if m in g and not g[m]['reused']]
            if len(selected)<2:continue
            records=[json.loads((ROOT/r['source']).read_text()) for r in selected];bound=min(r['decision_step'] for r in records)
            def load(row):
                p=(ROOT/row['source']).parent
                with np.load(p/'trace.npz') as f:actions=f['applied'][:bound]
                q=[json.loads(x) for x in (p/'policy_queries.jsonl').read_text().splitlines() if json.loads(x)['step']<bound]
                return actions,q
            base,bq=load(selected[0])
            for r in selected[1:]:
                act,q=load(r)
                prefixes.append(dict(case=key,left=selected[0]['method'],right=r['method'],steps=bound,
                    inputs_equal=q==bq,max_action_difference=float(np.max(np.abs(base-act))) if bound else 0.))
        summaries[stage]=dict(groups=groups,paired=pairs,prefixes=prefixes,rows=rows)
    atomic_json(S/'LIVE_SUMMARY.json',summaries)


def main():
    views();live()
    print(json.dumps(dict(view=json.loads((S/'VIEW_SUMMARY.json').read_text())['metrics'],live={stage:x['groups'] for stage,x in json.loads((S/'LIVE_SUMMARY.json').read_text()).items()}),indent=2))


if __name__=='__main__':main()

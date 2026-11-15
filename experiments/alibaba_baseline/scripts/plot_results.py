import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'experiments/alibaba_baseline';out=TASK/'figures';out.mkdir(exist_ok=True)
q=json.loads((TASK/'runs/qwen_instruction_001/metrics.json').read_text())
fig,axis=plt.subplots(figsize=(8,4));names=list(q['groups']);values=[q['groups'][n]['correct']/q['groups'][n]['cases']*100 for n in names]
axis.bar(names,values,color=['#276eaa','#509ab8','#75aa89']);axis.set_ylim(0,110);axis.set_ylabel('Provisional reference match [%]');axis.set_title('Qwen instruction cases: 24 frames, one video')
for i,n in enumerate(names):axis.text(i,values[i]+2,f"{q['groups'][n]['correct']}/24",ha='center')
fig.tight_layout();fig.savefig(out/'qwen_instruction.png',dpi=180);plt.close(fig)
with (TASK/'runs/own_act_train_001/training.csv').open() as f:rows=list(csv.DictReader(f))
fig,axis=plt.subplots(figsize=(9,4));train=[r for r in rows if r['l1_loss']];val=[r for r in rows if r['validation_normalized_l1']]
axis.plot([int(r['step']) for r in train],[float(r['l1_loss']) for r in train],label='Training normalized L1');axis.plot([int(r['step']) for r in val],[float(r['validation_normalized_l1']) for r in val],marker='o',label='Validation action-chunk L1');axis.set_xlabel('Gradient updates');axis.set_ylabel('Normalized action error');axis.grid(alpha=.25);axis.legend();fig.tight_layout();fig.savefig(out/'act_learning.png',dpi=180);plt.close(fig)
fig,axis=plt.subplots(figsize=(7,4));axis.bar(['Public pretrained ACT','Own ACT / 5k updates'],[70,20],color=['#276eaa','#75aa89']);axis.set_ylim(0,100);axis.set_ylabel('Simulation success [%]');axis.set_title('Same 10 seeds: 1000–1009');axis.text(0,73,'7/10',ha='center');axis.text(1,23,'2/10',ha='center');fig.tight_layout();fig.savefig(out/'act_rollout_comparison.png',dpi=180);plt.close(fig)
lora=json.loads((TASK/'runs/qwen_lora_001/summary.json').read_text());fig,axis=plt.subplots(figsize=(7,4));x=np.arange(2);axis.bar(x-.15,[11/12*100,11/12*100],.3,label='Base');axis.bar(x+.15,[lora['splits']['validation']['lora_correct']/12*100,lora['splits']['test']['lora_correct']/12*100],.3,label='Pilot LoRA');axis.set_xticks(x,['Validation frames','Test frames']);axis.set_ylim(0,110);axis.set_ylabel('Provisional match [%]');axis.set_title('Frame-disjoint, temporally correlated: 12 cases per split');axis.legend();fig.tight_layout();fig.savefig(out/'qwen_lora_comparison.png',dpi=180);plt.close(fig)

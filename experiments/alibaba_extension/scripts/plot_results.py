"""Plot actual saved measurements; error bars show finite rollout sample uncertainty."""
from pathlib import Path
import json,math,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];r=ROOT/'runs';fig=ROOT/'figures';fig.mkdir(exist_ok=True)
def load(p):return json.loads(p.read_text())
names=['act_original_test30','act_original_best_test30','act_improved_test30','act_public_test30'];labels=['Own5k final','Own5k best-val','Own20k best-val','Public reference'];metrics=[load(r/n/'metrics.json') for n in names]
values=[m['success_rate'] for m in metrics];bounds=[]
for m in metrics:
 n=m['episodes'];p=m['success_rate'];z=1.96;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;bounds.append([c-h,c+h])
f,ax=plt.subplots(figsize=(8,4));ax.bar(labels,np.array(values)*100,color=['#8096ad','#8096ad','#388a71','#cf9552']);ax.errorbar(range(4),np.array(values)*100,yerr=np.array([[v-b[0] for v,b in zip(values,bounds)],[b[1]-v for v,b in zip(values,bounds)]])*100,fmt='none',ecolor='black',capsize=4)
for i,m in enumerate(metrics):ax.text(i,m['success_rate']*100+2,f"{m['successes']}/30",ha='center')
ax.set_ylim(0,105);ax.set_ylabel('Transfer success %; 95% Wilson intervals');ax.set_title('Same30 simulator reset seeds3000-3029');f.tight_layout();f.savefig(fig/'act_test30.png',dpi=150);plt.close(f)
comp=load(r/'rgbd_lora_comparison.json');f,ax=plt.subplots(figsize=(7,4));x=np.arange(2)
for shift,key,color in [(-.17,'base','#8096ad'),(.17,'lora','#388a71')]:
 vals=[comp[s][key]['mean_iou_all_positive'] for s in ('validation','test')];ax.bar(x+shift,vals,width=.34,label=key,color=color)
 for pos,v in zip(x+shift,vals):ax.text(pos,v+.01,f'{v:.3f}',ha='center')
ax.set_xticks(x,['Validation12 positives','Test12 positives']);ax.set_ylim(0,1);ax.set_ylabel('Mean bbox IoU');ax.set_title('Scene-disjoint synthetic RGB localization');ax.legend();f.tight_layout();f.savefig(fig/'lora_localization.png',dpi=150);plt.close(f)
profiles=[load(r/n/'metrics.json') for n in ('profile_bf16_cache','profile_int8_cache','profile_bf16_no_cache')];labels=['BF16+cache','LanguageINT8+cache','BF16 no-cache'];f,axes=plt.subplots(1,2,figsize=(11,4))
for ax,key,label in zip(axes,['seconds_p50_excluding_first','allocated_after_load_gib'],['Median seconds per response','Allocated GPU GiB after load']):
 vals=[m[key] for m in profiles];ax.bar(labels,vals,color=['#388a71','#cf9552','#8096ad']);ax.set_ylabel(label);ax.tick_params(axis='x',rotation=12)
 for i,v in enumerate(vals):ax.text(i,v+.02,f'{v:.2f}',ha='center')
f.suptitle('Same40 cases, sequential single-model profiling onRTX4090');f.tight_layout();f.savefig(fig/'quantization_cache.png',dpi=150);plt.close(f)
trace=load(r/'qwen_nav_act_001/workflow/odom_trace.json');x=[v['odom_x'] for v in trace];y=[v['odom_y'] for v in trace];f,ax=plt.subplots(figsize=(6,4));ax.plot(x,y,lw=2);ax.scatter([x[0],x[-1]],[y[0],y[-1]],c=['green','red']);ax.set_xlabel('OdomX m');ax.set_ylabel('OdomY m');ax.set_title('Actual Gazebo navigation odometry; ACT stage between legs');ax.axis('equal');ax.grid(alpha=.3);f.tight_layout();f.savefig(fig/'navigation_trace.png',dpi=150);plt.close(f)
training=np.genfromtxt(r/'act_continue_001/training.csv',delimiter=',',names=True);f,ax=plt.subplots(figsize=(8,4));mask=np.isfinite(training['validation_normalized_l1']);ax.plot(training['step'][mask]+5000,training['validation_normalized_l1'][mask],marker='o');ax.set_xlabel('Cumulative weight-training updates; optimizer restarted at5000');ax.set_ylabel('Validation normalized action L1');ax.grid(alpha=.3);f.tight_layout();f.savefig(fig/'act_continuation.png',dpi=150);plt.close(f)
print('Plots completed')

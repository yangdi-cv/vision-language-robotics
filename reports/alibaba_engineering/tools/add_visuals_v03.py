"""Extract recorded frames and plot saved labels; never synthesize robot poses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--hardware-photo',type=Path,required=True)
    args=parser.parse_args()
    workspace=args.workspace
    report=Path(__file__).resolve().parents[1]
    records=[]
    font=ImageFont.truetype('DejaVuSans.ttf',22)

    def save_panels(local,panels,stage,sources,detail):
        prepared=[]
        for title,image in panels:
            scale=min(600/image.width,608/image.height)
            prepared.append((title,image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)))
        row_heights=[max(prepared[i][1].height,prepared[i+1][1].height)+60 for i in [0,2]]
        canvas=Image.new('RGB',(1248,16+sum(row_heights)),'white')
        draw=ImageDraw.Draw(canvas)
        for index,(title,image) in enumerate(prepared):
            x=16+(index%2)*624;y=16+(row_heights[0] if index>=2 else 0)
            draw.text((x,y),title,fill='black',font=font)
            canvas.paste(image,(x+(600-image.width)//2,y+40))
        target=report/local;target.parent.mkdir(parents=True,exist_ok=True);canvas.save(target)
        records.append({'kind':'figure','stage':stage,'local':local,'source':sources[0],
                        'sources':sources,'sha256':sha(target),'derivation':detail})

    video_specs=[('s05','ch03','desktop_pickplace_example'),('s08','ch03','pi05_official_long_task00'),
                 ('s09','ch04','hierarchical_long_example'),('s15','ch05','mobile_navigation_operation'),
                 ('s16','ch05','milestone_monitor_example'),('s20','ch06','policy_handover_success'),
                 ('s23','ch07','jetson_vision_pc_execution')]
    for stage,chapter,name in video_specs:
        source=report/'videos'/chapter/(name+'.mp4')
        capture=cv2.VideoCapture(str(source))
        count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=capture.get(cv2.CAP_PROP_FPS)
        assert count>4 and fps>0,source
        panels=[];details=[]
        for index,fraction in enumerate([0.02,0.35,0.68,0.98]):
            frame=min(count-1,round((count-1)*fraction));capture.set(cv2.CAP_PROP_POS_FRAMES,frame)
            ok,data=capture.read();assert ok,(source,frame)
            image=Image.fromarray(cv2.cvtColor(data,cv2.COLOR_BGR2RGB))
            # Fixed crop of the recorded world view, keeping the arm and work area.
            crop=None
            if chapter in {'ch05','ch06','ch07'}:
                w,h=image.size;crop=[round(w*.13),0,round(w*.87),round(h*.66)]
                image=image.crop(crop)
            panels.append((f'{chr(65+index)}  video {frame/fps:.2f} s',image))
            details.append({'frame':frame,'video_time_s':frame/fps,'crop':crop,'original_size':[data.shape[1],data.shape[0]]})
        capture.release()
        local=f'figures/{chapter}/{name}_frames.png'
        save_panels(local,panels,stage,[str(source.relative_to(report))],
                    {'operation':'recorded frame selection, fixed crop where specified, contact sheet',
                     'source_sha256':sha(source),'fps':fps,'frames':details,
                     'time_basis':'video playback time; not inference wall time or simulator steps'})

    # Four original training frames with simulator-derived point and box labels.
    dataset=workspace/'experiments/mobile_qwen/affordance/data_train_v1'
    rows=[json.loads(line) for line in (dataset/'labels.jsonl').read_text().splitlines()]
    panels=[];sources=[];annotations=[]
    for initial in [17,25,35,49]:
        row=next(x for x in rows if x['initial']==initial and x.get('handle_mask',{}).get('centroid'))
        source=dataset/'rgb'/(row['id']+'.png');image=Image.open(source).convert('RGB');draw=ImageDraw.Draw(image)
        for field,color in [('handle_mask','#e02020'),('bowl_mask','#1060d0')]:
            x,y=row[field]['centroid'];draw.line((x-9,y,x+9,y),fill=color,width=3);draw.line((x,y-9,x,y+9),fill=color,width=3)
        draw.rectangle(row['cabinet_all_bbox'],outline='#209040',width=3)
        panels.append((f"init {initial}  step {row.get('step','static')}",image))
        sources.append(str(source.relative_to(workspace)))
        annotations.append({'id':row['id'],'handle_centroid':row['handle_mask']['centroid'],
                            'bowl_centroid':row['bowl_mask']['centroid'],'cabinet_bbox':row['cabinet_all_bbox'],'source_sha256':sha(source)})
    save_panels('figures/ch06/affordance_training_labels.png',panels,'s17',sources,
                {'operation':'overlay recorded simulator mask centroids and cabinet boxes',
                 'labels_source':str((dataset/'labels.jsonl').relative_to(workspace)),
                 'labels_sha256':sha(dataset/'labels.jsonl'),'samples':annotations})

    # Model observations from one saved memory-monitor episode, not global renders.
    episode=workspace/'experiments/mobile_qwen/results/ms2_zh_i08/qwen_inputs'
    panels=[];sources=[]
    for step in [0,100,180,250]:
        source=episode/(f'check_{step:04d}.png' if step else 'manip_start.png');assert source.is_file(),source
        panels.append((f'policy step {step}' if step else 'manipulation start',Image.open(source).convert('RGB')))
        sources.append(str(source.relative_to(workspace)))
    save_panels('figures/ch05/milestone_model_observations.png',panels,'s16',sources,
                {'operation':'contact sheet of original saved model inputs','policy_steps':[0,100,180,250],
                 'source_hashes':[sha(workspace/s) for s in sources]})

    # Saved model points on the exact corresponding camera images.
    cases=workspace/'experiments/mobile_qwen/closed_loop_20261115/results'
    panels=[];sources=[];predictions=[]
    for variant in ['camera_up_3cm','camera_right_3cm']:
        for mode in ['base','lora']:
            folder=cases/f'paired_v1_{variant}_i03_{mode}';source=folder/'grounding_000.png'
            prediction=json.loads((folder/'prediction_00.json').read_text())
            image=Image.open(source).convert('RGB');x,y=prediction['point_2d'];x=x*image.width/1000;y=y*image.height/1000
            draw=ImageDraw.Draw(image);draw.ellipse((x-9,y-9,x+9,y+9),outline='#e02020',width=4)
            panels.append((f'{variant.replace("camera_", "")} / {mode}',image))
            sources.extend([str(source.relative_to(workspace)),str((folder/'prediction_00.json').relative_to(workspace))])
            predictions.append({'case':folder.name,'point_2d':prediction['point_2d'],'image_sha256':sha(source),
                                'prediction_sha256':sha(folder/'prediction_00.json')})
    save_panels('figures/ch06/camera_shift_point_predictions.png',panels,'s19',sources,
                {'operation':'overlay saved model prediction points in normalized 0-1000 coordinates','predictions':predictions})

    # Three Jetson joint episodes plus a stable-validation input, explicitly distinct sets.
    panels=[];sources=[];predictions=[]
    for initial in [2,3,4]:
        folder=cases/f'jetson_joint_v1_nominal_i{initial:02d}_lora';source=folder/'grounding_000.png'
        pred=json.loads((folder/'prediction_00.json').read_text())
        image=Image.open(source).convert('RGB');x,y=pred['point_2d'];x=x*image.width/1000;y=y*image.height/1000
        ImageDraw.Draw(image).ellipse((x-9,y-9,x+9,y+9),outline='#e02020',width=4)
        panels.append((f'Jetson joint init {initial}',image))
        sources.extend([str(source.relative_to(workspace)),str((folder/'prediction_00.json').relative_to(workspace))])
        predictions.append({'initial':initial,'point_2d':pred['point_2d'],'image_sha256':sha(source),'prediction_sha256':sha(folder/'prediction_00.json')})
    source=workspace/'experiments/mobile_qwen/jetson/stage3_20261115/validation/static_i13_0.png'
    panels.append(('validation init 13 (separate set)',Image.open(source).convert('RGB')))
    sources.append(str(source.relative_to(workspace)))
    save_panels('figures/ch07/jetson_model_inputs.png',panels,'s23',sources,
                {'operation':'three saved joint-run point overlays plus an unannotated validation input',
                 'predictions':predictions,'validation_image_sha256':sha(source)})

    local='figures/ch02/so101_wrist_sensor_mount.jpg';target=report/local
    shutil.copyfile(args.hardware_photo,target)
    records.append({'kind':'figure','stage':'s01','local':local,'source':'user_photo/IMG_0829.jpg',
                    'sha256':sha(target),'derivation':{'operation':'unchanged user-provided historical photo',
                    'source_sha256':sha(args.hardware_photo)}})
    (report/'VISUAL_ADDITIONS_V03.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    manifest=json.loads((report/'asset_manifest.json').read_text())
    new_names={x['local'] for x in records}
    manifest=[x for x in manifest if x['local'] not in new_names]+records
    (report/'asset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'Added {len(records)} figures; manifest has {sum(x["kind"]=="figure" for x in manifest)} figures')


if __name__=='__main__':
    main()

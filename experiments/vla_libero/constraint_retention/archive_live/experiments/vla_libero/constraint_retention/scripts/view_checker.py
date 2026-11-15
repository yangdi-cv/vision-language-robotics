"""Matched three-image control isolates wrist pixels from prompt/token-count changes."""
import json

SCHEMES=('base_pair','duplicate_control','wrist')


def prompt(subgoal):
    return f'''Check this single robot subgoal: "{subgoal}".
Image 1 is the third-person view when the subgoal started. Image 2 is the current third-person view. Image 3 is a current auxiliary view; it may repeat image 2.
Use the views together to identify the TARGET and its DESTINATION. A placement is done only when the target is visibly at the destination and released from the gripper. Do not infer completion merely because the target disappeared from its starting position. If destination, identity or release cannot be established from the images, choose unknown. Do not treat another object at the destination as the target.
Return only JSON: {{"status":"done" or "not_done" or "unknown", "reason":"one short sentence"}}.'''


def input_images(row,scheme):
    images=row['images'];ref=images['reference']['path'];agent=images['agent']['path'];wrist=images['wrist']['path']
    if scheme=='base_pair':return [ref,agent]
    if scheme=='duplicate_control':return [ref,agent,agent]
    if scheme=='wrist':return [ref,agent,wrist]
    raise ValueError(scheme)


def parse(raw):
    text=raw.strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    value=json.loads(text)
    if not isinstance(value,dict) or value.get('status') not in ['done','not_done','unknown']:
        raise ValueError('Invalid completion status')
    return dict(status=value['status'],done=value['status']=='done',reason=str(value.get('reason','')))

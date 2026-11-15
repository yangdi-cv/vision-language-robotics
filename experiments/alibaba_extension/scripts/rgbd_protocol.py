"""Shared evaluation/training protocol; deterministic labels from renderer metadata."""
def prompt(case):
 return f'''指令：{case['instruction']}
请先识别指令所指颜色的方块是否实际可见，禁止换成其他颜色。
可见则给bbox_1000=[左,上,右,下]，坐标按整幅图宽高归一到0-1000；不可见则bbox_1000=null。
唯一已实现技能transfer_red_cube只支持抓取红色方块。蓝色/绿色没有可执行技能；否定指令只能observe。
只输出JSON：{{"target_visible":true或false,"bbox_1000":数组或null,"selected_skill":"transfer_red_cube"或"observe"}}。'''
def answer(case):
 box=case['bbox_xyxy_pixels']
 if box is not None:box=[round(box[0]*1000/640),round(box[1]*1000/480),round(box[2]*1000/640),round(box[3]*1000/480)]
 return {'target_visible':case['target_visible'],'bbox_1000':box,'selected_skill':case['expected_skill']}

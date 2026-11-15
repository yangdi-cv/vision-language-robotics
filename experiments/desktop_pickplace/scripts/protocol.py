def prompt(case):
    return f"""桌面自主抓放任务，指令：{case['instruction']}
先看图：目标颜色是否存在，两个盒子分别是画面左侧和右侧。只有目标和目的盒子均明确，且不是否定指令时才能执行。
缺少目标、指令歧义或否定操作时，should_execute=false，target_color和destination均为null。
只输出JSON：{{"target_color":"red"或"blue"或"green"或null,"destination":"left"或"right"或null,"should_execute":true或false}}。"""

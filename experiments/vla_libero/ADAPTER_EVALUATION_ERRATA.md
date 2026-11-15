# 适配器部署实现更正（2026-11-14）

在本地固定openpi源码的注意力输出层，LoRA实际两次收缩为`BTNH,NHL->BTL`和`BTL,NLD->BTD`。原skill_bootstrap/scripts/chunk_math.py对所有层统一按逐头A@B合并，未保留这里的跨头归约。target_skill/SEMANTIC_CHECK.json用真正上游模块apply独立验证了这一不等价；修正公式与实际模块的小张量float32输出最大差约1.9e-6。旧测试只是验证了分块A@B算术，未验证上游运算语义，覆盖不足。

受影响的是使用非零LoRA合并部署的skill_bootstrap和point_target策略评测及其能力归因。原始分数仍准确描述当时实际运行的错误部署程序，但不能据此否定正确加载后的适配器。未加适配器的官方π0.5基座、CAG免训练基座、训练权重本身、教师数据及其物理验收不因这处错误改变。

原始源码、权重、报告和压缩包不覆盖。新实现位于target_skill/scripts/lora_merge.py与loader.py，将用同权重、同观察和同噪声核对部署输出，并重新做闭环。结果出来前不宣称此修正足以解决目标选择。

外部同类线索：https://github.com/Physical-Intelligence/openpi/issues/958 。该帖是问题报告，不替代本地验证；本次依据本地固定源码和实际模块算子对照修正。没有使用该帖的JAX→PyTorch转换，本地一直使用JAX。

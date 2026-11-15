# 引用与许可

平面移动关节/速度执行器设计参考ARISE-Initiative/robosuite的Omron底座，MIT许可及上游Apache声明保存在third_party/LICENSE.robosuite；固定提交与文件SHA256见third_party/sources.json。新适配器使用原有LIBERO/robosuite安装，不修改其源码；LIBERO、OpenPI、MuJoCo及模型权重保留各自许可，不将本项目代码许可覆盖到第三方资产。

Nav2配置由本机ROS2 Humble官方默认参数改编；官方navigation2代码采用Apache-2.0。实际使用其现有规划、控制和代价地图插件，未复制或重新实现规划器。参照官方RPP实现核查了跟踪与旋转逻辑。

后续可供性/分解方案参考：RoboPoint（https://github.com/wentaoyuan/RoboPoint，Apache系列许可需按下载版本继续核查；其旧Torch/Transformers依赖不覆盖当前环境）；RoboInter（https://github.com/InternRobotics/RoboInter，README明确MIT）。当前仅研究接口设计，没有下载或套用这些模型权重，也没有将对象框直接当作可供性标注。

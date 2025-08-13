# 多模态支持
本文档主要介绍xLLM推理引擎中多模态的吃场景支持模型及模态类型，以及离在线接口等。

## 支持模型
- Qwen2.5-VL: 包括7B/32B/72B多个系列。
- MiniCPM-V-2_6: 7B。

## 模态类型
- 图片: 支持单图、多图的输入，以及图片+Prompt组合、纯文本Promot等输入方式。


!!! warning "注意事项"
    - 目前，xLLM统一基于JinJa渲染ChatTemplate，部署MiniCPM-V-2_6，模型目录需提供ChatTemplate文件。
    - 图片仅支持Base64输入， 不支持输入图片Url。
    - 目前多模态模型主要支持了图片模态，视频、音频等模态正在推进中。

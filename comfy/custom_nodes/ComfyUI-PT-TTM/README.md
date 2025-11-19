PT-TTM: "为创作者而生，为自由动态而战" - 由 PT 倾情奉献Time-to-Move的强度可控实现

摘要
该技术来自于 https://github.com/time-to-move/TTM 技术团队的（Time-to-Move, TTM），
该技术由 https://github.com/kijai/ComfyUI-WanVideoWrapper kijai 进行了实现,用于通义万相2.2的图生视频高级参考图引导.

然而传统的TTM方法在参考视频动态迁移方面表现出色，但缺乏对迁移强度的精细控制。
PT通过线性插值算法在零潜变量和参考潜变量之间建立连续过渡，同时同步调整掩码强度，确保了整体影响的一致性。实验结果表明，该方法在保持视频质量的同时，为用户提供了从完全无影响到完全影响的连续控制能力。
关键词: 视频生成, 时间动态迁移, 潜变量插值, 强度控制, 计算机视觉

本工作为WanVideoWrapper的WanVideo Add TTMLatents节点添加了强度可控的TTM机制：提出了一种基于线性插值的强度控制方法，允许用户在0到1之间连续调节参考视频的影响程度
PT创作了新的PT TTM Reference Control节点,此节点可以完全替代原节点,并实现了掩码强度与潜变量强度的同步调节，确保空间和时间影响的一致性

<img width="1155" height="596" alt="B0363B12EDB4E7A580F5E81A3CAEB991" src="https://github.com/user-attachments/assets/0b754faf-fff7-4d17-82f3-335421e6cac0" />


安装方法
将PT_TTM节点包放置在ComfyUI的custom_nodes目录中，重启ComfyUI即可在节点菜单中找到"PT TTM Reference Control"节点。
此节点需要安装 https://github.com/kijai/ComfyUI-WanVideoWrapper

关于TTM的使用教程科普视频,来自b站大佬Aiwood:

https://www.bilibili.com/video/BV1syCYBCErv/?vd_source=8c0c4ef61dc5c5d2356d3037a952ed3b

以下是测试对比视频:

https://github.com/user-attachments/assets/b8c2f5f0-82a5-4240-88c7-e3335ee8fb1c




https://github.com/user-attachments/assets/c39d4080-a4de-4c6c-8771-d2868b797148


高强度（1.0）: 高度保持参考视频的动态特性


https://github.com/user-attachments/assets/07c84782-d69b-4027-a711-15aff7d77564



https://github.com/user-attachments/assets/f8ea91ca-5486-4b52-ba23-1d665763eaf1


中等强度（0.75）: 在创造性和保真度之间达到良好平衡


https://github.com/user-attachments/assets/b360193a-d91a-4754-a57e-54b103febe14


https://github.com/user-attachments/assets/53fbd215-c1c5-4356-8357-77b686f15370



低强度（0.5）: 参考视频的动态特征轻微影响生成结果，保持较高的创造性


应用场景
创意视频生成: 通过调节强度探索不同的动态效果

视频风格迁移: 控制源视频动态对目标风格的影响程度

动态特效制作: 精确控制参考动态的融入程度

结论与展望
本文提出的强度可控TTM方法解决了传统时间动态迁移技术缺乏精细控制的问题。通过简单的线性插值机制，实现了对参考视频影响的连续调节，为视频生成提供了更大的创作灵活性。

未来工作将集中在：

非线性控制策略: 探索更复杂的强度调节函数

时空分离控制: 独立控制空间和时间维度的影响

自适应强度调节: 根据内容自动优化强度参数

🙏 致谢

特别感谢： AIWOOD 开源社区 B站ID:AIWOOD QQ3群:1031710987

https://github.com/time-to-move/TTM 技术团队

通义万相团队 - 提供了优秀的原始模型

kijai: https://github.com/kijai/ComfyUI-WanVideoWrapper

ComfyUI 社区 - 创造了这个强大的平台

所有测试用户 - 你们的反馈让这个项目不断完善

开源精神 - 让知识和技术自由流动

📄 许可证

本项目采用 MIT 许可证，鼓励学习、使用和分享。

💌 联系作者

如果您在使用过程中遇到问题或有改进建议，欢迎通过 GitHub Issues 反馈。

谨以此项目，献给所有在开源道路上默默奉献的开发者们。

PT - 用代码改变世界，用热情点亮未来 ✨

"优秀的代码不仅是功能的实现，更是艺术与工程的完美结合" - PT

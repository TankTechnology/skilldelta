# SkillDelta：DeepSeek Harness 接入

插件在每个用户回合的第一个模型步骤前，根据历史配对收益决定是否注入给定
skill。它是可选的运行时接入层；论文结果由核心预测器独立复现。

在仓库根目录运行：

```bash
make plugin-demo
```

示例只用合成向量和 Python 标准库，不联网、不调用模型，预期输出
`predicted_gain: 1.0`、`enable_skill: true`。

接入 Harness：运行 `npm ci --prefix plugins/dsh-skilldelta`，修改
[配置示例](examples/profile.patch.yml) 的绝对路径，再通过
`dsh --profile headless --patch /absolute/path/profile.patch.yml --json "任务"`
加载。目标模型由已有 profile 配置，真实运行会调用该模型服务。

支持文件应提前选定技能/家族范围，包含任务 ID、向量和 Use/Skip 收益差。
本插件按余弦相似度选邻居，对 signed gain 做**均匀平均**；论文默认预测器使用
非负余弦权重。查询任务从支持邻居中排除；查询向量可来自不含标签的 query index，
也可由嵌入服务生成。

`failMode` 只在路由失败时生效：`always-on` 使用技能、`always-off` 跳过、
`reject` 中止当前步骤。示例采用 `reject`。正常预测的 Skip skill 不受其影响。
日志包含任务文本、邻居和回退原因。

完整格式与配置见 [英文说明](README.md)。`make test-plugin` 检查路由及 hook，
不调用模型。

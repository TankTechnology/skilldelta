// SkillDelta task-conditional routing for DeepSeek Harness.
//
// The frozen predictor lives in scripts/route.py so the same implementation
// can be replayed offline.  This module is deliberately a thin dsh adapter:
// it registers audit tools and adds the selected skill at agent/pre-step.

import Schema from '@deepseek-ai/schemastery'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { appendFile, mkdir, readFile } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

export const name = 'skilldelta'
export const inject = ['tools', 'shell', 'systemPrompt']

const PLUGIN_DIR = dirname(fileURLToPath(import.meta.url)) + '/..'

export const Config = Schema.object({
  pythonCmd: Schema.string().default('python3'),
  routerScript: Schema.string().description('Path to scripts/route.py'),
  support: Schema.string().description('Frozen SkillDelta support-index JSON'),
  queryIndex: Schema.string().description('Optional query-only embedding index for sealed task replay'),
  skill: Schema.string().description('Frozen skill document to inject when enabled'),
  routeLog: Schema.string().description('Append-only JSONL route audit path'),
  threshold: Schema.number().default(0).description('Enable when predicted gain is above this threshold'),
  k: Schema.number().default(10).description('Number of neighbors in the supplied eligible support index'),
  autoRoute: Schema.boolean().default(true).description('Inject the selected skill on the first step of each user turn'),
  failMode: Schema.union(['always-on', 'always-off', 'reject']).default('always-on'),
  taskIdEnv: Schema.string().default('SKILLDELTA_TASK_ID').description('Optional environment variable carrying a frozen task id for sealed replay'),
  timeoutMs: Schema.number().default(30_000),
})

function q(value) {
  const text = String(value)
  if (text.includes('\u0000')) throw new Error('router arguments cannot contain a NUL byte')
  // POSIX single quoting preserves task text without shell interpolation.
  return `'${text.replace(/'/g, `'\\''`)}'`
}

function absolute(value, cwd = process.cwd()) {
  if (!value) return value
  return value.startsWith('/') ? value : resolve(cwd, value)
}

function textOf(message) {
  if (!message?.content) return ''
  return message.content.map((block) => {
    if (typeof block === 'string') return block
    if (block?.type === 'text') return block.text || ''
    return ''
  }).filter(Boolean).join('\n')
}

function outputText(output) {
  if (output === undefined || output === null) return ''
  if (typeof output === 'string') return output
  return output.text || ''
}

function render(_args, value) {
  return [{ type: 'text', text: value }]
}

function errorResult(error) {
  return { error: error?.message || String(error) }
}

export function apply(ctx, config = {}) {
  const shell = ctx.shell
  const python = config.pythonCmd || 'python3'
  const router = absolute(config.routerScript || join(PLUGIN_DIR, 'scripts', 'route.py'))
  const cwdOf = (agent) => agent?.session?.header?.cwd || process.cwd()

  async function run(argv, agent, signal) {
    const command = argv.map(q).join(' ')
    const request = { command, timeoutMs: config.timeoutMs, signal }
    const spec = typeof shell.resolve === 'function'
      ? shell.resolve({ ...request, workdir: cwdOf(agent) })
      : request
    const result = await shell.run(spec)
    const stdout = outputText(result?.stdout)
    const stderr = outputText(result?.stderr)
    if (result?.timedOut) throw new Error(`router timed out: ${stderr || stdout}`)
    if (result?.exitCode !== 0) throw new Error(stderr || stdout || `router exit=${result?.exitCode}`)
    return stdout.trim()
  }

  function routeArgv(args, agent) {
    const support = absolute(args.support || config.support, cwdOf(agent))
    if (!support) throw new Error('SkillDelta support index is not configured')
    const task = args.task || ''
    const argv = [python, router, 'route', '--support', support,
      '--task-text', task, '--threshold', args.threshold ?? config.threshold ?? 0,
      '--k', args.k ?? config.k ?? 10]
    if (args.taskId) argv.push('--task-id', args.taskId)
    const queryIndex = absolute(args.queryIndex || config.queryIndex, cwdOf(agent))
    if (queryIndex) argv.push('--query-index', queryIndex)
    if (args.queryEmbedding) argv.push('--query-embedding', args.queryEmbedding)
    return argv
  }

  async function route(args, agent, signal) {
    const raw = await run(routeArgv(args, agent), agent, signal)
    const value = JSON.parse(raw)
    if (value.schema !== 'skilldelta-route-v1') throw new Error('router returned an unexpected schema')
    return value
  }

  async function writeAudit(row, agent) {
    const configured = config.routeLog
    if (!configured) return
    const path = absolute(configured, cwdOf(agent))
    await mkdir(dirname(path), { recursive: true })
    await appendFile(path, `${JSON.stringify(row, null, 0)}\n`, 'utf8')
  }

  ctx.systemPrompt.section({
    name: 'skilldelta:route',
    order: 115,
    text: 'SkillDelta is active. The plugin evaluates the first user task of each turn with a frozen task-conditional gate and injects the configured skill only when the gate enables it. Do not duplicate the injected skill or infer success from the routing decision.',
  })

  const routeTool = defineTool({
    name: 'skilldelta_route',
    description: 'Inspect the frozen SkillDelta route for a task without injecting a skill.',
    parameters: {
      task: { type: 'string', required: true, description: 'The user task text' },
      taskId: { type: 'string', description: 'Known frozen task id for a no-network replay' },
      threshold: { type: 'number', description: 'Override the configured gain threshold' },
      k: { type: 'number', description: 'Override the support neighbor count' },
      queryEmbedding: { type: 'string', description: 'JSON vector for sealed offline replay' },
    },
    output: { schema: { type: 'string' }, render },
    async execute(args, exec) {
      try {
        return JSON.stringify(await route(args || {}, exec?.agent, exec?.signal))
      } catch (error) {
        return JSON.stringify(errorResult(error))
      }
    },
  })
  ctx.tools.register(routeTool)

  ctx.tools.register(defineTool({
    name: 'skilldelta_status',
    description: 'Show the frozen SkillDelta support index and configured skill digest.',
    parameters: {},
    output: { schema: { type: 'string' }, render },
    async execute(_args, exec) {
      try {
        const support = absolute(config.support, cwdOf(exec?.agent))
        if (!support) throw new Error('SkillDelta support index is not configured')
        const argv = [python, router, 'status', '--support', support]
        if (config.skill) argv.push('--skill', absolute(config.skill, cwdOf(exec?.agent)))
        return await run(argv, exec?.agent, exec?.signal)
      } catch (error) {
        return JSON.stringify(errorResult(error))
      }
    },
  }))

  ctx.on('agent/pre-step', async ({ agent, messages, turn, step, signal }, next) => {
    const downstream = await next()
    if (downstream.kind === 'reject' || !config.autoRoute || step !== 1) return downstream
    if (!config.skill || !config.support) return downstream
    const task = messages.map(textOf).filter(Boolean).join('\n\n').trim()
    if (!task) return downstream
    let decision
    try {
      const taskId = config.taskIdEnv ? process.env[config.taskIdEnv] : undefined
      decision = await route({ task, ...(taskId ? { taskId } : {}) }, agent, signal)
    } catch (error) {
      const mode = config.failMode || 'always-on'
      const fallback = mode === 'always-on'
      const row = { schema: 'skilldelta-route-v1', turn, step, task, fallback: mode,
        enable_skill: fallback, error: error?.message || String(error) }
      if (mode === 'reject') {
        await writeAudit(row, agent)
        return { kind: 'reject' }
      }
      decision = row
    }
    await writeAudit({ ...decision, turn, step }, agent)
    if (!decision.enable_skill) return downstream
    const skillPath = absolute(config.skill, cwdOf(agent))
    const skillText = await readFile(skillPath, 'utf8')
    const injected = createUserMessage({
      content: [{ type: 'text', text: skillText }],
      source: { kind: 'plugin', plugin: name, form: 'skilldelta-route', turn },
    })
    return { ...downstream, messages: [...downstream.messages, injected] }
  })

  ctx.logger?.info?.('[skilldelta] registered route/status tools and agent pre-step gate')
}

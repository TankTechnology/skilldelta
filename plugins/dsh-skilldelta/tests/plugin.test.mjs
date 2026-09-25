import assert from 'node:assert/strict'
import { exec } from 'node:child_process'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import test from 'node:test'
import { apply } from '../src/index.js'

const root = fileURLToPath(new URL('../', import.meta.url))
const execute = promisify(exec)
const python = process.env.SKILLDELTA_TEST_PYTHON || 'python3'

async function setup(t, options = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'skilldelta-plugin-'))
  t.after(() => rm(directory, { recursive: true, force: true }))
  const log = join(directory, 'nested', 'routes.jsonl')
  const tools = new Map()
  const hooks = new Map()
  const config = {
    support: join(root, 'examples/support.json'),
    queryIndex: join(root, 'examples/queries.json'),
    skill: join(root, 'examples/skill.md'),
    pythonCmd: python,
    routeLog: log,
    k: 2,
    threshold: 0,
    autoRoute: true,
    taskIdEnv: 'SKILLDELTA_PLUGIN_TEST_TASK_ID',
    failMode: 'reject',
    ...options,
  }
  const previous = process.env.SKILLDELTA_PLUGIN_TEST_TASK_ID
  process.env.SKILLDELTA_PLUGIN_TEST_TASK_ID = 'demo-help'
  t.after(() => {
    if (previous === undefined) delete process.env.SKILLDELTA_PLUGIN_TEST_TASK_ID
    else process.env.SKILLDELTA_PLUGIN_TEST_TASK_ID = previous
  })
  apply({
    shell: {
      resolve(request) { return request },
      async run(request) {
        const env = { ...process.env, SKILLDELTA_EMBEDDING_BASE_URL: '', SKILLDELTA_EMBEDDING_API_KEY: '' }
        try {
          const output = await execute(request.command, { cwd: request.workdir, env })
          return { exitCode: 0, stdout: { text: output.stdout }, stderr: { text: output.stderr } }
        } catch (error) {
          return { exitCode: error.code, stdout: { text: error.stdout || '' }, stderr: { text: error.stderr || '' } }
        }
      },
    },
    tools: { register(tool) { tools.set(tool.name, tool) } },
    systemPrompt: { section() {} },
    on(event, listener) { hooks.set(event, listener) },
    logger: { info() {} },
  }, config)
  const agent = { session: { header: { cwd: resolve(root) } } }
  const downstream = { kind: 'enter', messages: [] }
  async function step(number = 1) {
    return hooks.get('agent/pre-step')({ agent, turn: 1, step: number,
      messages: [{ content: [{ type: 'text', text: 'Example calculation task' }] }],
      signal: new AbortController().signal,
    }, async () => downstream)
  }
  async function audit() {
    return (await readFile(log, 'utf8')).trim().split('\n').map(JSON.parse)
  }
  return { tools, agent, step, audit, downstream }
}

test('real router injects the supplied skill only at the first step', async t => {
  const f = await setup(t)
  const result = await f.step()
  assert.equal(result.kind, 'enter')
  assert.equal(result.messages.length, 1)
  assert.match(result.messages[0].content[0].text, /Example calculation skill/)
  assert.equal((await f.audit())[0].enable_skill, true)
  assert.equal(await f.step(2), f.downstream)
  assert.equal((await f.audit()).length, 1)
})

test('a normal negative prediction skips even with always-on error fallback', async t => {
  const f = await setup(t, { failMode: 'always-on' })
  process.env.SKILLDELTA_PLUGIN_TEST_TASK_ID = 'demo-skip'
  assert.equal(await f.step(), f.downstream)
  const [row] = await f.audit()
  assert.equal(row.enable_skill, false)
  assert.equal(row.fallback, undefined)
})

for (const mode of ['always-on', 'always-off', 'reject']) {
  test(`routing failure is logged and follows ${mode}`, async t => {
    const f = await setup(t, { support: join(root, 'examples/missing.json'), failMode: mode })
    const result = await f.step()
    const [row] = await f.audit()
    assert.equal(row.fallback, mode)
    assert.ok(row.error)
    assert.equal(row.enable_skill, mode === 'always-on')
    if (mode === 'reject') assert.equal(result.kind, 'reject')
    else assert.equal(result.messages.length, mode === 'always-on' ? 1 : 0)
  })
}

test('inspection tool preserves quotes, newlines, and shell characters as task text', async t => {
  const f = await setup(t)
  const task = "What's 2 + 2?\nKeep $HOME and $(printf injected) literal."
  const value = JSON.parse(await f.tools.get('skilldelta_route').execute(
    { task, taskId: 'demo-help' }, { agent: f.agent },
  ))
  assert.equal(value.task_text, task)
  assert.equal(value.query_source, 'frozen-query-index')
  const status = JSON.parse(await f.tools.get('skilldelta_status').execute({}, { agent: f.agent }))
  assert.equal(status.support.count, 4)
  assert.match(status.skill.sha256, /^[0-9a-f]{64}$/)
})

import { describe, expect, it } from 'vitest'
import { Agent } from '../agent.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { createMockTool } from '../../__fixtures__/tool-helpers.js'
import { TextBlock, ToolResultBlock } from '../../types/messages.js'
import { AfterToolCallEvent, BeforeToolCallEvent, BeforeToolsEvent } from '../../hooks/events.js'
import { InterruptResponseContent } from '../../types/interrupt.js'

function toolResultIds(agent: Agent): string[] {
  return agent.messages.flatMap((message) =>
    message.content
      .filter((block): block is ToolResultBlock => block.type === 'toolResultBlock')
      .map((block) => block.toolUseId)
  )
}

describe('tool results in the conversation', () => {
  it('records one result when a resume is cancelled while the tool runs', async () => {
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'approver', toolUseId: 'tool-1', input: {} })
      .addTurn({ type: 'textBlock', text: 'Done' })
      .addTurn({ type: 'textBlock', text: 'Nothing else' })
    let raised = false
    let agentRef: Agent | undefined
    const tool = createMockTool('approver', (context) => {
      if (!raised) {
        raised = true
        context.interrupt({ name: 'approve', reason: 'proceed?' })
      }
      agentRef!.cancel()
      return 'approved'
    })
    const agent = new Agent({ model, tools: [tool], printer: false })
    agentRef = agent

    const interrupted = await agent.invoke('go')
    const response = [new InterruptResponseContent({ interruptId: interrupted.interrupts![0]!.id, response: 'yes' })]
    expect((await agent.invoke(response)).stopReason).toBe('cancelled')
    await agent.invoke('what happened?')

    expect(toolResultIds(agent)).toEqual(['tool-1'])
  })

  it('records one result when a resume is cancelled before the tool runs', async () => {
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'approver', toolUseId: 'tool-1', input: {} })
      .addTurn({ type: 'textBlock', text: 'Done' })
    let raised = false
    const tool = createMockTool('approver', (context) => {
      if (!raised) {
        raised = true
        context.interrupt({ name: 'approve', reason: 'proceed?' })
      }
      return 'approved'
    })
    const agent = new Agent({ model, tools: [tool], printer: false })

    const interrupted = await agent.invoke('go')
    const response = [new InterruptResponseContent({ interruptId: interrupted.interrupts![0]!.id, response: 'yes' })]
    const removeCancelHook = agent.addHook(BeforeToolsEvent, () => {
      agent.cancel()
    })
    expect((await agent.invoke(response)).stopReason).toBe('cancelled')
    removeCancelHook()
    expect((await agent.invoke(response)).stopReason).toBe('endTurn')

    expect(toolResultIds(agent)).toEqual(['tool-1'])
  })

  it('records a result for a concurrent tool that throws a non-Error value', async () => {
    const model = new MockMessageModel()
      .addTurn([
        { type: 'toolUseBlock', name: 'thrower', toolUseId: 't1', input: {} },
        { type: 'toolUseBlock', name: 'okTool', toolUseId: 't2', input: {} },
      ])
      .addTurn({ type: 'textBlock', text: 'Done' })
    const thrower = createMockTool('thrower', () => {
      throw new DOMException('aborted elsewhere', 'AbortError')
    })
    const okTool = createMockTool('okTool', () => 'ok')
    const agent = new Agent({ model, tools: [thrower, okTool], printer: false })

    expect((await agent.invoke('go')).stopReason).toBe('endTurn')

    expect(toolResultIds(agent)).toEqual(['t1', 't2'])
  })

  it('keeps tool-use order when a tool name is unknown', async () => {
    const model = new MockMessageModel()
      .addTurn([
        { type: 'toolUseBlock', name: 'okTool', toolUseId: 't1', input: {} },
        { type: 'toolUseBlock', name: 'functions.okTool', toolUseId: 't2', input: {} },
      ])
      .addTurn({ type: 'textBlock', text: 'Done' })
    const agent = new Agent({ model, tools: [createMockTool('okTool', () => 'ok')], printer: false })

    await agent.invoke('go')

    expect(toolResultIds(agent)).toEqual(['t1', 't2'])
  })

  it('keeps tool-use order when an interrupt splits the batch', async () => {
    const model = new MockMessageModel()
      .addTurn([
        { type: 'toolUseBlock', name: 'approver', toolUseId: 't1', input: {} },
        { type: 'toolUseBlock', name: 'okTool', toolUseId: 't2', input: {} },
      ])
      .addTurn({ type: 'textBlock', text: 'Done' })
    let raised = false
    const approver = createMockTool('approver', (context) => {
      if (!raised) {
        raised = true
        context.interrupt({ name: 'approve', reason: 'proceed?' })
      }
      return 'approved'
    })
    const agent = new Agent({ model, tools: [approver, createMockTool('okTool', () => 'ok')], printer: false })

    const interrupted = await agent.invoke('go')
    await agent.invoke([new InterruptResponseContent({ interruptId: interrupted.interrupts![0]!.id, response: 'yes' })])

    expect(toolResultIds(agent)).toEqual(['t1', 't2'])
  })

  it('keeps the model-issued toolUseId when a hook replaces toolUse', async () => {
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'okTool', toolUseId: 't1', input: {} })
      .addTurn({ type: 'textBlock', text: 'Done' })
    const agent = new Agent({ model, tools: [createMockTool('okTool', () => 'ok')], printer: false })
    agent.addHook(BeforeToolCallEvent, (event) => {
      event.toolUse = { ...event.toolUse, toolUseId: 'replaced' }
    })

    await agent.invoke('go')

    expect(toolResultIds(agent)).toEqual(['t1'])
  })

  it('keeps the model-issued toolUseId when a hook replaces the result', async () => {
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'okTool', toolUseId: 't1', input: {} })
      .addTurn({ type: 'textBlock', text: 'Done' })
    const agent = new Agent({ model, tools: [createMockTool('okTool', () => 'ok')], printer: false })
    agent.addHook(AfterToolCallEvent, (event) => {
      event.result = new ToolResultBlock({ toolUseId: 'replaced', status: 'success', content: [new TextBlock('x')] })
    })

    await agent.invoke('go')

    expect(toolResultIds(agent)).toEqual(['t1'])
  })

  it('does not run a completed tool again when a BeforeToolsEvent interrupt fires on the resume', async () => {
    const model = new MockMessageModel()
      .addTurn([
        { type: 'toolUseBlock', name: 'charge', toolUseId: 't1', input: {} },
        { type: 'toolUseBlock', name: 'approver', toolUseId: 't2', input: {} },
      ])
      .addTurn({ type: 'textBlock', text: 'Done' })
    const charges: string[] = []
    let pass = 0
    const charge = createMockTool('charge', () => {
      charges.push('charged')
      return 'charged'
    })
    const approver = createMockTool('approver', (context) => {
      if (pass === 1) context.interrupt({ name: 'approve', reason: 'proceed?' })
      return 'approved'
    })
    const agent = new Agent({ model, tools: [charge, approver], toolExecutor: 'sequential', printer: false })
    agent.addHook(BeforeToolsEvent, (event) => {
      pass += 1
      if (pass === 2) event.interrupt({ name: 'batch_gate', reason: 'confirm the batch' })
    })
    const responses = (): InterruptResponseContent[] =>
      Object.keys(
        (agent as unknown as { _interruptState: { interrupts: Record<string, unknown> } })._interruptState.interrupts
      ).map((interruptId) => new InterruptResponseContent({ interruptId, response: 'yes' }))

    expect((await agent.invoke('go')).stopReason).toBe('interrupt')
    expect((await agent.invoke(responses())).stopReason).toBe('interrupt')
    expect((await agent.invoke(responses())).stopReason).toBe('endTurn')

    expect(charges).toEqual(['charged'])
    expect(toolResultIds(agent)).toEqual(['t1', 't2'])
  })

  it('runs the AfterToolCallEvent hooks once when a hook rethrows the tool error', async () => {
    const model = new MockMessageModel()
      .addTurn({ type: 'toolUseBlock', name: 'failing', toolUseId: 't1', input: {} })
      .addTurn({ type: 'textBlock', text: 'Done' })
    const failing = createMockTool('failing', () => {
      throw new Error('boom')
    })
    const agent = new Agent({ model, tools: [failing], printer: false })
    const seen: string[] = []
    agent.addHook(AfterToolCallEvent, (event) => {
      seen.push(event.toolUse.toolUseId)
      if (event.error) {
        throw event.error
      }
    })

    await expect(agent.invoke('go')).rejects.toThrow('boom')

    expect(seen).toEqual(['t1'])
  })
})

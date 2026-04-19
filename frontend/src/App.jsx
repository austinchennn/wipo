import { useCallback, useEffect, useRef, useState } from 'react'
import { useSimulation } from './hooks/useSimulation'

// ── Agent 类型颜色映射 ──
const AGENT_COLORS = {
  HostAgent: '#60A5FA',
  NormalAgent: '#9CA3AF',
  InstTraderAgent: '#34D399',
  RetailTraderAgent: '#F87171',
}

const AGENT_LABELS = {
  HostAgent: 'HOST',
  NormalAgent: 'NM',
  InstTraderAgent: 'IA',
  RetailTraderAgent: 'RT',
}

const SENTIMENT_COLORS = {
  bull: '#34D399',
  bear: '#F87171',
  neutral: '#9CA3AF',
}

// ── 控制面板 ──
function ControlPanel({ onStart, running, connected }) {
  const [params, setParams] = useState({
    n_normal: 20,
    n_inst: 5,
    n_retail: 15,
    seed: 42,
  })

  const handleStart = () => {
    onStart(params)
  }

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-zinc-300">控制面板</h2>
        <span
          className={`h-2 w-2 rounded-full ${connected ? 'bg-green-400' : 'bg-red-400'}`}
          title={connected ? 'WebSocket 已连接' : 'WebSocket 未连接'}
        />
      </div>
      <div className="grid grid-cols-4 gap-2 text-xs">
        {Object.entries({ n_normal: '普通人', n_inst: '机构', n_retail: '散户', seed: '种子' }).map(
          ([key, label]) => (
            <label key={key} className="flex flex-col gap-1 text-zinc-400">
              {label}
              <input
                type="number"
                className="rounded border border-zinc-600 bg-zinc-800 px-2 py-1 text-zinc-200 [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                value={params[key]}
                onChange={(e) => setParams((p) => ({ ...p, [key]: parseInt(e.target.value) || 0 }))}
                disabled={running}
              />
            </label>
          ),
        )}
      </div>
      <button
        className="mt-3 w-full rounded bg-indigo-600 py-1.5 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:opacity-50"
        onClick={handleStart}
        disabled={running || !connected}
      >
        {running ? '模拟运行中...' : '启动模拟'}
      </button>
    </div>
  )
}

// ── 状态指示器 ──
function StatusBar({ simState }) {
  const pct = simState.totalRounds > 0 ? (simState.round / simState.totalRounds) * 100 : 0
  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <div className="mb-2 flex items-center justify-between text-xs text-zinc-400">
        <span>轮次 {simState.round} / {simState.totalRounds}</span>
        <span>{simState.currentPhase || 'Idle'}</span>
        <span>{simState.currentTopic || '-'}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-zinc-700">
        <div
          className="h-full rounded-full bg-indigo-500 transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ── 评论流 ──
function CommentFeed({ comments }) {
  const feedRef = useRef(null)
  const prevLen = useRef(0)

  // 自动滚动到底部
  if (feedRef.current && comments.length > prevLen.current) {
    requestAnimationFrame(() => {
      feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight, behavior: 'smooth' })
    })
  }
  prevLen.current = comments.length

  return (
    <div className="flex flex-col rounded-lg border border-zinc-700 bg-zinc-900">
      <h2 className="border-b border-zinc-700 px-4 py-2 text-sm font-semibold text-zinc-300">
        评论流 <span className="text-zinc-500">({comments.length})</span>
      </h2>
      <div ref={feedRef} className="max-h-[500px] overflow-y-auto p-3 space-y-2">
        {comments.length === 0 ? (
          <p className="text-center text-xs text-zinc-500">等待模拟启动...</p>
        ) : (
          comments.map((c, i) => (
            <div key={c.id || i} className="rounded border border-zinc-800 bg-zinc-950 p-2.5 text-xs">
              <div className="mb-1 flex items-center gap-2">
                <span
                  className="rounded px-1.5 py-0.5 text-[10px] font-bold text-white"
                  style={{ backgroundColor: AGENT_COLORS[c.agentType] || '#666' }}
                >
                  {AGENT_LABELS[c.agentType] || c.agentType}
                </span>
                <span className="text-zinc-500">#{c.agentId}</span>
                <span className="text-zinc-600">P{c.phase}</span>
                {c.parentId && <span className="text-zinc-600">↩ {c.parentId}</span>}
                <span
                  className="ml-auto rounded px-1 py-0.5 text-[10px]"
                  style={{ color: SENTIMENT_COLORS[c.sentiment] || '#999' }}
                >
                  {c.sentiment}
                </span>
              </div>
              <p className="text-zinc-300 leading-relaxed">{c.content}</p>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

// ── K 线简易图 ──
function KLineChart({ klines }) {
  if (klines.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center rounded-lg border border-zinc-700 bg-zinc-900 text-xs text-zinc-500">
        等待交易数据...
      </div>
    )
  }

  const last = klines[klines.length - 1]
  const first = klines[0]
  const changePct = ((last.close - first.open) / first.open * 100).toFixed(2)
  const isUp = last.close >= first.open

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-zinc-300">股价 K 线</h2>
        <div className="text-right">
          <span className={`text-lg font-bold ${isUp ? 'text-green-400' : 'text-red-400'}`}>
            ${last.close.toFixed(2)}
          </span>
          <span className={`ml-2 text-xs ${isUp ? 'text-green-400' : 'text-red-400'}`}>
            {isUp ? '+' : ''}{changePct}%
          </span>
        </div>
      </div>
      <div className="flex h-24 items-end gap-px">
        {klines.map((bar, i) => {
          const maxH = Math.max(...klines.map((b) => b.high))
          const minL = Math.min(...klines.map((b) => b.low))
          const range = maxH - minL || 1
          const bodyTop = Math.max(bar.open, bar.close)
          const bodyBot = Math.min(bar.open, bar.close)
          const bodyH = Math.max(((bodyTop - bodyBot) / range) * 96, 1)
          const bodyY = ((maxH - bodyTop) / range) * 96
          const up = bar.close >= bar.open

          return (
            <div
              key={i}
              className="relative flex-1"
              style={{ height: '96px' }}
              title={`T${bar.tick} O:${bar.open.toFixed(2)} H:${bar.high.toFixed(2)} L:${bar.low.toFixed(2)} C:${bar.close.toFixed(2)} V:${bar.volume}`}
            >
              <div
                className={`absolute left-1/2 w-px -translate-x-1/2 ${up ? 'bg-green-500' : 'bg-red-500'}`}
                style={{
                  top: `${((maxH - bar.high) / range) * 96}px`,
                  height: `${((bar.high - bar.low) / range) * 96}px`,
                }}
              />
              <div
                className={`absolute left-0 right-0 rounded-sm ${up ? 'bg-green-500' : 'bg-red-500'}`}
                style={{ top: `${bodyY}px`, height: `${bodyH}px` }}
              />
            </div>
          )
        })}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-zinc-500">
        <span>T{klines[0].tick}</span>
        <span>成交 {klines.reduce((s, b) => s + b.volume, 0)} 股</span>
        <span>T{last.tick}</span>
      </div>
    </div>
  )
}

// ── 系统日志 ──
function LogPanel({ logs }) {
  const logRef = useRef(null)
  const prevLen = useRef(0)

  if (logRef.current && logs.length > prevLen.current) {
    requestAnimationFrame(() => {
      logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' })
    })
  }
  prevLen.current = logs.length

  const levelColor = { SYS: 'text-blue-400', HOST: 'text-cyan-400', MKT: 'text-yellow-400', ERR: 'text-red-400' }

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900">
      <h2 className="border-b border-zinc-700 px-4 py-2 text-sm font-semibold text-zinc-300">
        系统日志
      </h2>
      <div ref={logRef} className="max-h-60 overflow-y-auto p-3 font-mono text-[11px]">
        {logs.map((log, i) => (
          <div key={i} className="flex gap-2 leading-5">
            <span className="text-zinc-600 shrink-0">{log.time}</span>
            <span className={`shrink-0 w-8 ${levelColor[log.level] || 'text-zinc-400'}`}>
              {log.level}
            </span>
            <span className="text-zinc-300">{log.msg}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── 情绪快照 ──
function SentimentPanel({ sentiments }) {
  if (sentiments.length === 0) return null

  const latest = sentiments[sentiments.length - 1]
  const { bull = 0, bear = 0, neutral = 0 } = latest.summary
  const total = bull + bear + neutral || 1

  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <h2 className="mb-2 text-sm font-semibold text-zinc-300">
        情绪分布 <span className="text-zinc-500 text-xs">Phase {latest.phase}</span>
      </h2>
      <div className="flex h-4 overflow-hidden rounded-full">
        <div className="bg-green-500 transition-all" style={{ width: `${(bull / total) * 100}%` }} />
        <div className="bg-red-500 transition-all" style={{ width: `${(bear / total) * 100}%` }} />
        <div className="bg-zinc-500 transition-all" style={{ width: `${(neutral / total) * 100}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-[10px]">
        <span className="text-green-400">Bull {bull}</span>
        <span className="text-red-400">Bear {bear}</span>
        <span className="text-zinc-400">Neutral {neutral}</span>
      </div>
    </div>
  )
}

// ── 设置面板（API Key） ──
function SettingsPanel({ open, onClose }) {
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('wipo_api_key') || '')
  const [visible, setVisible] = useState(false)

  const handleSave = () => {
    const trimmed = apiKey.trim()
    if (trimmed) {
      localStorage.setItem('wipo_api_key', trimmed)
    } else {
      localStorage.removeItem('wipo_api_key')
    }
    onClose()
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="w-full max-w-md rounded-lg border border-zinc-700 bg-zinc-900 p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="mb-4 text-sm font-semibold text-zinc-200">设置</h3>
        <label className="block text-xs text-zinc-400 mb-1">Google Gemini API Key</label>
        <div className="relative">
          <input
            type={visible ? 'text' : 'password'}
            className="w-full rounded border border-zinc-600 bg-zinc-800 px-3 py-2 pr-16 text-sm text-zinc-200 placeholder-zinc-500 focus:border-indigo-500 focus:outline-none"
            placeholder="AIza..."
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            autoFocus
          />
          <button
            type="button"
            className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-zinc-400 hover:text-zinc-200"
            onClick={() => setVisible((v) => !v)}
          >
            {visible ? '隐藏' : '显示'}
          </button>
        </div>
        <p className="mt-1 text-[10px] text-zinc-500">仅保存在浏览器本地，不会上传到数据库。</p>
        <div className="mt-4 flex justify-end gap-2">
          <button
            className="rounded px-3 py-1.5 text-xs text-zinc-400 hover:text-zinc-200"
            onClick={onClose}
          >
            取消
          </button>
          <button
            className="rounded bg-indigo-600 px-4 py-1.5 text-xs font-medium text-white hover:bg-indigo-500"
            onClick={handleSave}
          >
            保存
          </button>
        </div>
      </div>
    </div>
  )
}

// ═══════════════════════════════════════════════════
//  App
// ═══════════════════════════════════════════════════

function App() {
  const [settingsOpen, setSettingsOpen] = useState(false)

  const {
    connected,
    simState,
    comments,
    posts,
    klines,
    sentiments,
    logs,
    start,
  } = useSimulation()

  const handleStart = useCallback(
    async (params) => {
      const apiKey = localStorage.getItem('wipo_api_key') || ''
      const res = await start({ ...params, api_key: apiKey })
      if (!res.ok) {
        alert(res.error || '启动失败')
      }
    },
    [start],
  )

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200">
      <SettingsPanel open={settingsOpen} onClose={() => setSettingsOpen(false)} />

      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-3">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <h1 className="text-base font-bold tracking-tight">
            WIPO <span className="text-indigo-400">金融舆情论坛沙盘</span>
          </h1>
          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <span>帖子 {posts.length}</span>
            <span>评论 {comments.length}</span>
            <span>K线 {klines.length}</span>
            <button
              className="ml-2 rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200 transition"
              onClick={() => setSettingsOpen(true)}
              title="设置"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
              </svg>
            </button>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="mx-auto max-w-7xl p-4">
        <div className="grid grid-cols-12 gap-4">
          {/* Left Column: Controls + K线 + 情绪 + 日志 */}
          <div className="col-span-4 space-y-4">
            <ControlPanel onStart={handleStart} running={simState.running} connected={connected} />
            <StatusBar simState={simState} />
            <KLineChart klines={klines} />
            <SentimentPanel sentiments={sentiments} />
            <LogPanel logs={logs} />
          </div>

          {/* Right Column: 评论流 */}
          <div className="col-span-8">
            <CommentFeed comments={comments} />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App

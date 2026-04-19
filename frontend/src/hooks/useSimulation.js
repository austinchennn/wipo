import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * useSimulation — 管理 WebSocket 连接和模拟状态的核心 Hook
 *
 * 返回：
 *   state     — 当前模拟状态
 *   comments  — 实时评论流
 *   klines    — K 线序列
 *   logs      — 系统日志
 *   start     — 启动模拟
 *   connected — WebSocket 是否连接
 */

const WS_URL =
  (location.protocol === 'https:' ? 'wss://' : 'ws://') +
  location.host +
  '/ws/feed'

export function useSimulation() {
  const wsRef = useRef(null)
  const reconnectTimer = useRef(null)

  const [connected, setConnected] = useState(false)
  const [simState, setSimState] = useState({
    running: false,
    round: 0,
    totalRounds: 12,
    currentPhase: null,
    currentTopic: null,
  })
  const [comments, setComments] = useState([])
  const [posts, setPosts] = useState([])
  const [klines, setKlines] = useState([])
  const [sentiments, setSentiments] = useState([])
  const [logs, setLogs] = useState([])

  const addLog = useCallback((level, msg) => {
    const time = new Date().toLocaleTimeString('en-GB')
    setLogs((prev) => [...prev.slice(-199), { time, level, msg }])
  }, [])

  // ── WebSocket 消息处理 ──
  const handleMessage = useCallback(
    (event) => {
      let data
      try {
        data = JSON.parse(event.data)
      } catch {
        return
      }

      switch (data.type) {
        case 'system':
          if (data.event === 'sim_start') {
            setSimState((s) => ({
              ...s,
              running: true,
              round: 0,
              totalRounds: data.rounds,
            }))
            setComments([])
            setPosts([])
            setKlines([])
            setSentiments([])
            addLog('SYS', `模拟启动 | Agent=${data.n_total} (Active=${data.n_active})`)
          } else if (data.event === 'round_start') {
            setSimState((s) => ({ ...s, round: data.round }))
            addLog('SYS', `═══ 宏观轮次 ${data.round}/${data.total_rounds} ═══`)
          } else if (data.event === 'sim_end') {
            setSimState((s) => ({ ...s, running: false }))
            addLog(
              'SYS',
              `模拟结束 | 帖子=${data.total_posts} 评论=${data.total_comments} 最新价=${data.market?.last_price?.toFixed(2)}`,
            )
          } else if (data.event === 'sim_error') {
            setSimState((s) => ({ ...s, running: false }))
            addLog('ERR', `模拟错误: ${data.error}`)
          }
          break

        case 'post':
          setPosts((prev) => [
            ...prev,
            {
              id: data.post_id,
              round: data.round,
              topic: data.topic,
              topicCn: data.topic_cn,
              content: data.content,
              timestamp: data.created_at,
            },
          ])
          setSimState((s) => ({ ...s, currentTopic: data.topic_cn }))
          addLog('HOST', `发帖 ${data.post_id} → ${data.topic_cn}`)
          break

        case 'phase':
          if (data.event === 'phase_start') {
            setSimState((s) => ({
              ...s,
              currentPhase: `Phase ${data.phase}`,
            }))
            addLog('SYS', `── Phase ${data.phase}: ${data.label} ──`)
          }
          break

        case 'comment':
          setComments((prev) => [...prev, data.comment])
          break

        case 'sentiment':
          setSentiments((prev) => [
            ...prev,
            {
              postId: data.post_id,
              phase: data.phase,
              summary: data.summary,
              totalComments: data.total_comments,
            },
          ])
          addLog(
            'SYS',
            `Phase ${data.phase} 完成 | 评论=${data.total_comments} | bull=${data.summary.bull} bear=${data.summary.bear} neutral=${data.summary.neutral}`,
          )
          break

        case 'trade':
          if (data.bar) {
            setKlines((prev) => [...prev, data.bar])
            addLog(
              'MKT',
              `撮合完成 | 价格=${data.last_price?.toFixed(2)} 成交=${data.bar.trade_count}笔 | ${data.event}`,
            )
          } else {
            addLog('MKT', `${data.event} (无成交)`)
          }
          break

        case 'pong':
          break

        default:
          break
      }
    },
    [addLog],
  )

  // ── WebSocket 连接管理 ──
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      addLog('SYS', 'WebSocket 已连接')
    }
    ws.onclose = () => {
      setConnected(false)
      // 自动重连
      reconnectTimer.current = setTimeout(connectWs, 2000)
    }
    ws.onerror = () => ws.close()
    ws.onmessage = handleMessage
  }, [handleMessage, addLog])

  useEffect(() => {
    connectWs()
    return () => {
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connectWs])

  // ── 心跳 ──
  useEffect(() => {
    const iv = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping')
      }
    }, 30_000)
    return () => clearInterval(iv)
  }, [])

  // ── 启动模拟 ──
  const start = useCallback(
    async (params = {}) => {
      const resp = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      return resp.json()
    },
    [],
  )

  return {
    connected,
    simState,
    comments,
    posts,
    klines,
    sentiments,
    logs,
    start,
  }
}

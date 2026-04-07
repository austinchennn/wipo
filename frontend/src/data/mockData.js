/**
 * Mock data — 全部前端静态数据，不连接后端
 */

// ── 系统状态 ──
export const SYSTEM_STATE = {
  macroRound: 4,
  totalRounds: 12,
  currentPhase: 'Phase 2',
  currentTopic: 'Financials',
  agentsResponded: 1420,
  totalAgents: 10000,
  tradesCompleted: 12400,
  statusText: '聚合 Phase 2 评论快照中...',
}

// ── 模拟股票 ──
export const STOCK_INFO = {
  symbol: 'NEXTGEN_IPO',
  currentPrice: 124.5,
  change24h: -4.2,
  high24h: 132.8,
  low24h: 121.3,
  volume: '2.4M',
  marketCap: '12.4B',
}

// ── Agent 类型配置 ──
export const AGENT_TYPES = {
  HOST: { label: 'HOST', abbr: 'HT', color: '#60A5FA', role: '主持人' },
  NORMAL: { label: 'Normal', abbr: 'NM', color: '#707070', role: '普通人' },
  INST: { label: 'Inst Analyst', abbr: 'IA', color: '#34D399', role: '机构交易者' },
  RETAIL: { label: 'Retail Trader', abbr: 'RT', color: '#F87171', role: '散户交易者' },
}

// ── Host 帖子 ──
export const HOST_POST = {
  id: 'R4_financial',
  round: 4,
  topic: 'financial',
  title: 'NEXTGEN AI CHIP INC. — S-1 Financial Analysis',
  content:
    '2025 年全年营收 $1.2B，同比增长 45%。毛利率 62.3%，净利率 18.1%。' +
    'R&D 支出占比 28%，高于行业均值。现金储备 $890M，短期债务压力可控。' +
    '关注点：客户集中度高——前 5 大客户贡献 73% 营收。',
  timestamp: '14:32:08',
}

// ── Phase 1 评论 ──
const phase1Comments = [
  {
    id: 'c001',
    agentId: 3847,
    agentType: 'INST',
    content: '毛利率 62.3% 在芯片设计行业属于第一梯队。但客户集中度 73% 是个隐雷，一旦大客户砍单，现金流会断崖。建议关注 Q1 2026 的订单确认情况。',
    temp: 0.82,
    sentiment: 'bull',
    phase: 1,
    timestamp: '14:33:12',
    children: [],
  },
  {
    id: 'c002',
    agentId: 1205,
    agentType: 'RETAIL',
    content: '[MASKED] 营收看起来***不错***，但是具体数字看不太清…… 感觉这家公司应该还行？涨了不少了不知道能不能追。',
    temp: 0.65,
    sentiment: 'bull',
    phase: 1,
    timestamp: '14:33:45',
    isMasked: true,
    children: [],
  },
  {
    id: 'c003',
    agentId: 8821,
    agentType: 'NORMAL',
    content: '我只知道他们的芯片好像挺厉害的？新闻上说被很多数据中心用了。但财务这块我完全不懂。',
    temp: 0.41,
    sentiment: 'neutral',
    phase: 1,
    timestamp: '14:34:02',
    children: [],
  },
  {
    id: 'c004',
    agentId: 5590,
    agentType: 'INST',
    content: '净利率 18% 在高增长期居然能做到，说明成本控制远好于预期。但我更关心的是出口管制的影响——如果 Q3 新政落地，海外营收直接砍半。机构应该会在 IPO 定价时 discount 这个风险。',
    temp: 0.91,
    sentiment: 'bear',
    phase: 1,
    timestamp: '14:34:30',
    children: [],
  },
  {
    id: 'c005',
    agentId: 2133,
    agentType: 'RETAIL',
    content: '[MASKED] 看了半天，数字全是模糊的…… 但评论区机构大佬似乎都在讨论风险？我有点慌了。',
    temp: 0.73,
    sentiment: 'bear',
    phase: 1,
    timestamp: '14:35:01',
    isMasked: true,
    children: [],
  },
  {
    id: 'c006',
    agentId: 7764,
    agentType: 'NORMAL',
    content: '这个芯片能用在手机上吗？感觉 AI 芯片概念挺火的，身边朋友都在讨论。',
    temp: 0.33,
    sentiment: 'neutral',
    phase: 1,
    timestamp: '14:35:22',
    children: [],
  },
  {
    id: 'c007',
    agentId: 4401,
    agentType: 'INST',
    content: 'R&D 28% 是在烧钱买未来。如果下一代架构能从 5nm 跳到 3nm，护城河会极深。我倾向于长线建仓，但 IPO 当天大概率被散户炒高，等回调。',
    temp: 0.88,
    sentiment: 'bull',
    phase: 1,
    timestamp: '14:35:55',
    children: [],
  },
]

// ── Phase 2 二级评论 ──
const phase2Comments = [
  {
    id: 'c101',
    agentId: 2133,
    agentType: 'RETAIL',
    parentId: 'c001',
    content: '[MASKED] 大佬说的客户集中度是什么意思？73% 是高还是低啊？有点想跑了……',
    temp: 0.68,
    sentiment: 'bear',
    phase: 2,
    timestamp: '14:37:10',
    isMasked: true,
    children: [],
  },
  {
    id: 'c102',
    agentId: 4401,
    agentType: 'INST',
    parentId: 'c004',
    content: '同意出口管制的判断。但如果管制只针对高端 GPU，NEXTGEN 的边缘计算芯片反而可能受益——国产替代逻辑。两面看。',
    temp: 0.85,
    sentiment: 'bull',
    phase: 2,
    timestamp: '14:37:44',
    children: [],
  },
  {
    id: 'c103',
    agentId: 1205,
    agentType: 'RETAIL',
    parentId: 'c007',
    content: '[MASKED] 3nm 技术迭代这种事我不太懂…… 但如果机构都看好长线，我也先少买一点试试水？',
    temp: 0.59,
    sentiment: 'bull',
    phase: 2,
    timestamp: '14:38:15',
    isMasked: true,
    children: [],
  },
  {
    id: 'c104',
    agentId: 8821,
    agentType: 'NORMAL',
    parentId: 'c003',
    content: '回复自己——刚去搜了一下，好像营收确实涨了不少。但评论区吵成这样，到底是好还是不好？',
    temp: 0.45,
    sentiment: 'neutral',
    phase: 2,
    timestamp: '14:38:50',
    children: [],
  },
  {
    id: 'c105',
    agentId: 3847,
    agentType: 'INST',
    parentId: 'c005',
    content: '散户朋友，建议不要因为"评论区氛围"做决策。你看到的信息本身就是不完整的——先搞清楚基本面再操作。这不是赌场。',
    temp: 0.77,
    sentiment: 'neutral',
    phase: 2,
    timestamp: '14:39:20',
    children: [],
  },
]

// ── Phase 3 三级评论 ──
const phase3Comments = [
  {
    id: 'c201',
    agentId: 1205,
    agentType: 'RETAIL',
    parentId: 'c101',
    content: '[MASKED] 我也不懂，但看到好多人说"跑"，我决定先观望不动了。这市场太吓人。',
    temp: 0.55,
    sentiment: 'bear',
    phase: 3,
    timestamp: '14:41:05',
    isMasked: true,
  },
  {
    id: 'c202',
    agentId: 5590,
    agentType: 'INST',
    parentId: 'c102',
    content: '国产替代逻辑成立但落地周期长，短期两年内看不到实质营收贡献。别给散户画饼。',
    temp: 0.90,
    sentiment: 'bear',
    phase: 3,
    timestamp: '14:41:33',
  },
  {
    id: 'c203',
    agentId: 7764,
    agentType: 'NORMAL',
    parentId: 'c105',
    content: '说得轻松，普通人哪有时间研究基本面…… 你们机构天天盯着看当然不一样。',
    temp: 0.38,
    sentiment: 'neutral',
    phase: 3,
    timestamp: '14:42:01',
  },
]

// ── 组装嵌套结构 ──
function assembleComments() {
  const p1 = phase1Comments.map(c => ({ ...c, children: [] }))
  const p1Map = Object.fromEntries(p1.map(c => [c.id, c]))

  for (const c2 of phase2Comments) {
    const parent = p1Map[c2.parentId]
    if (parent) parent.children.push({ ...c2, children: [] })
  }

  for (const c3 of phase3Comments) {
    for (const c1 of p1) {
      const target = c1.children.find(c => c.id === c3.parentId)
      if (target) { target.children.push(c3); break }
    }
  }

  return p1
}

export const THREAD_COMMENTS = assembleComments()

// ── K 线数据 (模拟 50 ticks) ──
function generateKLineData() {
  const data = []
  let price = 130
  const events = {
    8: 'Phase 1: 机构看涨情绪主导',
    15: 'Host 发布财务帖 → 市场暂停',
    22: 'Phase 2: 散户集体恐慌跟帖',
    30: '流动性收缩恐慌因子触发',
    38: 'Phase 3: 机构分歧加剧',
    45: '订单簿匹配完成 → 价格修正',
  }

  for (let i = 0; i < 50; i++) {
    const volatility = (Math.random() - 0.48) * 6
    const open = price
    const close = +(price + volatility).toFixed(2)
    const high = +(Math.max(open, close) + Math.random() * 2).toFixed(2)
    const low = +(Math.min(open, close) - Math.random() * 2).toFixed(2)
    const volume = Math.floor(50000 + Math.random() * 150000)
    price = close

    data.push({
      time: i,
      open, high, low, close, volume,
      event: events[i] || null,
    })
  }
  return data
}

export const KLINE_DATA = generateKLineData()

// ── 订单簿 ──
function generateOrderBook() {
  const mid = 124.50
  const bids = []
  const asks = []
  for (let i = 0; i < 15; i++) {
    bids.push({
      price: +(mid - 0.05 * (i + 1)).toFixed(2),
      size: Math.floor(100 + Math.random() * 2000),
      total: 0,
    })
    asks.push({
      price: +(mid + 0.05 * (i + 1)).toFixed(2),
      size: Math.floor(100 + Math.random() * 2000),
      total: 0,
    })
  }
  // cumulative totals
  let bt = 0, at = 0
  for (const b of bids) { bt += b.size; b.total = bt }
  for (const a of asks) { at += a.size; a.total = at }
  return { bids, asks, spread: +(asks[0].price - bids[0].price).toFixed(2) }
}

export const ORDER_BOOK = generateOrderBook()

// ── 10,000 Agent 情绪矩阵 ──
export function generatePopulationGrid(rows = 100, cols = 100) {
  const grid = []
  for (let r = 0; r < rows; r++) {
    const row = []
    for (let c = 0; c < cols; c++) {
      const rand = Math.random()
      // 40% bull, 35% bear, 25% neutral — biased toward fear in Phase 2
      if (rand < 0.35) row.push('bull')
      else if (rand < 0.70) row.push('bear')
      else row.push('neutral')
    }
    grid.push(row)
  }
  return grid
}

// ── 系统日志 ──
export const SYSTEM_LOGS = [
  { time: '14:30:00', level: 'SYS', msg: 'ForumModel initialized. 10,000 agents online.' },
  { time: '14:30:01', level: 'SYS', msg: 'Macro-Round 4/12 started.' },
  { time: '14:32:08', level: 'HOST', msg: 'Thread R4_financial published → Financial Analysis' },
  { time: '14:32:09', level: 'MKT', msg: 'Market halted. Processing new information...' },
  { time: '14:33:00', level: 'LLM', msg: 'Inference batch dispatched → 10,000 agents × Phase 1' },
  { time: '14:36:00', level: 'SYS', msg: 'Phase 1 complete. 1,420 comments collected. Snapshot generated.' },
  { time: '14:36:01', level: 'MKT', msg: 'Order book matching paused...' },
  { time: '14:37:00', level: 'LLM', msg: 'Phase 2 sub-commenting active. Information cascading...' },
  { time: '14:39:30', level: 'SYS', msg: 'Phase 2 complete. 890 sub-comments. Snapshot generated.' },
  { time: '14:39:31', level: 'MKT', msg: '⚠ Volatility Alert! Liquidity shifting...' },
  { time: '14:40:00', level: 'LLM', msg: 'Phase 3 final wave. 420 agents responding...' },
  { time: '14:42:10', level: 'SYS', msg: 'Thread R4_financial locked. Sentiment tensor aggregated.' },
  { time: '14:42:11', level: 'MKT', msg: 'Matching orders... 12,400 trades completed.' },
  { time: '14:42:15', level: 'MKT', msg: 'Tick updated. Price: $124.50 (-4.2%)' },
  { time: '14:42:16', level: 'SYS', msg: 'Awaiting next thread → R4_policy' },
]

// ── 状态机文案 ──
export const STATE_MESSAGES = {
  forum: [
    'System initializing...',
    'Host Agent is compiling Prospectus (Sec 2: Financials)...',
    'LLM Inference running... Generating Phase 1 comments (1,420 Agents responded)',
    'Sub-commenting active. Information cascading...',
    'Thread locked. Aggregating final sentiment tensor.',
    'Awaiting next round...',
  ],
  market: [
    'Engine Idle',
    'Market halted. Processing new information...',
    'Order Book matching paused...',
    'Volatility Alert! Liquidity shifting...',
    'Matching orders... 12,400 trades completed.',
    'Updating Chart. Next Tick rendering...',
  ],
}

import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/* ──────────────────────────────────────────────────────────────────────── */
/* Clean SVG Icon Components                                                */
/* ──────────────────────────────────────────────────────────────────────── */
const AskIcon = ({ className = "w-8 h-8" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M8 10h.01M12 10h.01M16 10h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
  </svg>
)

const CompareIcon = ({ className = "w-8 h-8" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
)

const IdeasIcon = ({ className = "w-8 h-8" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
  </svg>
)

const RecommendIcon = ({ className = "w-8 h-8" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
  </svg>
)

const DocumentIcon = ({ className = "w-5 h-5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
  </svg>
)

const UserAvatarIcon = ({ className = "w-4 h-4" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
  </svg>
)

const AiAvatarIcon = ({ className = "w-4 h-4" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
  </svg>
)

const CalendarIcon = ({ className = "w-4 h-4" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
  </svg>
)

const LinkIcon = ({ className = "w-3.5 h-3.5" }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
  </svg>
)

const LoadingSpinner = ({ className = "w-4 h-4" }) => (
  <svg className={`animate-spin ${className}`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
  </svg>
)

const CARDS = [
  {
    id: 'ask',
    title: 'Ask Questions',
    icon: <AskIcon className="w-8 h-8 text-cyan-400" />,
    smallIcon: <AskIcon className="w-5 h-5 text-cyan-400" />,
    desc: 'Chat with uploaded research papers using semantic AI understanding.'
  },
  {
    id: 'compare',
    title: 'Compare Papers',
    icon: <CompareIcon className="w-8 h-8 text-purple-400" />,
    smallIcon: <CompareIcon className="w-5 h-5 text-purple-400" />,
    desc: 'Detect contradictions, similarities, and technical differences.'
  },
  {
    id: 'ideas',
    title: 'Research Ideas',
    icon: <IdeasIcon className="w-8 h-8 text-emerald-400" />,
    smallIcon: <IdeasIcon className="w-5 h-5 text-emerald-400" />,
    desc: 'Generate novel hybrid AI research ideas from uploaded papers.'
  },
  {
    id: 'recommend',
    title: 'Related Papers',
    icon: <RecommendIcon className="w-8 h-8 text-orange-400" />,
    smallIcon: <RecommendIcon className="w-5 h-5 text-orange-400" />,
    desc: 'Discover research papers related to your uploaded documents.'
  }
]

const PROMPT_SUGGESTIONS = [
  'Summarize key methodology',
  'What datasets were used?',
  'What are the main limitations?',
  'Explain the core algorithms'
]

/* ──────────────────────────────────────────────────────────────────────── */
/* Custom Markdown Typography Components for AI Responses                 */
/* ──────────────────────────────────────────────────────────────────────── */
const markdownComponents = {
  h1: ({ children }) => (
    <h1 className="text-2xl font-bold text-cyan-300 mt-6 mb-3 border-b border-slate-700 pb-2">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-xl font-bold text-cyan-300 mt-5 mb-3">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-lg font-semibold text-cyan-400 mt-5 mb-2">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="text-base font-semibold text-cyan-500 mt-4 mb-2">{children}</h4>
  ),
  p: ({ children }) => (
    <p className="mb-3 leading-relaxed text-slate-200">{children}</p>
  ),
  ul: ({ children }) => (
    <ul className="list-disc pl-5 space-y-1.5 my-3 text-slate-200">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal pl-5 space-y-1.5 my-3 text-slate-200">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="leading-relaxed text-slate-300">{children}</li>
  ),
  strong: ({ children }) => (
    <strong className="text-cyan-300 font-semibold">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="text-slate-400 italic">{children}</em>
  ),
  code: ({ children }) => (
    <code className="bg-slate-800 text-cyan-200 px-1.5 py-0.5 rounded text-xs font-mono">{children}</code>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-4 border-cyan-500/50 pl-4 my-3 text-slate-400 italic">{children}</blockquote>
  ),
  table: ({ children }) => (
    <div className="overflow-x-auto my-4">
      <table className="min-w-full border-collapse text-sm text-slate-300">{children}</table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-slate-700 bg-slate-800/80 px-3 py-2 text-left text-cyan-300 font-semibold text-xs uppercase tracking-wide">{children}</th>
  ),
  td: ({ children }) => (
    <td className="border border-slate-700 px-3 py-2 text-slate-300">{children}</td>
  ),
}

function App() {
  const [activeCard, setActiveCard] = useState(null)
  const [question, setQuestion] = useState("")
  const [chatMessages, setChatMessages] = useState([
    {
      id: 1,
      sender: 'ai',
      text: "Hello! I'm your AI Research Co-Pilot. Ask me anything about your uploaded research papers!",
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ])
  const [isAiThinking, setIsAiThinking] = useState(false)
  const [comparison, setComparison] = useState("")
  const [isComparing, setIsComparing] = useState(false)
  const [ideas, setIdeas] = useState("")
  const [isGeneratingIdeas, setIsGeneratingIdeas] = useState(false)
  const [recommendations, setRecommendations] = useState([])
  const [isRecommending, setIsRecommending] = useState(false)
  const [uploadMessages, setUploadMessages] = useState([])
  const [paperList, setPaperList] = useState([])

  const chatEndRef = useRef(null)

  const fetchPapers = async () => {
    try {
      const response = await axios.get('http://127.0.0.1:8000/papers')
      if (response.data && response.data.papers) {
        setPaperList(
          response.data.papers.map((p) => ({
            id: p.paper_id,
            name: p.name
          }))
        )
      }
    } catch (error) {
      console.error('Error fetching papers:', error)
    }
  }

  useEffect(() => {
    fetchPapers()
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatMessages, isAiThinking])

  const askAI = async (queryToAsk) => {
    const finalQuestion = queryToAsk || question
    if (!finalQuestion.trim()) return

    const userMessage = {
      id: Date.now(),
      sender: 'user',
      text: finalQuestion,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }

    setChatMessages((prev) => [...prev, userMessage])
    if (!queryToAsk) setQuestion("")
    setIsAiThinking(true)

    try {
      const response = await axios.get('http://127.0.0.1:8000/ask', {
        params: { question: finalQuestion }
      })

      const aiMessage = {
        id: Date.now() + 1,
        sender: 'ai',
        text: response.data.answer || "No response received.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }

      setChatMessages((prev) => [...prev, aiMessage])
    } catch (error) {
      console.error(error)
      setChatMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: 'ai',
          text: "Connection error: Could not fetch answer from backend.",
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ])
    } finally {
      setIsAiThinking(false)
    }
  }

  const compareAI = async () => {
    setIsComparing(true)
    try {
      const response = await axios.get('http://127.0.0.1:8000/compare')
      setComparison(response.data.comparison)
    } catch (error) {
      console.error(error)
      setComparison("Failed to load comparison.")
    } finally {
      setIsComparing(false)
    }
  }

  const generateIdeas = async () => {
    setIsGeneratingIdeas(true)
    try {
      const response = await axios.get('http://127.0.0.1:8000/ideas')
      setIdeas(response.data.ideas)
    } catch (error) {
      console.error(error)
      setIdeas("Failed to generate research ideas.")
    } finally {
      setIsGeneratingIdeas(false)
    }
  }

  const recommendPapers = async () => {
    setIsRecommending(true)
    try {
      const response = await axios.get('http://127.0.0.1:8000/recommend', { timeout: 30000 })
      if (response.data && Array.isArray(response.data.papers)) {
        setRecommendations(response.data.papers)
      } else {
        setRecommendations([
          {
            error: "Unable to fetch related papers right now. Please try again.",
            title: "Unable to fetch related papers right now. Please try again.",
            relevance: "Unable to fetch related papers right now. Please try again."
          }
        ])
      }
    } catch (error) {
      console.error('Error fetching recommendations:', error)
      setRecommendations([
        {
          error: "Unable to fetch related papers right now. Please try again.",
          title: "Unable to fetch related papers right now. Please try again.",
          relevance: "Unable to fetch related papers right now. Please try again."
        }
      ])
    } finally {
      setIsRecommending(false)
    }
  }

  const uploadPDF = async (file) => {
    if (!file) return

    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await axios.post('http://127.0.0.1:8000/upload', formData)

      setUploadMessages((prev) => [...prev, response.data.message])

      if (response.data && response.data.papers) {
        setPaperList(
          response.data.papers.map((p) => ({
            id: p.paper_id,
            name: p.name
          }))
        )
      } else {
        fetchPapers()
      }
    } catch (error) {
      console.error(error)
    }
  }

  const deletePaper = async (paperName) => {
    try {
      const response = await axios.delete(
        `http://127.0.0.1:8000/paper/${encodeURIComponent(paperName)}`
      )
      if (response.data && response.data.papers) {
        setPaperList(
          response.data.papers.map((p) => ({
            id: p.paper_id,
            name: p.name
          }))
        )
      } else {
        fetchPapers()
      }
      setUploadMessages((prev) => prev.filter((msg) => !msg.includes(paperName)))
    } catch (error) {
      console.error('Failed to remove paper:', error)
      alert("Failed to remove paper.")
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      askAI()
    }
  }

  const activeCardObj = CARDS.find((c) => c.id === activeCard)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-black text-white p-6 md:p-8 selection:bg-cyan-500 selection:text-black">
      <div className="max-w-7xl mx-auto">
        
        {/* ========================================================= */}
        {/* OVERVIEW MODE (When no card is actively focused)           */}
        {/* ========================================================= */}
        {activeCard === null ? (
          <div className="transition-all duration-700">
            {/* Title Header */}
            <h1 className="text-5xl md:text-6xl font-extrabold text-center mb-4 bg-gradient-to-r from-cyan-400 via-teal-300 to-blue-500 text-transparent bg-clip-text drop-shadow-md">
              AI Research Co-Pilot
            </h1>

            <p className="text-center text-slate-400 text-lg mb-10 max-w-2xl mx-auto">
              Interactive Multi-Paper Research Workspace powered by RAG + LLMs
            </p>

            {/* Upload Section */}
            <div className="bg-slate-900/60 border border-slate-800 backdrop-blur-xl rounded-3xl p-8 mb-12 shadow-2xl">
              <h2 className="text-2xl font-semibold mb-6 flex items-center gap-3 text-cyan-300">
                <DocumentIcon className="w-6 h-6 text-cyan-400" />
                <span>Upload Research Papers</span>
              </h2>

              <div className="border-2 border-dashed border-cyan-500/40 hover:border-cyan-400 rounded-2xl p-10 text-center hover:bg-cyan-500/5 transition duration-300">
                <input
                  type="file"
                  accept=".pdf"
                  multiple
                  onChange={(e) => {
                    const files = Array.from(e.target.files)
                    files.forEach((file) => uploadPDF(file))
                  }}
                  className="text-white file:mr-4 file:py-2 file:px-6 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-cyan-500/20 file:text-cyan-300 hover:file:bg-cyan-500/30 cursor-pointer"
                />

                <p className="text-slate-400 mt-4 text-sm">
                  Upload research papers (PDF) for AI indexing & analysis
                </p>

                {uploadMessages.length > 0 && (
                  <div className="mt-4 space-y-2">
                    {uploadMessages.map((msg, index) => (
                      <p key={index} className="text-cyan-400 text-sm">
                        {msg}
                      </p>
                    ))}
                  </div>
                )}

                {paperList.length > 0 && (
                  <div className="mt-6 bg-slate-950/80 rounded-2xl p-5 border border-slate-800 text-left">
                    <h3 className="text-base font-semibold text-cyan-300 mb-3">
                      Uploaded Paper Mapping
                    </h3>

                    <div className="space-y-3">
                      {paperList.map((paper) => (
                        <div
                          key={paper.id}
                          className="flex items-center justify-between bg-slate-900/80 px-4 py-3 rounded-xl border border-slate-800"
                        >
                          <div className="text-slate-300 text-sm break-all pr-4 flex items-center gap-2">
                            <DocumentIcon className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                            <span>Paper {paper.id} → {paper.name}</span>
                          </div>

                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              deletePaper(paper.name)
                            }}
                            className="bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/30 px-3 py-1.5 rounded-lg text-xs font-semibold transition duration-300 whitespace-nowrap cursor-pointer"
                          >
                            Remove
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* 4 Feature Cards arranged side by side */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {CARDS.map((card) => (
                <div
                  key={card.id}
                  onClick={() => setActiveCard(card.id)}
                  className="bg-slate-900/60 border border-slate-800/80 hover:border-cyan-400/50 backdrop-blur-xl rounded-3xl p-6 shadow-xl cursor-pointer transition-all duration-300 hover:-translate-y-1.5 hover:shadow-cyan-500/10 flex flex-col justify-between group"
                >
                  <div>
                    <div className="mb-4 group-hover:scale-110 transition duration-300">
                      {card.icon}
                    </div>
                    <h3 className="text-xl font-bold text-white mb-2 group-hover:text-cyan-300 transition duration-300">
                      {card.title}
                    </h3>
                    <p className="text-slate-400 text-sm leading-relaxed">
                      {card.desc}
                    </p>
                  </div>

                  <div className="mt-6 flex items-center gap-2 text-cyan-400 text-sm font-semibold group-hover:translate-x-1 transition duration-300">
                    <span>Open Tool</span>
                    <span>&rarr;</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          /* ========================================================= */
          /* FOCUS MODE: Split View (Sidebar LEFT, Tool RIGHT)         */
          /* ========================================================= */
          <div className="flex flex-col lg:flex-row gap-6 min-h-[80vh] items-stretch transition-all duration-500">
            
            {/* LEFT SIDEBAR NAVIGATION */}
            <div className="w-full lg:w-80 flex-shrink-0 bg-slate-900/80 border border-slate-800 backdrop-blur-xl rounded-3xl p-6 shadow-2xl flex flex-col justify-between">
              <div>
                {/* Back / Close button */}
                <button
                  onClick={() => setActiveCard(null)}
                  className="w-full mb-6 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-4 py-3 rounded-2xl flex items-center justify-between transition duration-300 font-semibold cursor-pointer text-sm"
                >
                  <span className="flex items-center gap-2">
                    <span>&larr;</span>
                    <span>Close Overview</span>
                  </span>
                  <span className="text-slate-400 text-xs">X</span>
                </button>

                <h3 className="text-xs uppercase tracking-wider font-semibold text-slate-500 mb-4 px-2">
                  Tools & Assistants
                </h3>

                {/* Vertical Sidebar List */}
                <div className="space-y-3">
                  {CARDS.map((card) => {
                    const isActive = activeCard === card.id
                    return (
                      <div
                        key={card.id}
                        onClick={() => setActiveCard(card.id)}
                        className={`
                          p-4 rounded-2xl cursor-pointer border transition-all duration-300 flex items-center gap-4
                          ${
                            isActive
                              ? 'bg-cyan-500/15 border-cyan-400 text-white shadow-[0_0_20px_rgba(6,182,212,0.25)]'
                              : 'bg-slate-950/40 border-slate-800/80 hover:border-slate-700 text-slate-400 hover:text-slate-200'
                          }
                        `}
                      >
                        <span className="flex-shrink-0">{card.smallIcon}</span>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-semibold text-sm truncate">
                            {card.title}
                          </h4>
                          <p className="text-xs text-slate-500 truncate">
                            {card.desc}
                          </p>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Sidebar Footer info */}
              <div className="mt-8 pt-4 border-t border-slate-800/80 text-xs text-slate-500 flex items-center justify-between">
                <span>Active Papers: {paperList.length}</span>
                <span className="text-cyan-400">RAG Ready</span>
              </div>
            </div>

            {/* RIGHT MAIN EXPANDED WORKSPACE */}
            <div className="flex-1 bg-slate-900/60 border border-slate-800 backdrop-blur-2xl rounded-3xl p-6 md:p-8 shadow-2xl flex flex-col overflow-hidden min-h-[700px]">
              
              {/* Top Title Bar of Tool */}
              <div className="flex items-center justify-between pb-6 border-b border-slate-800/80 mb-6">
                <div>
                  <h2 className="text-3xl font-bold flex items-center gap-3 text-white">
                    <span>{activeCardObj?.smallIcon}</span>
                    <span>{activeCardObj?.title}</span>
                  </h2>
                  <p className="text-slate-400 text-sm mt-1">
                    {activeCardObj?.desc}
                  </p>
                </div>

                <button
                  onClick={() => setActiveCard(null)}
                  className="bg-slate-800/80 hover:bg-slate-700 text-slate-300 w-9 h-9 rounded-full flex items-center justify-center transition duration-200 cursor-pointer border border-slate-700 text-sm font-semibold"
                  title="Close and return to overview"
                >
                  X
                </button>
              </div>

              {/* TOOL CONTENT RENDERING */}

              {/* ------------------------------------------------------------- */}
              {/* TOOL 1: ASK QUESTIONS (INTERACTIVE CONVERSATIONAL AI CHAT)    */}
              {/* ------------------------------------------------------------- */}
              {activeCard === 'ask' && (
                <div className="flex-1 flex flex-col min-h-0">
                  {/* Chat Messages Stream */}
                  <div className="flex-1 overflow-y-auto space-y-4 pr-2 mb-4 min-h-[400px] max-h-[500px]">
                    {chatMessages.map((msg) => (
                      <div
                        key={msg.id}
                        className={`flex gap-3 ${
                          msg.sender === 'user' ? 'justify-end' : 'justify-start'
                        }`}
                      >
                        {msg.sender === 'ai' && (
                          <div className="w-8 h-8 rounded-full bg-cyan-500/20 border border-cyan-400/50 flex items-center justify-center flex-shrink-0 mt-1">
                            <AiAvatarIcon className="w-4 h-4 text-cyan-300" />
                          </div>
                        )}

                        <div
                          className={`max-w-[80%] rounded-2xl p-4 text-sm leading-relaxed ${
                            msg.sender === 'user'
                              ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white rounded-br-none shadow-lg'
                              : 'bg-slate-950/80 border border-slate-800 text-slate-200 rounded-bl-none shadow-md'
                          }`}
                        >
                          <div className="prose prose-invert max-w-none text-sm">
                            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                              {msg.text}
                            </ReactMarkdown>
                          </div>
                          <span
                            className={`block text-[10px] mt-2 ${
                              msg.sender === 'user' ? 'text-cyan-200 text-right' : 'text-slate-500 text-left'
                            }`}
                          >
                            {msg.timestamp}
                          </span>
                        </div>

                        {msg.sender === 'user' && (
                          <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0 mt-1">
                            <UserAvatarIcon className="w-4 h-4 text-white" />
                          </div>
                        )}
                      </div>
                    ))}

                    {/* AI Thinking Loading Animation */}
                    {isAiThinking && (
                      <div className="flex gap-3 justify-start items-center">
                        <div className="w-8 h-8 rounded-full bg-cyan-500/20 border border-cyan-400/50 flex items-center justify-center flex-shrink-0 animate-pulse">
                          <AiAvatarIcon className="w-4 h-4 text-cyan-300" />
                        </div>
                        <div className="bg-slate-950/80 border border-slate-800 rounded-2xl px-5 py-3 text-cyan-300 text-xs flex items-center gap-2">
                          <LoadingSpinner className="w-4 h-4 text-cyan-400" />
                          <span>AI is reading paper context & formulation...</span>
                        </div>
                      </div>
                    )}

                    <div ref={chatEndRef} />
                  </div>

                  {/* Suggestion Chips */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {PROMPT_SUGGESTIONS.map((chip, index) => (
                      <button
                        key={index}
                        onClick={() => askAI(chip)}
                        className="bg-slate-800/60 hover:bg-cyan-500/20 hover:border-cyan-400 text-slate-300 hover:text-cyan-200 border border-slate-700/80 px-3 py-1.5 rounded-full text-xs transition duration-200 cursor-pointer"
                      >
                        {chip}
                      </button>
                    ))}
                  </div>

                  {/* Chat Input Bar */}
                  <div className="flex gap-3 items-center bg-slate-950/90 border border-slate-800 rounded-2xl p-2">
                    <textarea
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder="Ask anything about your research papers... (Press Enter to send)"
                      rows={2}
                      className="flex-1 bg-transparent text-white placeholder-slate-500 px-3 py-2 text-sm outline-none resize-none"
                    />

                    <button
                      onClick={() => askAI()}
                      disabled={isAiThinking || !question.trim()}
                      className="bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 disabled:opacity-40 text-black font-bold px-6 py-3 rounded-xl transition duration-300 cursor-pointer text-sm whitespace-nowrap"
                    >
                      {isAiThinking ? 'Thinking...' : 'Ask AI'}
                    </button>
                  </div>
                </div>
              )}

              {/* ------------------------------------------------------------- */}
              {/* TOOL 2: COMPARE PAPERS                                        */}
              {/* ------------------------------------------------------------- */}
              {activeCard === 'compare' && (
                <div className="flex-1 flex flex-col">
                  <div className="mb-6 flex justify-between items-center">
                    <p className="text-slate-300 text-sm">
                      Compare methodologies, architectures, datasets, and contradictions across uploaded papers.
                    </p>
                    <button
                      onClick={compareAI}
                      disabled={isComparing}
                      className="bg-purple-600 hover:bg-purple-500 disabled:opacity-50 text-white font-semibold px-6 py-3 rounded-xl transition duration-300 cursor-pointer text-sm flex items-center gap-2"
                    >
                      {isComparing ? 'Comparing...' : 'Run Paper Comparison'}
                    </button>
                  </div>

                  <div className="flex-1 bg-slate-950/80 rounded-2xl p-6 border border-slate-800 overflow-y-auto max-h-[550px]">
                    <div className="prose prose-invert max-w-none text-slate-300 text-sm leading-relaxed">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                        {comparison || 'Click "Run Paper Comparison" above to generate structured technical comparison results across your uploaded research papers.'}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              )}

              {/* ------------------------------------------------------------- */}
              {/* TOOL 3: RESEARCH IDEAS                                        */}
              {/* ------------------------------------------------------------- */}
              {activeCard === 'ideas' && (
                <div className="flex-1 flex flex-col">
                  <div className="mb-6 flex justify-between items-center">
                    <p className="text-slate-300 text-sm">
                      Generate novel hybrid research concepts and system architecture ideas based on paper findings.
                    </p>
                    <button
                      onClick={generateIdeas}
                      disabled={isGeneratingIdeas}
                      className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-black font-semibold px-6 py-3 rounded-xl transition duration-300 cursor-pointer text-sm flex items-center gap-2"
                    >
                      {isGeneratingIdeas ? 'Inventing...' : 'Generate Research Ideas'}
                    </button>
                  </div>

                  <div className="flex-1 bg-slate-950/80 rounded-2xl p-6 border border-slate-800 overflow-y-auto max-h-[550px]">
                    <div className="prose prose-invert max-w-none text-slate-300 text-sm leading-relaxed">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                        {ideas || 'Click "Generate Research Ideas" above to synthesise novel research hybrid concepts.'}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              )}

              {/* ------------------------------------------------------------- */}
              {/* TOOL 4: RELATED PAPERS                                        */}
              {/* ------------------------------------------------------------- */}
              {activeCard === 'recommend' && (
                <div className="flex-1 flex flex-col">
                  <div className="mb-6 flex justify-between items-center">
                    <p className="text-slate-300 text-sm">
                      Discover closely related literature and search queries tailored to your uploaded papers.
                    </p>
                    <button
                      onClick={recommendPapers}
                      disabled={isRecommending}
                      className="bg-orange-500 hover:bg-orange-400 disabled:opacity-50 text-black font-semibold px-6 py-3 rounded-xl transition duration-300 cursor-pointer text-sm flex items-center gap-2"
                    >
                      {isRecommending ? 'Searching...' : 'Recommend Related Papers'}
                    </button>
                  </div>

                  <div className="flex-1 bg-slate-950/80 rounded-2xl p-6 border border-slate-800 overflow-y-auto max-h-[550px]">
                    {recommendations.length > 0 ? (
                      <div className="space-y-4">
                        {recommendations.map((paper, index) => {
                          if (paper.error || paper.title?.includes("Unable to fetch") || paper.relevance?.includes("Unable to fetch")) {
                            return (
                              <div key={index} className="bg-red-950/40 p-5 rounded-xl border border-red-800 text-red-300 text-sm flex items-center gap-2">
                                <span>Unable to fetch related papers right now. Please try again.</span>
                              </div>
                            )
                          }

                          if (paper.reason && !paper.authors && !paper.relevance) {
                            return (
                              <div key={index} className="bg-slate-900/90 p-5 rounded-xl border border-slate-800">
                                <h4 className="font-semibold text-orange-300 mb-2 text-base flex items-center gap-2">
                                  <DocumentIcon className="w-5 h-5 text-orange-400 flex-shrink-0" />
                                  <span>{paper.title}</span>
                                </h4>
                                <div className="text-slate-300 text-sm whitespace-pre-wrap">
                                  {paper.reason}
                                </div>
                              </div>
                            )
                          }

                          return (
                            <div
                              key={index}
                              className="bg-slate-900/90 p-5 rounded-xl border border-slate-800 hover:border-slate-700 transition flex flex-col gap-2 text-sm"
                            >
                              <h4 className="font-semibold text-orange-300 text-base leading-snug flex items-start gap-2">
                                <DocumentIcon className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
                                <span>{paper.title}</span>
                              </h4>

                              {paper.authors && (
                                <p className="text-slate-300 flex items-center gap-2 text-xs">
                                  <UserAvatarIcon className="w-3.5 h-3.5 text-slate-400" />
                                  <span className="font-medium text-slate-200">{paper.authors}</span>
                                </p>
                              )}

                              <div className="flex items-center gap-6 text-slate-400 text-xs">
                                {paper.year && (
                                  <span className="flex items-center gap-1.5">
                                    <CalendarIcon className="w-3.5 h-3.5 text-slate-400" />
                                    <span>{paper.year}</span>
                                  </span>
                                )}
                                {paper.venue && (
                                  <span className="flex items-center gap-1.5">
                                    <DocumentIcon className="w-3.5 h-3.5 text-slate-400" />
                                    <span>{paper.venue}</span>
                                  </span>
                                )}
                              </div>

                              {paper.relevance && (
                                <p className="text-slate-300 bg-slate-950/50 p-3 rounded-lg border border-slate-800/80 leading-relaxed text-xs">
                                  <strong className="text-orange-400">Relevance:</strong> {paper.relevance}
                                </p>
                              )}

                              {paper.url ? (
                                <a
                                  href={paper.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1.5 text-orange-400 hover:text-orange-300 font-medium text-xs self-start mt-1 underline transition"
                                >
                                  <LinkIcon className="w-3.5 h-3.5" />
                                  <span>View Paper</span>
                                </a>
                              ) : (
                                <span className="text-slate-500 text-xs mt-1 italic">URL not available</span>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    ) : (
                      <p className="text-slate-400 text-sm">
                        Click "Recommend Related Papers" above to get AI literature suggestions.
                      </p>
                    )}
                  </div>
                </div>
              )}

            </div>
          </div>
        )}

      </div>
    </div>
  )
}

export default App
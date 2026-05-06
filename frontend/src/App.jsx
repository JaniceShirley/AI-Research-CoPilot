import { useState } from 'react'
import axios from 'axios'

function App() {

  const [activeCard, setActiveCard] = useState(null)
  const [question, setQuestion] = useState("")
  const [answer, setAnswer] = useState("")

  const askAI = async () => {

    try {

      const response = await axios.get(
        'http://127.0.0.1:8000/ask',
        {
          params: {
            question: question
          }
        }
      )

      setAnswer(response.data.answer)

    } catch (error) {
      console.error(error)
    }

  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-slate-900 to-black text-white p-8 overflow-hidden">

      <div className="max-w-7xl mx-auto">

        {/* Title */}
        <h1 className="text-6xl font-bold text-center mb-6 bg-gradient-to-r from-cyan-400 to-blue-500 text-transparent bg-clip-text">
           🧠 AI Research Co-Pilot
        </h1>

        <p className="text-center text-gray-400 text-lg mb-14">
          Multi-Paper AI Research Assistant powered by RAG + LLMs
        </p>


        {/* Upload Section */}
        <div className="bg-white/10 border border-white/20 backdrop-blur-lg rounded-3xl p-8 mb-12 shadow-2xl">

          <h2 className="text-3xl font-semibold mb-6">
             📄 Upload Research Papers
          </h2>

          <div className="border-2 border-dashed border-cyan-400 rounded-2xl p-12 text-center hover:bg-cyan-500/10 transition duration-300 cursor-pointer">
            <p className="text-2xl font-medium text-cyan-300">
              Drag & Drop PDFs Here
            </p>

            <p className="text-gray-400 mt-3">
              Upload multiple research papers for AI-powered analysis
            </p>
          </div>

        </div>


        {/* Feature Cards */}
        <div className={`flex gap-8 transition-all duration-500 items-start`}>

          {/* Ask Questions */}
          <div
            onClick={() => setActiveCard(activeCard === 'ask' ? null : 'ask')}
            className={`
              bg-white/10 backdrop-blur-lg border border-white/20 rounded-3xl p-8 shadow-2xl cursor-pointer
              transition-all duration-700 hover:scale-[1.02]
              ${activeCard === 'ask' ? 'flex-[3] min-h-[600px]' : 'flex-1'}
              ${activeCard && activeCard !== 'ask' ? 'opacity-30 scale-90 blur-[1px]' : ''}
            `}
          >
            <h2 className="text-3xl font-semibold mb-4">
              💬 Ask Questions
            </h2>

            <p className="text-gray-300 mb-6">
              Chat with uploaded research papers using semantic AI understanding.
            </p>

            {activeCard === 'ask' && (
              <div className="mt-8 transition-all duration-500">

                <textarea
                  onClick={(e) => e.stopPropagation()}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask anything about your papers..."
                  className="w-full bg-black/40 border border-cyan-400 rounded-2xl p-5 text-white outline-none h-40"
                />

                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    askAI()
                  }}
                  className="mt-5 bg-cyan-500 hover:bg-cyan-400 text-black font-semibold px-8 py-3 rounded-xl transition duration-300"
                >
                  Ask AI
                </button>

                <div className="mt-8 bg-black/40 rounded-2xl p-6 border border-white/10">
                  <h3 className="text-xl font-semibold mb-4 text-cyan-300">
                    AI Response
                  </h3>

                  <p className="text-gray-300 leading-relaxed">
                    {answer || 'Your AI-generated research answer will appear here.'}
                  </p>
                </div>

              </div>
            )}
          </div>


          {/* Compare Papers */}
          <div
            onClick={() => setActiveCard(activeCard === 'compare' ? null : 'compare')}
            className={`
              bg-white/10 backdrop-blur-lg border border-white/20 rounded-3xl p-8 shadow-2xl cursor-pointer
              transition-all duration-700 hover:scale-[1.02]
              ${activeCard === 'compare' ? 'flex-[3] min-h-[600px]' : 'flex-1'}
              ${activeCard && activeCard !== 'compare' ? 'opacity-30 scale-90 blur-[1px]' : ''}
            `}
          >
            <h2 className="text-3xl font-semibold mb-4">
               🔍 Compare Papers
            </h2>

            <p className="text-gray-300 mb-6">
              Detect contradictions, similarities, and technical differences.
            </p>

            {activeCard === 'compare' && (
              <div className="mt-8 transition-all duration-500">

                <button
                  onClick={(e) => e.stopPropagation()}
                  className="bg-purple-500 hover:bg-purple-400 text-white font-semibold px-8 py-3 rounded-xl transition duration-300"
                >
                  Compare Papers
                </button>

                <div className="mt-8 bg-black/40 rounded-2xl p-6 border border-white/10">
                  <h3 className="text-xl font-semibold mb-4 text-purple-300">
                    Comparison Results
                  </h3>

                  <p className="text-gray-300 leading-relaxed">
                    Technical paper comparison results will appear here.
                  </p>
                </div>

              </div>
            )}
          </div>


          {/* Research Ideas */}
          <div
            onClick={() => setActiveCard(activeCard === 'ideas' ? null : 'ideas')}
            className={`
              bg-white/10 backdrop-blur-lg border border-white/20 rounded-3xl p-8 shadow-2xl cursor-pointer
              transition-all duration-700 hover:scale-[1.02]
              ${activeCard === 'ideas' ? 'flex-[3] min-h-[600px]' : 'flex-1'}
              ${activeCard && activeCard !== 'ideas' ? 'opacity-30 scale-90 blur-[1px]' : ''}
            `}
          >
            <h2 className="text-3xl font-semibold mb-4">
               💡 Research Ideas
            </h2>

            <p className="text-gray-300 mb-6">
              Generate novel hybrid AI research ideas from uploaded papers.
            </p>

            {activeCard === 'ideas' && (
              <div className="mt-8 transition-all duration-500">

                <button
                  onClick={(e) => e.stopPropagation()}
                  className="bg-green-500 hover:bg-green-400 text-black font-semibold px-8 py-3 rounded-xl transition duration-300"
                >
                  Generate Ideas
                </button>

                <div className="mt-8 bg-black/40 rounded-2xl p-6 border border-white/10">
                  <h3 className="text-xl font-semibold mb-4 text-green-300">
                    AI Innovation Suggestions
                  </h3>

                  <p className="text-gray-300 leading-relaxed">
                    AI-generated future research ideas will appear here.
                  </p>
                </div>

              </div>
            )}
          </div>

        </div>

      </div>

    </div>
  )
}

export default App
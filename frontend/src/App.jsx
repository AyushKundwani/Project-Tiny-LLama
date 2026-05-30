// ============================================================
// App.jsx — Root React Component
// ============================================================
// What: The root component that sets up the page layout and
//       renders the chatbot interface.
// Why:  Keeps everything organized in one entry point.
// ============================================================

import { useState, useEffect, useRef } from "react";
import "./App.css";

// ─────────────────────────────────────────────
// API Configuration
// ─────────────────────────────────────────────
// What: The base URL of our FastAPI backend.
// Why:  If we change the port, we only update it here.
const API_BASE_URL = "http://localhost:8000";

// ─────────────────────────────────────────────
// Suggested Starter Questions
// ─────────────────────────────────────────────
// What: A list of example questions shown as clickable chips.
// Why:  Helps beginners get started without typing.
const STARTER_QUESTIONS = [
  "What is compound interest?",
  "What is an ETF?",
  "How does a 401k work?",
  "What is dollar cost averaging?",
  "What is diversification?",
  "What is a Roth IRA?",
];

// ─────────────────────────────────────────────
// Main App Component
// ─────────────────────────────────────────────
export default function App() {
  // State: list of chat messages { role: "user"|"bot", text: string }
  const [messages, setMessages] = useState([]);

  // State: current text in the input box
  const [inputText, setInputText] = useState("");

  // State: whether the bot is currently thinking
  const [isLoading, setIsLoading] = useState(false);

  // State: whether the backend is reachable
  const [backendStatus, setBackendStatus] = useState("checking");

  // Ref: points to the bottom of the messages list for auto-scrolling
  const messagesEndRef = useRef(null);

  // ─────────────────────────────────────────────
  // Check Backend Health on Mount
  // ─────────────────────────────────────────────
  // What: Calls /health when the page loads to check if the API is live.
  // Why:  Show the user a clear status badge instead of a confusing error.
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
          setBackendStatus("online");
        } else {
          setBackendStatus("offline");
        }
      } catch {
        setBackendStatus("offline");
      }
    };
    checkHealth();
  }, []); // Empty array = runs once when component first mounts

  // ─────────────────────────────────────────────
  // Auto-scroll to Latest Message
  // ─────────────────────────────────────────────
  // What: Scrolls the chat window to the bottom whenever a new message appears.
  // Why:  So the user always sees the latest message without scrolling manually.
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ─────────────────────────────────────────────
  // Send Message to Backend
  // ─────────────────────────────────────────────
  // What: Sends the user's question to /chat and adds the response to messages.
  // Why:  This is the core function that powers the chatbot.
  const sendMessage = async (question) => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return; // Don't send empty messages

    // Add user message to the chat immediately
    const userMessage = { role: "user", text: trimmedQuestion };
    setMessages((prev) => [...prev, userMessage]);
    setInputText("");
    setIsLoading(true);

    try {
      // POST request to the FastAPI backend
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",  // Tell the server we're sending JSON
        },
        body: JSON.stringify({
          question: trimmedQuestion,
          max_tokens: 200,
        }),
      });

      if (!response.ok) {
        // Server returned an error (e.g. 400, 500)
        const errorData = await response.json();
        throw new Error(errorData.detail || "Server error");
      }

      // Parse the JSON response
      const data = await response.json();

      // Add bot's answer to the chat
      const botMessage = { role: "bot", text: data.answer };
      setMessages((prev) => [...prev, botMessage]);

    } catch (error) {
      // Network error or server error — show error message in chat
      console.error("Error:", error);
      const errorMessage = {
        role: "bot",
        text: `⚠️ Error: ${error.message}. Make sure the backend is running on port 8000.`,
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false); // Always stop loading, even if there's an error
    }
  };

  // ─────────────────────────────────────────────
  // Handle Form Submit (Enter key or button click)
  // ─────────────────────────────────────────────
  const handleSubmit = (e) => {
    e.preventDefault(); // Prevent page reload on form submit
    sendMessage(inputText);
  };

  // ─────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────
  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">₿</span>
            <div>
              <h1>FinBot</h1>
              <p>Powered by TinyLlama + LoRA</p>
            </div>
          </div>
          {/* Backend status badge */}
          <div className={`status-badge ${backendStatus}`}>
            <span className="status-dot" />
            {backendStatus === "online"  && "Model Online"}
            {backendStatus === "offline" && "Backend Offline"}
            {backendStatus === "checking" && "Connecting..."}
          </div>
        </div>
      </header>

      {/* ── Main Chat Area ── */}
      <main className="chat-container">

        {/* Welcome screen shown when there are no messages */}
        {messages.length === 0 && (
          <div className="welcome">
            <div className="welcome-icon">💰</div>
            <h2>Ask me anything about personal finance</h2>
            <p>I'm fine-tuned on financial Q&A data to help you understand money concepts.</p>

            {/* Starter question chips */}
            <div className="starter-questions">
              {STARTER_QUESTIONS.map((q, i) => (
                <button
                  key={i}
                  className="starter-chip"
                  onClick={() => sendMessage(q)}
                  disabled={backendStatus !== "online"}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Message bubbles */}
        <div className="messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`message-row ${msg.role}`}
            >
              {/* Avatar */}
              <div className="avatar">
                {msg.role === "user" ? "👤" : "🤖"}
              </div>
              {/* Text bubble */}
              <div className={`bubble ${msg.role} ${msg.isError ? "error" : ""}`}>
                {msg.text}
              </div>
            </div>
          ))}

          {/* Typing indicator shown while waiting for response */}
          {isLoading && (
            <div className="message-row bot">
              <div className="avatar">🤖</div>
              <div className="bubble bot typing">
                <span />
                <span />
                <span />
              </div>
            </div>
          )}

          {/* Invisible div at the bottom for auto-scrolling */}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* ── Input Bar ── */}
      <footer className="input-area">
        <form className="input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Ask a finance question... e.g. What is a stock?"
            disabled={isLoading || backendStatus !== "online"}
            className="input-box"
            maxLength={500}
          />
          <button
            type="submit"
            disabled={isLoading || !inputText.trim() || backendStatus !== "online"}
            className="send-button"
          >
            {isLoading ? "..." : "Send →"}
          </button>
        </form>
        {backendStatus === "offline" && (
          <p className="offline-warning">
            ⚠️ Backend is offline. Run <code>python main.py</code> in the backend folder.
          </p>
        )}
      </footer>
    </div>
  );
}

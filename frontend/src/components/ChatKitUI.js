import React, { useState, useEffect, useRef } from 'react';

// Simple ChatKit-like component
const ChatKitUI = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: 'Hello! I\'m your AI documentation assistant. How can I help you today?', sender: 'bot', timestamp: new Date() }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Get current language from Docusaurus context or default to 'en'
      // In Docusaurus, we can get the locale from the global context
      let currentLanguage = 'en';
      if (typeof window !== 'undefined' && window.location.pathname.startsWith('/ur/')) {
        currentLanguage = 'ur';
      } else if (typeof window !== 'undefined') {
        // Try to get the locale from the URL or from a global variable if available
        const pathParts = window.location.pathname.split('/');
        if (pathParts.length > 1 && pathParts[1] && pathParts[1] !== 'docs') {
          currentLanguage = pathParts[1]; // This would be 'ur' for /ur/ paths
        }
      }

      // Call backend API to get AI response
      const response = await fetch('http://localhost:8000/api/chatbot/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,
          language: currentLanguage
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const botMessage = {
          id: Date.now() + 1,
          text: data.response || 'I\'m not sure how to help with that. Please try rephrasing your question.',
          sender: 'bot',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      } else {
        const botMessage = {
          id: Date.now() + 1,
          text: 'Sorry, I encountered an error processing your request. Please try again.',
          sender: 'bot',
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
      }
    } catch (error) {
      const botMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I\'m having trouble connecting to the server. Please check your connection and try again.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, botMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
    if (!isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  // Simple color mode detection without hooks (fallback to light mode)
  const getColorMode = () => {
    if (typeof window !== 'undefined') {
      const savedMode = localStorage.getItem('theme');
      if (savedMode) return savedMode;
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'light';
  };

  const colorMode = getColorMode();

  const isDarkMode = colorMode === 'dark';

  return (
    <>
      {/* Floating Chat Button */}
      {!isOpen && (
        <button
          onClick={toggleChat}
          className="chatkit-float-button"
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            backgroundColor: '#667eea',
            color: 'white',
            border: 'none',
            fontSize: '24px',
            cursor: 'pointer',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          💬
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div
          className={`chatkit-window ${colorMode}`}
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '380px',
            height: '500px',
            backgroundColor: isDarkMode ? '#1a202c' : 'white',
            border: `1px solid ${isDarkMode ? '#2d3748' : '#e2e8f0'}`,
            borderRadius: '12px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.15)',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 1000,
            fontFamily: 'system-ui, -apple-system, sans-serif',
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: '16px',
              backgroundColor: isDarkMode ? '#2d3748' : '#f8fafc',
              borderTopLeftRadius: '12px',
              borderTopRightRadius: '12px',
              borderBottom: `1px solid ${isDarkMode ? '#4a5568' : '#e2e8f0'}`,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div style={{ fontWeight: '600', color: isDarkMode ? 'white' : '#1a202c' }}>
              Documentation Assistant
            </div>
            <button
              onClick={toggleChat}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '18px',
                cursor: 'pointer',
                color: isDarkMode ? 'white' : '#1a202c',
                padding: '4px',
                borderRadius: '4px',
              }}
              onMouseEnter={(e) => {
                e.target.style.backgroundColor = isDarkMode ? '#4a5568' : '#e2e8f0';
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = 'transparent';
              }}
            >
              ×
            </button>
          </div>

          {/* Messages */}
          <div
            style={{
              flex: 1,
              padding: '16px',
              overflowY: 'auto',
              backgroundColor: isDarkMode ? '#1a202c' : '#ffffff',
            }}
          >
            {messages.map((message) => (
              <div
                key={message.id}
                style={{
                  marginBottom: '12px',
                  textAlign: message.sender === 'user' ? 'right' : 'left',
                }}
              >
                <div
                  style={{
                    display: 'inline-block',
                    padding: '10px 14px',
                    borderRadius: '18px',
                    backgroundColor: message.sender === 'user'
                      ? (isDarkMode ? '#4299e1' : '#3182ce')
                      : (isDarkMode ? '#2d3748' : '#e2e8f0'),
                    color: message.sender === 'user'
                      ? 'white'
                      : (isDarkMode ? 'white' : '#1a202c'),
                    maxWidth: '80%',
                    wordWrap: 'break-word',
                  }}
                >
                  {message.text}
                </div>
                <div
                  style={{
                    fontSize: '12px',
                    color: isDarkMode ? '#a0aec0' : '#718096',
                    marginTop: '4px',
                  }}
                >
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            ))}
            {isLoading && (
              <div style={{ textAlign: 'left', marginBottom: '12px' }}>
                <div
                  style={{
                    display: 'inline-block',
                    padding: '10px 14px',
                    borderRadius: '18px',
                    backgroundColor: isDarkMode ? '#2d3748' : '#e2e8f0',
                    color: isDarkMode ? 'white' : '#1a202c',
                  }}
                >
                  Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div
            style={{
              padding: '12px',
              backgroundColor: isDarkMode ? '#2d3748' : '#f8fafc',
              borderBottomLeftRadius: '12px',
              borderBottomRightRadius: '12px',
              borderTop: `1px solid ${isDarkMode ? '#4a5568' : '#e2e8f0'}`,
            }}
          >
            <div style={{ display: 'flex', gap: '8px' }}>
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask about the documentation..."
                style={{
                  flex: 1,
                  padding: '12px',
                  borderRadius: '20px',
                  border: `1px solid ${isDarkMode ? '#4a5568' : '#e2e8f0'}`,
                  backgroundColor: isDarkMode ? '#1a202c' : 'white',
                  color: isDarkMode ? 'white' : '#1a202c',
                  resize: 'none',
                  minHeight: '40px',
                  maxHeight: '80px',
                }}
                rows={1}
              />
              <button
                onClick={handleSendMessage}
                disabled={isLoading || !inputValue.trim()}
                style={{
                  padding: '12px 16px',
                  borderRadius: '20px',
                  backgroundColor: inputValue.trim() && !isLoading ? '#667eea' : (isDarkMode ? '#4a5568' : '#cbd5e0'),
                  color: 'white',
                  border: 'none',
                  cursor: inputValue.trim() && !isLoading ? 'pointer' : 'not-allowed',
                  fontWeight: '600',
                }}
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatKitUI;
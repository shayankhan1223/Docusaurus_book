import React, { useState, useEffect } from 'react';

const TextSelectionHandler = () => {
  const [selectedText, setSelectedText] = useState('');
  const [showPopup, setShowPopup] = useState(false);
  const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState('');

  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        const selection = window.getSelection();
        if (selection.rangeCount > 0) {
          const range = selection.getRangeAt(0);
          const rect = range.getBoundingClientRect();
          setPopupPosition({ x: rect.left, y: rect.top - 10 });
          setSelectedText(selectedText);
          setShowPopup(true);
        }
      } else {
        setShowPopup(false);
        setResponse('');
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  const handleAskAI = async () => {
    if (!selectedText.trim()) return;

    setIsLoading(true);
    setResponse('');

    try {
      // Get current language from Docusaurus context or default to 'en'
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

      // Call backend API to get explanation for selected text
      const response = await fetch('http://localhost:8000/api/text/explain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: selectedText,
          language: currentLanguage
        }),
      });

      if (response.ok) {
        const data = await response.json();
        setResponse(data.response || 'I\'m not sure how to explain this. Please try rephrasing.');
      } else {
        setResponse('Sorry, I encountered an error processing your request. Please try again.');
      }
    } catch (error) {
      setResponse('Sorry, I\'m having trouble connecting to the server. Please check your connection and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleHidePopup = () => {
    setShowPopup(false);
    setSelectedText('');
    setResponse('');
    window.getSelection().removeAllRanges();
  };

  if (!showPopup) return null;

  return (
    <div
      style={{
        position: 'absolute',
        top: popupPosition.y,
        left: popupPosition.x,
        zIndex: 10000,
        backgroundColor: 'white',
        border: '1px solid #e2e8f0',
        borderRadius: '8px',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        maxWidth: '400px',
        fontSize: '14px',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      <div style={{ padding: '12px' }}>
        <div style={{ marginBottom: '8px', fontWeight: '500', color: '#2d3748' }}>
          Selected: "{selectedText.substring(0, 60)}{selectedText.length > 60 ? '...' : ''}"
        </div>

        <button
          onClick={handleAskAI}
          disabled={isLoading}
          style={{
            backgroundColor: '#667eea',
            color: 'white',
            border: 'none',
            padding: '6px 12px',
            borderRadius: '4px',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            fontSize: '13px',
            fontWeight: '500',
            marginBottom: '8px',
          }}
        >
          {isLoading ? 'Asking AI...' : 'Ask AI to Explain'}
        </button>

        {response && (
          <div style={{
            marginTop: '8px',
            padding: '8px',
            backgroundColor: '#f8fafc',
            borderRadius: '4px',
            border: '1px solid #e2e8f0',
            fontSize: '13px',
          }}>
            <strong>AI Response:</strong> {response}
          </div>
        )}

        <button
          onClick={handleHidePopup}
          style={{
            backgroundColor: '#e2e8f0',
            color: '#4a5568',
            border: 'none',
            padding: '4px 8px',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '12px',
            marginTop: '4px',
          }}
        >
          Close
        </button>
      </div>
    </div>
  );
};

export default TextSelectionHandler;
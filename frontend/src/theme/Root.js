import React from 'react';
import ChatKitUI from '../components/ChatKitUI';
import TextSelectionHandler from '../components/TextSelectionHandler';

function Root({ children }) {
  return (
    <>
      {children}
      <TextSelectionHandler />
      <ChatKitUI />
    </>
  );
}

export default Root;
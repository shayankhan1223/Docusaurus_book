import React from 'react';
import ChatKitUI from './ChatKitUI';
import TextSelectionHandler from './TextSelectionHandler';

const App = ({ children }) => {
  return (
    <>
      {children}
      <TextSelectionHandler />
      <ChatKitUI />
    </>
  );
};

export default App;
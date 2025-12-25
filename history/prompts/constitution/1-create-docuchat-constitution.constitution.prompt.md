---
id: 1
title: Create DocuChat Constitution
stage: constitution
date: 2025-12-22
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: sp.constitution
labels: ["constitution", "setup", "architecture"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests: []
---

# Constitution Creation for DocuChat App

## Original Prompt

```text
------******  Rough plan   ******------

If i want to create an Book/App with Docusaurus, in which user will Login and SignUp and after succesfull login he will be able to see the docs,
  for Authentication BetterAuth will be used, after login he will be able to see the docs created by Docusaurus, he will see the ChatBot UI on the right side of docs, Chatbot UI will be
  created by ChatKit(you can read this:https://platform.openai.com/docs/guides/chatkit), 1 more additional feature will be on the docs 'Selection Based Question Answers' when user selects
  some text, an input field appears near the selected text. If user ask any question on the chatbot or selected text, releted to the doc will be answered by Agent, and that Agent will be
  created by OpenAI Agents SDK(Read documentation of OpenAI Agents SDK: https://openai.github.io/openai-agents-python/) and the embedded data will be stored on Qdrant vector Database.
  How Agent created by OpenAI Agents SDK will answer, When user will ask Question in ChatBot or selected text input area agent will call it's tool(this tool will also be created with
  agent) to get embedded data from the Qdrant Vector DB, to answer the user related to their query.


------******  Rules   ******------

-Plan the system architecture of my app first.
-All essential libraies and dependencies will be installed before writing code.
-Docs will be created in Docusaurus.
-Content of the docs will be written by you.
-Don't read .env file, create .env.example file.
-Code Should be clean.
-Code should be modular when needed in standarize way.
-Detailed comments for code of each file.
-Frontend will be created first.
-Frontend will be tested before moving to backend.
-Testing critaria: All files and features should be working properly.
-Utils file will be created for extra functions.
---Tech That will be used are given below---
1) Docusaurus (for documentation)
2) Chatkit (for Chatbot UI)
3) OpenAI Agents (for Chatbot to answer from the embedded data)
4) Cohere (for embedding the content of the docs)
5) Qdrant Vector Database (for storing the embedded data)
--Tel
```

## Response Summary

Created the project constitution for the DocuChat application with the following key elements:

1. **Core Principles**:
   - Modular Architecture: Components must be modular and independently testable
   - Clean Code Standards: Well-documented code with detailed comments
   - Test-First: TDD approach with comprehensive testing
   - Security-First Approach: Secure authentication and data protection
   - Performance Optimization: Efficient response times and resource usage
   - User Experience Priority: Intuitive frontend with seamless integration

2. **Technology Stack Requirements**:
   - Docusaurus for documentation
   - BetterAuth for authentication
   - ChatKit for chatbot UI
   - OpenAI Agents SDK for AI functionality
   - Cohere for embeddings
   - Qdrant Vector Database for storage

3. **Development Workflow Standards**:
   - Frontend-first development approach
   - Peer review and automated testing requirements
   - Environment variable management with .env.example files

The constitution establishes the foundational principles for the entire project, ensuring consistent development practices throughout the implementation of the documentation platform with authentication and AI-powered Q&A features.

## Outcome

Successfully created the project constitution at `.specify/memory/constitution.md` with version 1.0.0, establishing governance and development principles aligned with the user's requirements for the DocuChat application.
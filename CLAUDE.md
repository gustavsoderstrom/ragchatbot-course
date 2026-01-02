# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and FastAPI for the backend.

## Commands

**Use uv to manage dependencies and run Python files. Do not use pip directly.**

### Run the Application
```bash
cd starting-ragchatbot-codebase
./run.sh
```
Or manually:
```bash
cd starting-ragchatbot-codebase/backend
uv run uvicorn app:app --reload --port 8000
```
Access at http://localhost:8000

### Install Dependencies
```bash
cd starting-ragchatbot-codebase
uv sync
```

## Architecture

### Request Flow
1. **Frontend** (`frontend/`) - Static HTML/JS chat interface sends POST to `/api/query`
2. **FastAPI** (`backend/app.py`) - Receives request, delegates to RAGSystem
3. **RAGSystem** (`backend/rag_system.py`) - Orchestrates the query pipeline:
   - Retrieves conversation history from SessionManager
   - Calls AIGenerator with tools enabled
4. **AIGenerator** (`backend/ai_generator.py`) - Calls Claude API, handles tool use loop:
   - Claude decides whether to use CourseSearchTool
   - If yes: executes tool → gets results → calls Claude again with results
5. **CourseSearchTool** (`backend/search_tools.py`) - Searches VectorStore
6. **VectorStore** (`backend/vector_store.py`) - Queries ChromaDB with semantic search

### Key Components
- **RAGSystem** - Main orchestrator that wires all components together
- **AIGenerator** - Manages Claude API calls and tool execution loop
- **VectorStore** - ChromaDB wrapper with two collections: `course_content` (chunks) and `course_catalog` (metadata)
- **SessionManager** - In-memory conversation history per session
- **DocumentProcessor** - Parses PDF/DOCX/TXT into chunks for indexing
- **ToolManager** - Registers tools and manages their execution

### Data Flow for Tool Use
When Claude decides to search:
1. `AIGenerator.generate_response()` receives `stop_reason: "tool_use"`
2. Calls `_handle_tool_execution()` which invokes `ToolManager.execute_tool()`
3. CourseSearchTool queries VectorStore and stores sources in `last_sources`
4. Tool results sent back to Claude for final answer generation
5. RAGSystem retrieves sources via `tool_manager.get_last_sources()`

## Configuration

Settings in `backend/config.py`:
- `ANTHROPIC_MODEL`: Claude model (default: claude-sonnet-4-20250514)
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `CHUNK_SIZE`: 800 characters per chunk
- `MAX_RESULTS`: 5 search results returned
- `MAX_HISTORY`: 2 conversation turns remembered

Environment variables in `.env`:
- `ANTHROPIC_API_KEY`: Required for Claude API access

## Course Documents

Place course documents (PDF, DOCX, TXT) in `starting-ragchatbot-codebase/docs/`. They are automatically loaded on server startup.

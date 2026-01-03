"""
Tests for RAG system integration in rag_system.py

These tests verify:
1. query() method (lines 105-146)
2. Component orchestration
3. End-to-end content query flow
4. Error propagation and handling
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAGSystemQuery:
    """Test RAGSystem.query() method - the main entry point"""

    def test_query_returns_response_and_sources(self, mock_config):
        """Test that query() returns a tuple of (response, sources)"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            # Setup mocks
            mock_vs_instance = Mock()
            mock_vs_instance.get_all_courses_metadata.return_value = []  # Return empty list for link replacement
            MockVectorStore.return_value = mock_vs_instance

            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "Python is great!"
            MockAIGenerator.return_value = mock_ai_instance

            from rag_system import RAGSystem
            rag = RAGSystem(mock_config)

            response, sources = rag.query("What is Python?")

            assert isinstance(response, str)
            assert isinstance(sources, list)

    def test_query_uses_session_history(self, mock_config):
        """Test that query() retrieves and uses session history"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager') as MockSessionManager:

            mock_session = Mock()
            mock_session.get_conversation_history.return_value = "Previous conversation..."
            MockSessionManager.return_value = mock_session

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Response"
            MockAIGenerator.return_value = mock_ai

            from rag_system import RAGSystem
            rag = RAGSystem(mock_config)

            rag.query("Follow up question", session_id="session_1")

            mock_session.get_conversation_history.assert_called_with("session_1")
            # Verify history was passed to AI
            call_args = mock_ai.generate_response.call_args
            assert call_args.kwargs.get("conversation_history") == "Previous conversation..."

    def test_query_updates_session_after_response(self, mock_config):
        """Test that query() adds exchange to session history"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager') as MockSessionManager:

            mock_session = Mock()
            MockSessionManager.return_value = mock_session

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "AI Response"
            MockAIGenerator.return_value = mock_ai

            from rag_system import RAGSystem
            rag = RAGSystem(mock_config)

            rag.query("User question", session_id="session_1")

            # Verify exchange was added
            mock_session.add_exchange.assert_called_once()
            call_args = mock_session.add_exchange.call_args[0]
            assert "User question" in call_args[1]  # Original query is wrapped in prompt
            assert call_args[2] == "AI Response"

    def test_query_passes_tools_to_ai_generator(self, mock_config):
        """Test that query() provides tools to AIGenerator"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Response"
            MockAIGenerator.return_value = mock_ai

            from rag_system import RAGSystem
            rag = RAGSystem(mock_config)

            rag.query("What is Python?")

            call_args = mock_ai.generate_response.call_args
            assert "tools" in call_args.kwargs
            assert "tool_manager" in call_args.kwargs
            assert call_args.kwargs["tools"] is not None

    def test_query_resets_sources_after_retrieval(self, mock_config):
        """Test that sources are reset after being retrieved"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Response"
            MockAIGenerator.return_value = mock_ai

            from rag_system import RAGSystem
            rag = RAGSystem(mock_config)

            # Set up mock sources
            rag.tool_manager.get_last_sources = Mock(return_value=[{"text": "Source 1"}])
            rag.tool_manager.reset_sources = Mock()

            rag.query("Question")

            rag.tool_manager.reset_sources.assert_called_once()


class TestRAGSystemWithBrokenConfig:
    """Tests that reproduce the 'query failed' bug"""

    def test_broken_config_has_zero_max_results(self, broken_config):
        """
        CRITICAL TEST: Verifies that the broken config has MAX_RESULTS=0
        which causes VectorStore to return no results
        """
        assert broken_config.MAX_RESULTS == 0, \
            "Broken config should have MAX_RESULTS=0 to reproduce the bug"

    def test_correct_config_has_positive_max_results(self, mock_config):
        """Test that the correct config has positive MAX_RESULTS"""
        assert mock_config.MAX_RESULTS > 0, \
            "Correct config should have MAX_RESULTS > 0"

    def test_production_config_bug_detection(self):
        """
        CRITICAL TEST: Detects if production config has the bug.
        This test will FAIL if MAX_RESULTS=0 in config.py
        """
        from config import config

        if config.MAX_RESULTS == 0:
            pytest.fail(
                "BUG DETECTED: config.MAX_RESULTS is 0!\n"
                "This causes VectorStore.search() to return no results.\n"
                "FIX: Set MAX_RESULTS to a positive integer (e.g., 5) in config.py line 21"
            )


class TestRAGSystemInitialization:
    """Test RAGSystem initialization and component wiring"""

    def test_all_components_initialized(self, mock_config):
        """Test that all required components are initialized"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            from rag_system import RAGSystem
            rag = RAGSystem(mock_config)

            assert rag.document_processor is not None
            assert rag.vector_store is not None
            assert rag.ai_generator is not None
            assert rag.session_manager is not None
            assert rag.tool_manager is not None

    def test_search_tools_registered(self, mock_config):
        """Test that search tools are registered with ToolManager"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            from rag_system import RAGSystem
            rag = RAGSystem(mock_config)

            tool_defs = rag.tool_manager.get_tool_definitions()
            tool_names = [t["name"] for t in tool_defs]

            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names

    def test_vector_store_receives_max_results(self, mock_config):
        """Test that VectorStore is initialized with MAX_RESULTS from config"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            from rag_system import RAGSystem
            RAGSystem(mock_config)

            # Verify VectorStore was called with max_results
            call_args = MockVectorStore.call_args
            assert call_args[0][2] == mock_config.MAX_RESULTS  # Third positional arg


class TestRAGSystemAddCourseLinks:
    """Test the _add_course_links method"""

    def test_add_course_links_replaces_course_titles(self, mock_config):
        """Test that course titles are replaced with markdown links"""
        with patch('rag_system.VectorStore') as MockVectorStore, \
             patch('rag_system.AIGenerator') as MockAIGenerator, \
             patch('rag_system.DocumentProcessor'), \
             patch('rag_system.SessionManager'):

            mock_vs = Mock()
            mock_vs.get_all_courses_metadata.return_value = [
                {"title": "Python 101", "course_link": "https://example.com/python"}
            ]
            MockVectorStore.return_value = mock_vs

            mock_ai = Mock()
            mock_ai.generate_response.return_value = "Check out Python 101 for more info."
            MockAIGenerator.return_value = mock_ai

            from rag_system import RAGSystem
            rag = RAGSystem(mock_config)

            response, _ = rag.query("Tell me about Python")

            # The response should have the link
            assert "[Python 101]" in response or "https://example.com/python" in response

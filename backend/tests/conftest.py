"""
Shared fixtures for RAG chatbot backend tests.
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


# === API Testing Fixtures ===
@pytest.fixture
def mock_rag_system():
    """Creates a mock RAGSystem for API testing"""
    mock_system = Mock()
    mock_system.session_manager = Mock()
    mock_system.session_manager.create_session.return_value = "test-session-123"
    mock_system.query.return_value = (
        "Python is a versatile programming language.",
        [{"text": "Introduction to Python - Lesson 1", "link": "https://example.com/python/1"}]
    )
    mock_system.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Introduction to Python", "Machine Learning Basics", "Web Development"]
    }
    return mock_system


@pytest.fixture
def mock_rag_system_error():
    """Creates a mock RAGSystem that raises exceptions"""
    mock_system = Mock()
    mock_system.session_manager = Mock()
    mock_system.session_manager.create_session.return_value = "test-session-123"
    mock_system.query.side_effect = Exception("Database connection failed")
    mock_system.get_course_analytics.side_effect = Exception("Analytics unavailable")
    return mock_system


# === Mock Configuration ===
@dataclass
class MockConfig:
    """Test configuration with sensible defaults"""

    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5  # Correct value
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def mock_config():
    """Provides a test configuration with correct MAX_RESULTS"""
    return MockConfig()


@pytest.fixture
def broken_config():
    """Provides configuration that reproduces the bug (MAX_RESULTS=0)"""
    config = MockConfig()
    config.MAX_RESULTS = 0
    return config


# === Sample Data Fixtures ===
@pytest.fixture
def sample_course():
    """Sample course with lessons for testing"""
    return Course(
        title="Introduction to Python",
        course_link="https://example.com/python",
        instructor="John Doe",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Getting Started",
                lesson_link="https://example.com/python/1",
            ),
            Lesson(
                lesson_number=2,
                title="Variables and Types",
                lesson_link="https://example.com/python/2",
            ),
        ],
    )


@pytest.fixture
def sample_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a programming language that is widely used for web development.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Variables store data values. In Python, you don't need to declare types.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample SearchResults for testing tool execution"""
    return SearchResults(
        documents=[
            "Python is a programming language that is widely used for web development."
        ],
        metadata=[{"course_title": "Introduction to Python", "lesson_number": 1}],
        distances=[0.15],
        error=None,
    )


@pytest.fixture
def empty_search_results():
    """Empty SearchResults for testing no-results scenario"""
    return SearchResults(documents=[], metadata=[], distances=[], error=None)


@pytest.fixture
def error_search_results():
    """SearchResults with error for testing error handling"""
    return SearchResults(
        documents=[], metadata=[], distances=[], error="Search error: connection failed"
    )


# === Mock VectorStore ===
@pytest.fixture
def mock_vector_store(sample_search_results):
    """Creates a mock VectorStore that returns sample results"""
    mock_store = Mock()
    mock_store.search.return_value = sample_search_results
    mock_store.get_lesson_link.return_value = "https://example.com/python/1"
    mock_store.course_catalog = Mock()
    mock_store.course_catalog.query.return_value = {
        "documents": [["Introduction to Python"]],
        "metadatas": [
            [
                {
                    "title": "Introduction to Python",
                    "course_link": "https://example.com/python",
                    "lessons_json": '[{"lesson_number": 1, "lesson_title": "Getting Started", "lesson_link": "https://example.com/python/1"}]',
                }
            ]
        ],
        "distances": [[0.1]],
    }
    return mock_store


# === Mock Anthropic Client ===
@pytest.fixture
def mock_anthropic_response_no_tool():
    """Mock Anthropic response without tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = "This is a direct response without tool use."
    mock_response.content = [mock_text_block]
    return mock_response


@pytest.fixture
def mock_anthropic_response_with_tool():
    """Mock Anthropic response with tool use request"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Create tool_use content block
    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "tool_123"
    tool_use_block.input = {"query": "Python programming", "course_name": None}

    mock_response.content = [tool_use_block]
    return mock_response


@pytest.fixture
def mock_anthropic_final_response():
    """Mock final Anthropic response after tool execution"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_text_block = Mock()
    mock_text_block.type = "text"
    mock_text_block.text = (
        "Based on the course materials, Python is a programming language."
    )
    mock_response.content = [mock_text_block]
    return mock_response


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response_no_tool):
    """Mock Anthropic client"""
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_anthropic_response_no_tool
    return mock_client


# === Multi-Round Tool Calling Fixtures ===
@pytest.fixture
def mock_anthropic_response_second_tool_use():
    """Mock Anthropic response requesting a second tool after first result"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "get_course_outline"
    tool_use_block.id = "tool_456"
    tool_use_block.input = {"course_title": "Introduction to Python"}

    mock_response.content = [tool_use_block]
    return mock_response


@pytest.fixture
def mock_multi_round_response_sequence(
    mock_anthropic_response_with_tool,
    mock_anthropic_response_second_tool_use,
    mock_anthropic_final_response,
):
    """Complete sequence for 2-round tool calling: tool1 -> tool2 -> final"""
    return [
        mock_anthropic_response_with_tool,  # Round 1: first tool request
        mock_anthropic_response_second_tool_use,  # Round 2: second tool request
        mock_anthropic_final_response,  # Final: text response
    ]


def create_tool_use_response(tool_name: str, tool_id: str, tool_input: dict):
    """Factory function to create mock tool use responses"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = tool_name
    tool_use_block.id = tool_id
    tool_use_block.input = tool_input

    mock_response.content = [tool_use_block]
    return mock_response


def create_text_response(text: str):
    """Factory function to create mock text responses"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"

    text_block = Mock()
    text_block.type = "text"
    text_block.text = text

    mock_response.content = [text_block]
    return mock_response

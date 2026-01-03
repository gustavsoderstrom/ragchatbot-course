"""
Shared fixtures for RAG chatbot backend tests.
"""
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


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
            Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/python/1"),
            Lesson(lesson_number=2, title="Variables and Types", lesson_link="https://example.com/python/2"),
        ]
    )


@pytest.fixture
def sample_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a programming language that is widely used for web development.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Variables store data values. In Python, you don't need to declare types.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample SearchResults for testing tool execution"""
    return SearchResults(
        documents=["Python is a programming language that is widely used for web development."],
        metadata=[{"course_title": "Introduction to Python", "lesson_number": 1}],
        distances=[0.15],
        error=None
    )


@pytest.fixture
def empty_search_results():
    """Empty SearchResults for testing no-results scenario"""
    return SearchResults(documents=[], metadata=[], distances=[], error=None)


@pytest.fixture
def error_search_results():
    """SearchResults with error for testing error handling"""
    return SearchResults(documents=[], metadata=[], distances=[], error="Search error: connection failed")


# === Mock VectorStore ===
@pytest.fixture
def mock_vector_store(sample_search_results):
    """Creates a mock VectorStore that returns sample results"""
    mock_store = Mock()
    mock_store.search.return_value = sample_search_results
    mock_store.get_lesson_link.return_value = "https://example.com/python/1"
    mock_store.course_catalog = Mock()
    mock_store.course_catalog.query.return_value = {
        'documents': [['Introduction to Python']],
        'metadatas': [[{
            'title': 'Introduction to Python',
            'course_link': 'https://example.com/python',
            'lessons_json': '[{"lesson_number": 1, "lesson_title": "Getting Started", "lesson_link": "https://example.com/python/1"}]'
        }]],
        'distances': [[0.1]]
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
    mock_text_block.text = "Based on the course materials, Python is a programming language."
    mock_response.content = [mock_text_block]
    return mock_response


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response_no_tool):
    """Mock Anthropic client"""
    mock_client = Mock()
    mock_client.messages.create.return_value = mock_anthropic_response_no_tool
    return mock_client

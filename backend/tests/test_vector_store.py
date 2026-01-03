"""
Tests for VectorStore in vector_store.py

These tests focus on isolating the search functionality
to identify why "query failed" occurs.

CRITICAL: These tests demonstrate that MAX_RESULTS=0 in config.py
causes empty search results - the root cause of "query failed".
"""

import pytest
import sys
import os
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test SearchResults dataclass"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB query results"""
        chroma_results = {
            'documents': [['Doc 1', 'Doc 2']],
            'metadatas': [[{'title': 'A'}, {'title': 'B'}]],
            'distances': [[0.1, 0.2]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ['Doc 1', 'Doc 2']
        assert len(results.metadata) == 2
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("Connection failed")

        assert results.is_empty()
        assert results.error == "Connection failed"

    def test_is_empty_true_for_no_documents(self):
        """Test is_empty returns True when no documents"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty()

    def test_is_empty_false_for_documents(self):
        """Test is_empty returns False when documents exist"""
        results = SearchResults(
            documents=["Some content"],
            metadata=[{"title": "Test"}],
            distances=[0.1]
        )
        assert not results.is_empty()


class TestVectorStoreSearch:
    """Test VectorStore.search() method - critical for identifying bug"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for ChromaDB"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_search_with_zero_max_results_returns_empty(self, temp_chroma_path):
        """
        CRITICAL BUG TEST: Demonstrates that MAX_RESULTS=0 returns no results

        This is the root cause of "query failed" - config.py has MAX_RESULTS=0
        """
        vs = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=0  # BUG: This is set to 0 in production config
        )

        # Add some test data
        chunks = [
            CourseChunk(
                content="Python programming basics and fundamentals",
                course_title="Python 101",
                lesson_number=1,
                chunk_index=0
            )
        ]
        vs.add_course_content(chunks)

        # Search should return empty due to max_results=0
        results = vs.search(query="Python")

        # This demonstrates the bug - no results returned
        assert results.is_empty(), \
            "With max_results=0, search returns empty results - THIS IS THE BUG"

    def test_search_with_positive_max_results_returns_data(self, temp_chroma_path):
        """Test that search works correctly with positive MAX_RESULTS"""
        vs = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5  # Correct value
        )

        # Add some test data
        chunks = [
            CourseChunk(
                content="Python programming basics and fundamentals",
                course_title="Python 101",
                lesson_number=1,
                chunk_index=0
            )
        ]
        vs.add_course_content(chunks)

        # Search should return results
        results = vs.search(query="Python")

        assert not results.is_empty(), \
            "With max_results=5, search should return results"
        assert "Python" in results.documents[0]

    def test_search_uses_limit_parameter_over_max_results(self, temp_chroma_path):
        """Test that limit parameter overrides max_results"""
        vs = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=0  # Would return nothing
        )

        # Add test data
        chunks = [
            CourseChunk(
                content="Python programming",
                course_title="Python 101",
                lesson_number=1,
                chunk_index=0
            )
        ]
        vs.add_course_content(chunks)

        # Search with explicit limit should work
        results = vs.search(query="Python", limit=5)

        assert not results.is_empty(), \
            "Explicit limit should override max_results=0"

    def test_search_with_course_filter(self, temp_chroma_path):
        """Test search filtering by course name"""
        vs = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        # Add course metadata for resolution
        course = Course(
            title="Python 101",
            course_link="http://example.com",
            instructor="Test"
        )
        vs.add_course_metadata(course)

        # Add chunks for multiple courses
        chunks = [
            CourseChunk(
                content="Python basics",
                course_title="Python 101",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="JavaScript basics",
                course_title="JS 101",
                lesson_number=1,
                chunk_index=1
            ),
        ]
        vs.add_course_content(chunks)

        results = vs.search(query="basics", course_name="Python")

        # Should only return Python course content
        assert not results.is_empty()
        for meta in results.metadata:
            assert meta["course_title"] == "Python 101"

    def test_search_nonexistent_course_returns_error(self, temp_chroma_path):
        """Test that searching for nonexistent course returns error"""
        vs = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        results = vs.search(query="anything", course_name="Nonexistent Course")

        assert results.error is not None
        assert "No course found" in results.error

    def test_search_with_lesson_filter(self, temp_chroma_path):
        """Test search filtering by lesson number"""
        vs = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        # Add chunks for different lessons
        chunks = [
            CourseChunk(
                content="Introduction to Python",
                course_title="Python 101",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Advanced Python topics",
                course_title="Python 101",
                lesson_number=2,
                chunk_index=1
            ),
        ]
        vs.add_course_content(chunks)

        results = vs.search(query="Python", lesson_number=1)

        assert not results.is_empty()
        for meta in results.metadata:
            assert meta["lesson_number"] == 1


class TestVectorStoreConfiguration:
    """Test VectorStore configuration validation"""

    def test_production_config_max_results_check(self):
        """
        CRITICAL TEST: Detects if production config has MAX_RESULTS=0

        This test will FAIL if the bug exists, clearly identifying the issue.
        """
        from config import config

        assert config.MAX_RESULTS != 0, (
            "\n\n"
            "============================================================\n"
            "BUG DETECTED: config.MAX_RESULTS is 0!\n"
            "============================================================\n"
            "\n"
            "This causes VectorStore.search() to return no results because:\n"
            "  - VectorStore passes n_results=0 to ChromaDB\n"
            "  - ChromaDB returns empty results\n"
            "  - CourseSearchTool returns 'No relevant content found'\n"
            "  - The chatbot appears to be 'failing'\n"
            "\n"
            "FIX: Change line 21 in backend/config.py from:\n"
            "    MAX_RESULTS: int = 0\n"
            "to:\n"
            "    MAX_RESULTS: int = 5\n"
            "============================================================\n"
        )

    def test_max_results_should_be_positive(self):
        """Test that MAX_RESULTS should be a positive integer"""
        from config import config

        assert config.MAX_RESULTS > 0, \
            f"MAX_RESULTS should be positive, got {config.MAX_RESULTS}"


class TestVectorStoreAddContent:
    """Test VectorStore content addition methods"""

    @pytest.fixture
    def temp_chroma_path(self):
        """Create temporary directory for ChromaDB"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_add_course_metadata(self, temp_chroma_path):
        """Test adding course metadata"""
        vs = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        course = Course(
            title="Test Course",
            course_link="https://example.com",
            instructor="Test Instructor",
            lessons=[
                Lesson(lesson_number=1, title="Lesson 1", lesson_link="https://example.com/1")
            ]
        )

        vs.add_course_metadata(course)

        # Verify course was added
        count = vs.get_course_count()
        assert count == 1

    def test_add_course_content_chunks(self, temp_chroma_path):
        """Test adding course content chunks"""
        vs = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        chunks = [
            CourseChunk(
                content="Test content 1",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Test content 2",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1
            ),
        ]

        vs.add_course_content(chunks)

        # Verify content was added by searching
        results = vs.search(query="Test content")
        assert not results.is_empty()

    def test_get_lesson_link(self, temp_chroma_path):
        """Test retrieving lesson link"""
        vs = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        course = Course(
            title="Test Course",
            course_link="https://example.com",
            instructor="Test Instructor",  # ChromaDB requires non-None metadata values
            lessons=[
                Lesson(lesson_number=1, title="Lesson 1", lesson_link="https://example.com/lesson1")
            ]
        )
        vs.add_course_metadata(course)

        link = vs.get_lesson_link("Test Course", 1)
        assert link == "https://example.com/lesson1"

"""
Tests for CourseSearchTool.execute() in search_tools.py

These tests isolate the CourseSearchTool to verify:
1. Successful search with results
2. Empty results handling
3. Error handling from VectorStore
4. Course/lesson filtering
5. Source tracking (last_sources)
"""

import pytest
import sys
import os
from unittest.mock import Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test CourseSearchTool.execute() method (lines 52-86)"""

    def test_execute_returns_formatted_results(
        self, mock_vector_store, sample_search_results
    ):
        """Test that execute() returns properly formatted search results"""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="Python programming")

        # Verify VectorStore.search was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="Python programming", course_name=None, lesson_number=None
        )

        # Verify result contains expected content
        assert "Introduction to Python" in result
        assert "Lesson 1" in result
        assert "Python is a programming language" in result

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test execute() with course_name filter"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="variables", course_name="Python")

        mock_vector_store.search.assert_called_once_with(
            query="variables", course_name="Python", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test execute() with lesson_number filter"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="basics", lesson_number=1)

        mock_vector_store.search.assert_called_once_with(
            query="basics", course_name=None, lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test execute() with both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="syntax", course_name="Python", lesson_number=2)

        mock_vector_store.search.assert_called_once_with(
            query="syntax", course_name="Python", lesson_number=2
        )

    def test_execute_handles_empty_results(
        self, mock_vector_store, empty_search_results
    ):
        """Test that execute() returns appropriate message for empty results"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_handles_empty_results_with_course_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test empty results message includes course filter information"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="topic", course_name="Python")

        assert "No relevant content found" in result
        assert "course 'Python'" in result

    def test_execute_handles_empty_results_with_lesson_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test empty results message includes lesson filter information"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="topic", lesson_number=5)

        assert "No relevant content found" in result
        assert "lesson 5" in result

    def test_execute_handles_empty_results_with_both_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Test empty results message includes both filter information"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="topic", course_name="Python", lesson_number=5)

        assert "No relevant content found" in result
        assert "course 'Python'" in result
        assert "lesson 5" in result

    def test_execute_handles_error(self, mock_vector_store, error_search_results):
        """Test that execute() returns error message when search fails"""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute(query="anything")

        assert "Search error" in result

    def test_execute_tracks_sources(self, mock_vector_store, sample_search_results):
        """Test that execute() populates last_sources for UI"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="Python")

        assert len(tool.last_sources) > 0
        assert tool.last_sources[0]["text"] == "Introduction to Python - Lesson 1"
        assert "link" in tool.last_sources[0]
        assert "score" in tool.last_sources[0]

    def test_execute_calculates_relevance_score(
        self, mock_vector_store, sample_search_results
    ):
        """Test that relevance score is calculated correctly from distance"""
        tool = CourseSearchTool(mock_vector_store)

        tool.execute(query="Python")

        # Distance of 0.15 should give score of approximately 92-93
        # Formula: max(0, round(100 - (distance * 50))) = round(100 - 7.5) = 92 or 93
        score = tool.last_sources[0]["score"]
        assert 90 <= score <= 95


class TestCourseSearchToolDefinition:
    """Test CourseSearchTool.get_tool_definition()"""

    def test_tool_definition_structure(self, mock_vector_store):
        """Test that tool definition matches Anthropic's expected format"""
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "query" in definition["input_schema"]["required"]

    def test_tool_definition_has_optional_filters(self, mock_vector_store):
        """Test that course_name and lesson_number are defined but not required"""
        tool = CourseSearchTool(mock_vector_store)

        definition = tool.get_tool_definition()

        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        # Only query is required
        assert definition["input_schema"]["required"] == ["query"]


class TestCourseOutlineTool:
    """Test CourseOutlineTool functionality"""

    def test_execute_returns_course_outline(self, mock_vector_store):
        """Test that execute returns formatted course outline"""
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_title="Python")

        assert "Course Title: Introduction to Python" in result
        assert "Course Link:" in result
        assert "Lessons:" in result

    def test_execute_handles_no_course_found(self, mock_vector_store):
        """Test error handling when course not found"""
        mock_vector_store.course_catalog.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        tool = CourseOutlineTool(mock_vector_store)

        result = tool.execute(course_title="Nonexistent Course")

        assert "No course found" in result


class TestToolManager:
    """Test ToolManager functionality"""

    def test_register_and_execute_tool(self, mock_vector_store, sample_search_results):
        """Test registering and executing a tool through ToolManager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)
        result = manager.execute_tool("search_course_content", query="Python")

        assert "Python" in result or "Introduction" in result

    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool")

        assert "not found" in result

    def test_get_tool_definitions(self, mock_vector_store):
        """Test retrieving all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        names = [d["name"] for d in definitions]
        assert "search_course_content" in names
        assert "get_course_outline" in names

    def test_get_last_sources(self, mock_vector_store, sample_search_results):
        """Test retrieving sources after tool execution"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="Python")
        sources = manager.get_last_sources()

        assert len(sources) > 0

    def test_reset_sources(self, mock_vector_store, sample_search_results):
        """Test resetting sources after retrieval"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        manager.execute_tool("search_course_content", query="Python")
        manager.reset_sources()

        assert tool.last_sources == []

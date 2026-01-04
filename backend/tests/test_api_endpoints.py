"""
API endpoint tests for the RAG chatbot.

These tests define the API inline to avoid static file mounting issues
that occur when importing the main app.py.
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# === Test App Definition ===
# Define endpoints inline to avoid static file mounting issues from app.py

def create_test_app(mock_rag_system):
    """Factory to create a test FastAPI app with mocked RAGSystem"""

    test_app = FastAPI(title="Test Course Materials RAG System")

    # Store mock in app state for access in endpoints
    test_app.state.rag_system = mock_rag_system

    # Pydantic models (same as production)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[dict]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        """Process a query and return response with sources"""
        try:
            rag = test_app.state.rag_system
            session_id = request.session_id
            if not session_id:
                session_id = rag.session_manager.create_session()

            answer, sources = rag.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        """Get course analytics and statistics"""
        try:
            rag = test_app.state.rag_system
            analytics = rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/")
    async def root():
        """Health check endpoint"""
        return {"status": "ok", "message": "RAG System API"}

    return test_app


# === Test Fixtures ===
@pytest.fixture
def test_client(mock_rag_system):
    """Create test client with mocked RAGSystem"""
    app = create_test_app(mock_rag_system)
    return TestClient(app)


@pytest.fixture
def test_client_error(mock_rag_system_error):
    """Create test client with error-throwing RAGSystem"""
    app = create_test_app(mock_rag_system_error)
    return TestClient(app)


# === Query Endpoint Tests ===
class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_with_valid_request(self, test_client, mock_rag_system):
        """Test successful query returns answer and sources"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "Python is a versatile programming language."
        assert len(data["sources"]) == 1
        assert data["session_id"] == "test-session-123"

    def test_query_with_session_id(self, test_client, mock_rag_system):
        """Test query with existing session ID"""
        response = test_client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "existing-session-456"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing-session-456"
        # Verify mock was called with correct session ID
        mock_rag_system.query.assert_called_with("Tell me more", "existing-session-456")

    def test_query_creates_new_session_when_not_provided(self, test_client, mock_rag_system):
        """Test that a new session is created when session_id is not provided"""
        response = test_client.post(
            "/api/query",
            json={"query": "First question"}
        )

        assert response.status_code == 200
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_with_empty_query(self, test_client):
        """Test that empty query string is handled"""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )
        # FastAPI/Pydantic allows empty strings by default
        assert response.status_code == 200

    def test_query_missing_query_field(self, test_client):
        """Test that missing query field returns 422"""
        response = test_client.post(
            "/api/query",
            json={}
        )

        assert response.status_code == 422  # Unprocessable Entity

    def test_query_invalid_json(self, test_client):
        """Test that invalid JSON returns 422"""
        response = test_client.post(
            "/api/query",
            content="not json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_internal_error(self, test_client_error):
        """Test that RAGSystem errors return 500"""
        response = test_client_error.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


# === Courses Endpoint Tests ===
class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_get_courses_success(self, test_client, mock_rag_system):
        """Test successful course stats retrieval"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Introduction to Python" in data["course_titles"]

    def test_get_courses_internal_error(self, test_client_error):
        """Test that analytics errors return 500"""
        response = test_client_error.get("/api/courses")

        assert response.status_code == 500
        assert "Analytics unavailable" in response.json()["detail"]


# === Root Endpoint Tests ===
class TestRootEndpoint:
    """Tests for GET / endpoint"""

    def test_root_returns_status(self, test_client):
        """Test root endpoint returns health status"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# === Response Format Tests ===
class TestResponseFormats:
    """Tests for response structure and format"""

    def test_query_response_structure(self, test_client):
        """Test query response has correct structure"""
        response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        data = response.json()
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

    def test_sources_contain_required_fields(self, test_client):
        """Test source objects have text and link fields"""
        response = test_client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        data = response.json()
        for source in data["sources"]:
            assert "text" in source
            assert "link" in source

    def test_courses_response_structure(self, test_client):
        """Test courses response has correct structure"""
        response = test_client.get("/api/courses")

        data = response.json()
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)


# === Content-Type Tests ===
class TestContentTypes:
    """Tests for request/response content types"""

    def test_query_accepts_json(self, test_client):
        """Test query endpoint accepts application/json"""
        response = test_client.post(
            "/api/query",
            json={"query": "Test"},
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 200

    def test_query_returns_json(self, test_client):
        """Test query endpoint returns application/json"""
        response = test_client.post(
            "/api/query",
            json={"query": "Test"}
        )
        assert response.headers["content-type"] == "application/json"

    def test_courses_returns_json(self, test_client):
        """Test courses endpoint returns application/json"""
        response = test_client.get("/api/courses")
        assert response.headers["content-type"] == "application/json"

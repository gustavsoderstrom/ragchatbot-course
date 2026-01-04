from typing import List, Tuple, Optional, Dict
import os
import re
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ai_generator import AIGenerator
from session_manager import SessionManager
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from models import Course, Lesson, CourseChunk


class RAGSystem:
    """Main orchestrator for the Retrieval-Augmented Generation system"""

    def __init__(self, config):
        self.config = config

        # Initialize core components
        self.document_processor = DocumentProcessor(
            config.CHUNK_SIZE, config.CHUNK_OVERLAP
        )
        self.vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )
        self.ai_generator = AIGenerator(
            config.ANTHROPIC_API_KEY, config.ANTHROPIC_MODEL
        )
        self.session_manager = SessionManager(config.MAX_HISTORY)

        # Initialize search tools
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)
        self.outline_tool = CourseOutlineTool(self.vector_store)
        self.tool_manager.register_tool(self.outline_tool)

    def add_course_document(self, file_path: str) -> Tuple[Course, int]:
        """
        Add a single course document to the knowledge base.

        Args:
            file_path: Path to the course document

        Returns:
            Tuple of (Course object, number of chunks created)
        """
        try:
            # Process the document
            course, course_chunks = self.document_processor.process_course_document(
                file_path
            )

            # Add course metadata to vector store for semantic search
            self.vector_store.add_course_metadata(course)

            # Add course content chunks to vector store
            self.vector_store.add_course_content(course_chunks)

            return course, len(course_chunks)
        except Exception as e:
            print(f"Error processing course document {file_path}: {e}")
            return None, 0

    def add_course_folder(
        self, folder_path: str, clear_existing: bool = False
    ) -> Tuple[int, int]:
        """
        Add all course documents from a folder.

        Args:
            folder_path: Path to folder containing course documents
            clear_existing: Whether to clear existing data first

        Returns:
            Tuple of (total courses added, total chunks created)
        """
        total_courses = 0
        total_chunks = 0

        # Clear existing data if requested
        if clear_existing:
            print("Clearing existing data for fresh rebuild...")
            self.vector_store.clear_all_data()

        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            return 0, 0

        # Get existing course titles to avoid re-processing
        existing_course_titles = set(self.vector_store.get_existing_course_titles())

        # Process each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(
                (".pdf", ".docx", ".txt")
            ):
                try:
                    # Check if this course might already exist
                    # We'll process the document to get the course ID, but only add if new
                    course, course_chunks = (
                        self.document_processor.process_course_document(file_path)
                    )

                    if course and course.title not in existing_course_titles:
                        # This is a new course - add it to the vector store
                        self.vector_store.add_course_metadata(course)
                        self.vector_store.add_course_content(course_chunks)
                        total_courses += 1
                        total_chunks += len(course_chunks)
                        print(
                            f"Added new course: {course.title} ({len(course_chunks)} chunks)"
                        )
                        existing_course_titles.add(course.title)
                    elif course:
                        print(f"Course already exists: {course.title} - skipping")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

        return total_courses, total_chunks

    def query(
        self, query: str, session_id: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Process a user query using the RAG system with tool-based search.

        Args:
            query: User's question
            session_id: Optional session ID for conversation context

        Returns:
            Tuple of (response, sources list - empty for tool-based approach)
        """
        # Create prompt for the AI with clear instructions
        prompt = f"""Answer this question about course materials: {query}"""

        # Get conversation history if session exists
        history = None
        if session_id:
            history = self.session_manager.get_conversation_history(session_id)

        # Generate response using AI with tools
        response = self.ai_generator.generate_response(
            query=prompt,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager,
        )

        # Get sources from the search tool
        sources = self.tool_manager.get_last_sources()

        # Add course links to response text
        response = self._add_course_links(response)

        # Add lesson links to response text (uses sources for course context)
        response = self._add_lesson_links(response, sources)

        # Reset sources after retrieving them
        self.tool_manager.reset_sources()

        # Update conversation history
        if session_id:
            self.session_manager.add_exchange(session_id, query, response)

        # Return response with sources from tool searches
        return response, sources

    def get_course_analytics(self) -> Dict:
        """Get analytics about the course catalog"""
        return {
            "total_courses": self.vector_store.get_course_count(),
            "course_titles": self.vector_store.get_existing_course_titles(),
        }

    def _add_course_links(self, response: str) -> str:
        """Replace course title mentions with markdown links"""
        # Get all courses with their links
        all_courses = self.vector_store.get_all_courses_metadata()

        # Sort by title length (longest first) to avoid partial replacements
        # e.g., "MCP" shouldn't match inside "MCP: Build Rich-Context..."
        all_courses.sort(key=lambda c: len(c["title"]), reverse=True)

        # Replace each course mention with a link
        for course in all_courses:
            title = course["title"]
            link = course.get("course_link")
            if not link:
                continue

            # Skip if already a markdown link
            if f"[{title}]" in response:
                continue

            # Match title with optional surrounding quotes
            # Handles: "Course Name" or Course Name
            pattern = rf'"{re.escape(title)}"|{re.escape(title)}'

            # Replace with markdown link (removing quotes if present)
            def replace_with_link(_):
                return f"[{title}]({link})"

            response = re.sub(pattern, replace_with_link, response)

        return response

    def _add_lesson_links(self, response: str, sources: List[Dict]) -> str:
        """
        Replace lesson mentions (e.g., 'Lesson 6') with markdown links.

        Uses course context from sources to determine which course the lessons belong to.
        """
        if not sources:
            return response

        # Extract course titles from sources
        # Source text format: "Course Title - Lesson N" or just "Course Title"
        course_titles = []
        for source in sources:
            text = source.get("text", "")
            if " - Lesson" in text:
                course_title = text.split(" - Lesson")[0]
            else:
                course_title = text
            if course_title:
                course_titles.append(course_title)

        if not course_titles:
            return response

        # Get the most common course title (handles multi-course scenarios)
        from collections import Counter

        course_counter = Counter(course_titles)
        primary_course = course_counter.most_common(1)[0][0]

        # Pattern to match "Lesson X" or "lesson X" (case insensitive for 'lesson')
        pattern = r"\b([Ll]esson)\s+(\d+)\b"

        def replace_with_link(match):
            lesson_word = match.group(1)  # Preserves original case
            lesson_num = int(match.group(2))

            # Check if already inside a markdown link
            # Look for preceding [ that would indicate this is link text
            start = match.start()
            prefix = response[max(0, start - 50) : start]
            if "[" in prefix and "](" not in prefix[prefix.rfind("[") :]:
                # Inside a markdown link text, don't replace
                return match.group(0)

            # Get lesson link from vector store
            lesson_link = self.vector_store.get_lesson_link(primary_course, lesson_num)

            if lesson_link:
                return f"[{lesson_word} {lesson_num}]({lesson_link})"
            else:
                return match.group(0)  # No link found, keep original

        return re.sub(pattern, replace_with_link, response)

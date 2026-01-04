"""
Tests for AIGenerator tool calling flow in ai_generator.py

These tests verify:
1. generate_response() method (lines 45-89)
2. _handle_tool_execution() method (lines 91-137)
3. Two-turn conversation pattern with tools
4. Error handling in tool execution
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGeneratorGenerateResponse:
    """Test AIGenerator.generate_response() method"""

    def test_generate_response_without_tools(self, mock_anthropic_response_no_tool):
        """Test direct response when no tools are provided"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_anthropic_response_no_tool
            MockAnthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            result = generator.generate_response(query="What is Python?")

            assert result == "This is a direct response without tool use."
            mock_client.messages.create.assert_called_once()

    def test_generate_response_includes_tools_in_api_call(self, mock_anthropic_response_no_tool):
        """Test that tools are passed to API when provided"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_anthropic_response_no_tool
            MockAnthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            tools = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]

            generator.generate_response(query="Search for Python", tools=tools)

            call_args = mock_client.messages.create.call_args
            assert "tools" in call_args.kwargs
            assert call_args.kwargs["tool_choice"] == {"type": "auto"}

    def test_generate_response_includes_conversation_history(self, mock_anthropic_response_no_tool):
        """Test that conversation history is included in system prompt"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_anthropic_response_no_tool
            MockAnthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            history = "User: Hi\nAssistant: Hello!"

            generator.generate_response(query="What next?", conversation_history=history)

            call_args = mock_client.messages.create.call_args
            assert "Previous conversation" in call_args.kwargs["system"]
            assert history in call_args.kwargs["system"]

    def test_generate_response_without_history(self, mock_anthropic_response_no_tool):
        """Test that system prompt works without history"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create.return_value = mock_anthropic_response_no_tool
            MockAnthropic.return_value = mock_client

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            generator.generate_response(query="What is Python?")

            call_args = mock_client.messages.create.call_args
            assert "Previous conversation" not in call_args.kwargs["system"]


class TestAIGeneratorToolExecution:
    """Test AIGenerator._handle_tool_execution() method"""

    def test_tool_execution_flow(self, mock_anthropic_response_with_tool, mock_anthropic_final_response):
        """Test complete tool execution flow: request -> execute -> final response"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            # First call returns tool use, second call returns final response
            mock_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,
                mock_anthropic_final_response
            ]
            MockAnthropic.return_value = mock_client

            # Create mock tool manager
            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Python is a programming language used for web development."

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            tools = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]

            result = generator.generate_response(
                query="Tell me about Python",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once()

            # Verify final response
            assert "Python" in result
            assert mock_client.messages.create.call_count == 2

    def test_tool_execution_passes_correct_parameters(self, mock_anthropic_response_with_tool, mock_anthropic_final_response):
        """Test that tool parameters are correctly extracted and passed"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,
                mock_anthropic_final_response
            ]
            MockAnthropic.return_value = mock_client

            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool result"

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            generator.generate_response(
                query="test",
                tools=[{}],
                tool_manager=mock_tool_manager
            )

            # Check that tool was called with input from API response
            call_args = mock_tool_manager.execute_tool.call_args
            assert call_args[0][0] == "search_course_content"
            # Verify kwargs include the input parameters
            assert "query" in call_args[1]

    def test_tool_result_included_in_followup_message(self, mock_anthropic_response_with_tool, mock_anthropic_final_response):
        """Test that tool result is sent back to Claude in correct format"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,
                mock_anthropic_final_response
            ]
            MockAnthropic.return_value = mock_client

            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "This is the tool result"

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            generator.generate_response(
                query="test",
                tools=[{}],
                tool_manager=mock_tool_manager
            )

            # Check second API call includes tool result
            second_call = mock_client.messages.create.call_args_list[1]
            messages = second_call.kwargs["messages"]

            # Should have: user message, assistant tool_use, user tool_result
            assert len(messages) == 3
            assert messages[0]["role"] == "user"
            assert messages[1]["role"] == "assistant"
            assert messages[2]["role"] == "user"

            # Tool result is in content
            tool_result_content = messages[2]["content"]
            assert any(item["type"] == "tool_result" for item in tool_result_content)

    def test_second_api_call_includes_tools_for_potential_followup(
        self, mock_anthropic_response_with_tool, mock_anthropic_final_response
    ):
        """Test that the second API call includes tools (allowing sequential tool calls)"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,
                mock_anthropic_final_response
            ]
            MockAnthropic.return_value = mock_client

            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool result"

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            tools = [{"name": "test"}]

            generator.generate_response(
                query="test",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Check second API call DOES include tools (for potential sequential calls)
            second_call = mock_client.messages.create.call_args_list[1]
            assert "tools" in second_call.kwargs
            assert second_call.kwargs["tools"] == tools


class TestAIGeneratorConfiguration:
    """Test AIGenerator configuration and initialization"""

    def test_base_params_configured(self):
        """Test that base API parameters are set correctly"""
        with patch('anthropic.Anthropic'):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            assert generator.base_params["model"] == "claude-sonnet-4-20250514"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800

    def test_system_prompt_defined(self):
        """Test that system prompt is defined and contains key instructions"""
        with patch('anthropic.Anthropic'):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            assert "course materials" in generator.SYSTEM_PROMPT.lower()
            assert "tool" in generator.SYSTEM_PROMPT.lower()

    def test_system_prompt_has_tool_instructions(self):
        """Test that system prompt includes tool usage instructions"""
        with patch('anthropic.Anthropic'):
            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")

            assert "search_course_content" in generator.SYSTEM_PROMPT
            assert "get_course_outline" in generator.SYSTEM_PROMPT


class TestSequentialToolCalling:
    """Test sequential tool calling (up to 2 rounds per query)"""

    def test_two_sequential_tool_calls(self, mock_multi_round_response_sequence):
        """Test that Claude can make 2 sequential tool calls before final response"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            # Sequence: tool_use -> tool_use -> text
            mock_client.messages.create.side_effect = mock_multi_round_response_sequence
            MockAnthropic.return_value = mock_client

            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool result"

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            tools = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]

            result = generator.generate_response(
                query="Find courses related to lesson 4 of Python course",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify: 3 API calls (initial + 2 in loop)
            assert mock_client.messages.create.call_count == 3
            # Verify: 2 tool executions
            assert mock_tool_manager.execute_tool.call_count == 2
            # Verify: got final text response
            assert "Python" in result

    def test_single_tool_call_when_sufficient(
        self, mock_anthropic_response_with_tool, mock_anthropic_final_response
    ):
        """Test backward compatibility: single tool call still works"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            # Sequence: tool_use -> text (no second tool needed)
            mock_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,
                mock_anthropic_final_response
            ]
            MockAnthropic.return_value = mock_client

            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool result"

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            tools = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]

            result = generator.generate_response(
                query="What is Python?",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify: only 2 API calls (initial + 1 in loop)
            assert mock_client.messages.create.call_count == 2
            # Verify: only 1 tool execution
            assert mock_tool_manager.execute_tool.call_count == 1
            assert "Python" in result

    def test_max_rounds_enforced(self, mock_anthropic_response_with_tool, mock_anthropic_final_response):
        """Test that tool calling stops after MAX_TOOL_ROUNDS even if Claude wants more"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            # Claude keeps requesting tools indefinitely
            mock_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,  # Initial: tool 1
                mock_anthropic_response_with_tool,  # Round 1: tool 2
                mock_anthropic_response_with_tool,  # Round 2: tool 3 (should be forced to text)
                mock_anthropic_final_response       # Forced final response
            ]
            MockAnthropic.return_value = mock_client

            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool result"

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            tools = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]

            result = generator.generate_response(
                query="Complex query",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify: 4 API calls (initial + 2 rounds + forced final)
            assert mock_client.messages.create.call_count == 4
            # Verify: only 2 tool executions (MAX_TOOL_ROUNDS)
            assert mock_tool_manager.execute_tool.call_count == 2
            assert "Python" in result

    def test_tool_failure_terminates_loop(
        self, mock_anthropic_response_with_tool, mock_anthropic_final_response
    ):
        """Test that tool failure terminates the loop and gets graceful response"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create.side_effect = [
                mock_anthropic_response_with_tool,
                mock_anthropic_final_response  # Response after tool error
            ]
            MockAnthropic.return_value = mock_client

            mock_tool_manager = Mock()
            # Tool returns error
            mock_tool_manager.execute_tool.return_value = "Error: Tool execution failed"

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            tools = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]

            result = generator.generate_response(
                query="Search something",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Verify: only 2 API calls (initial + forced final after error)
            assert mock_client.messages.create.call_count == 2
            # Verify: only 1 tool execution (stopped after failure)
            assert mock_tool_manager.execute_tool.call_count == 1

    def test_tools_available_during_rounds(self, mock_multi_round_response_sequence):
        """Test that tools are passed to API calls during tool rounds (not stripped)"""
        with patch('anthropic.Anthropic') as MockAnthropic:
            mock_client = Mock()
            mock_client.messages.create.side_effect = mock_multi_round_response_sequence
            MockAnthropic.return_value = mock_client

            mock_tool_manager = Mock()
            mock_tool_manager.execute_tool.return_value = "Tool result"

            generator = AIGenerator(api_key="test-key", model="claude-sonnet-4-20250514")
            tools = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]

            generator.generate_response(
                query="Complex query",
                tools=tools,
                tool_manager=mock_tool_manager
            )

            # Check that second API call (first round in loop) includes tools
            second_call = mock_client.messages.create.call_args_list[1]
            assert "tools" in second_call.kwargs
            assert second_call.kwargs["tools"] == tools

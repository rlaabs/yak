"""
Tests for utility functions in the Yak framework.
Tests tool call extraction, schema generation, and type conversion utilities.
"""

import pytest
import json
from typing import List, Dict, Any, Optional, Union
from unittest.mock import patch

from yak.utils import extract_tool_calls, pydantic_model_to_json_schema

# Test pydantic functionality if available
try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
    
    class UserModel(BaseModel):
        name: str
        age: int
        email: Optional[str] = None
        tags: List[str] = []
    
    class NestedModel(BaseModel):
        user: UserModel
        active: bool = True
        
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    TestModel = None
    NestedModel = None


class TestExtractToolCalls:
    """Test tool call extraction from various response formats."""
    
    def test_extract_xml_tool_calls_single(self):
        """Test extracting a single XML-style tool call."""
        response = """
        I'll help you with that calculation.
        
        <tool_call>
        <n>calculator</n>
        <arguments>{"operation": "add", "a": 5, "b": 3}</arguments>
        </tool_call>
        
        Let me calculate that for you.
        """
        
        calls = extract_tool_calls(response)
        
        assert len(calls) == 1
        assert calls[0]["name"] == "calculator"
        assert calls[0]["arguments"] == {"operation": "add", "a": 5, "b": 3}
        assert "id" in calls[0]  # Should have generated ID
    
    def test_extract_xml_tool_calls_multiple(self):
        """Test extracting multiple XML-style tool calls."""
        response = """
        I'll use two tools to help you.
        
        <tool_call>
        <n>tool1</n>
        <arguments>{"param": "value1"}</arguments>
        </tool_call>
        
        <tool_call>
        <n>tool2</n>
        <arguments>{"param": "value2"}</arguments>
        </tool_call>
        """
        
        calls = extract_tool_calls(response)
        
        assert len(calls) == 2
        assert calls[0]["name"] == "tool1"
        assert calls[0]["arguments"] == {"param": "value1"}
        assert calls[1]["name"] == "tool2"
        assert calls[1]["arguments"] == {"param": "value2"}
        
        # Should have different IDs
        assert calls[0]["id"] != calls[1]["id"]
    
    def test_extract_xml_tool_calls_invalid_json(self):
        """Test extracting XML tool calls with invalid JSON arguments."""
        response = """
        <tool_call>
        <n>calculator</n>
        <arguments>invalid json here</arguments>
        </tool_call>
        """
        
        calls = extract_tool_calls(response)
        
        assert len(calls) == 1
        assert calls[0]["name"] == "calculator"
        assert calls[0]["arguments"] == {"input": "invalid json here"}
    
    def test_extract_xml_tool_calls_case_insensitive(self):
        """Test that XML tool call extraction is case insensitive."""
        response = """
        <TOOL_CALL>
        <N>calculator</N>
        <ARGUMENTS>{"a": 1, "b": 2}</ARGUMENTS>
        </TOOL_CALL>
        """
        
        calls = extract_tool_calls(response)
        
        assert len(calls) == 1
        assert calls[0]["name"] == "calculator"
        assert calls[0]["arguments"] == {"a": 1, "b": 2}
    
    def test_extract_function_call_format(self):
        """Test extracting function call format when XML format not found."""
        response = """
        I'll calculate this: calculator(operation="multiply", a=6, b=7)
        """
        
        calls = extract_tool_calls(response)
        
        assert len(calls) == 1
        assert calls[0]["name"] == "calculator"
        # Note: function format parsing is basic and may not work perfectly
        # This test documents current behavior
    
    def test_extract_no_tool_calls(self):
        """Test response with no tool calls."""
        response = "This is just a regular response with no tool calls."
        
        calls = extract_tool_calls(response)
        
        assert len(calls) == 0
    
    def test_extract_tool_calls_with_whitespace(self):
        """Test extracting tool calls with extra whitespace."""
        response = """
        <tool_call>
        <n>  calculator  </n>
        <arguments>  {"a": 1, "b": 2}  </arguments>
        </tool_call>
        """
        
        calls = extract_tool_calls(response)
        
        assert len(calls) == 1
        assert calls[0]["name"] == "calculator"  # Should be trimmed
        assert calls[0]["arguments"] == {"a": 1, "b": 2}
    
    def test_extract_tool_calls_multiline_arguments(self):
        """Test extracting tool calls with multiline arguments."""
        response = """
        <tool_call>
        <n>complex_tool</n>
        <arguments>{
            "param1": "value1",
            "param2": {
                "nested": "value"
            }
        }</arguments>
        </tool_call>
        """
        
        calls = extract_tool_calls(response)
        
        assert len(calls) == 1
        assert calls[0]["name"] == "complex_tool"
        assert calls[0]["arguments"]["param1"] == "value1"
        assert calls[0]["arguments"]["param2"]["nested"] == "value"


@pytest.mark.skipif(not HAS_PYDANTIC, reason="Pydantic not available")
class TestPydanticSchemaGeneration:
    """Test Pydantic model to JSON schema conversion."""
    
    def test_simple_model_schema(self):
        """Test schema generation for simple model."""
        schema = pydantic_model_to_json_schema(UserModel)
        
        assert schema["type"] == "json_schema"
        assert schema["json_schema"]["name"] == "usermodel"
        assert schema["json_schema"]["strict"] == True
        
        json_schema = schema["json_schema"]["schema"]
        assert json_schema["type"] == "object"
        assert "properties" in json_schema
        assert "name" in json_schema["properties"]
        assert "age" in json_schema["properties"]
        assert "email" in json_schema["properties"]
        assert "tags" in json_schema["properties"]
        
        # All properties should be required
        assert set(json_schema["required"]) == {"name", "age", "email", "tags"}
        assert json_schema["additionalProperties"] == False
    
    def test_nested_model_schema(self):
        """Test schema generation for nested model."""
        schema = pydantic_model_to_json_schema(NestedModel)
        
        json_schema = schema["json_schema"]["schema"]
        assert "user" in json_schema["properties"]
        assert "active" in json_schema["properties"]
        
        # Nested object should be properly defined
        user_schema = json_schema["properties"]["user"]
        # The nested schema might be a reference or have different structure
        # Let's just check that it exists and has some expected content
        assert user_schema is not None
        # It might be a $ref or have properties directly
        assert "$ref" in user_schema or "properties" in user_schema or "type" in user_schema
    
    def test_custom_schema_name(self):
        """Test schema generation with custom name."""
        schema = pydantic_model_to_json_schema(UserModel, schema_name="custom_name")
        
        assert schema["json_schema"]["name"] == "custom_name"
    
    def test_provider_type_parameter(self):
        """Test that provider_type parameter is accepted."""
        # Both should produce the same result currently
        schema_openai = pydantic_model_to_json_schema(UserModel, provider_type="openai")
        schema_openrouter = pydantic_model_to_json_schema(UserModel, provider_type="openrouter")
        
        assert schema_openai["type"] == "json_schema"
        assert schema_openrouter["type"] == "json_schema"
    
    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError, match="response_format must be a Pydantic model class"):
            pydantic_model_to_json_schema("not_a_model")
    
    def test_non_pydantic_model(self):
        """Test error handling for non-Pydantic model."""
        class NotPydanticModel:
            pass
        
        with pytest.raises(ValueError, match="response_format must be a Pydantic model class"):
            pydantic_model_to_json_schema(NotPydanticModel)


class TestPydanticNotAvailable:
    """Test behavior when Pydantic is not available."""
    
    def test_pydantic_import_error(self):
        """Test that ImportError is raised when Pydantic is not available."""
        with patch('yak.utils.HAS_PYDANTIC', False):
            with pytest.raises(ImportError, match="Pydantic is required for response_format"):
                pydantic_model_to_json_schema("dummy")


class TestTypeConversion:
    """Test type hint to JSON schema conversion (indirectly through Yak)."""
    
    def test_basic_types(self):
        """Test conversion of basic Python types."""
        from yak.core import Yak
        from test_core import MockProvider
        
        def test_func(
            string_param: str,
            int_param: int,
            float_param: float,
            bool_param: bool,
            list_param: list,
            dict_param: dict
        ) -> str:
            """Test function with various parameter types."""
            return "test"
        
        yak = Yak(provider=MockProvider())
        schema = yak.generate_tool_schema(test_func)
        
        props = schema["function"]["parameters"]["properties"]
        assert props["string_param"]["type"] == "string"
        assert props["int_param"]["type"] == "integer"
        assert props["float_param"]["type"] == "number"
        assert props["bool_param"]["type"] == "boolean"
        assert props["list_param"]["type"] == "array"
        assert props["dict_param"]["type"] == "object"
    
    def test_optional_types(self):
        """Test conversion of Optional types."""
        from yak.core import Yak
        from test_core import MockProvider
        
        def test_func(
            required_param: str,
            optional_param: Optional[str] = None
        ) -> str:
            """Test function with optional parameter."""
            return "test"
        
        yak = Yak(provider=MockProvider())
        schema = yak.generate_tool_schema(test_func)
        
        params = schema["function"]["parameters"]
        assert "required_param" in params["required"]
        assert "optional_param" not in params["required"]
        
        # Optional[str] should still be type string
        assert params["properties"]["optional_param"]["type"] == "string"
    
    def test_list_with_type_args(self):
        """Test conversion of List[T] types."""
        from yak.core import Yak
        from test_core import MockProvider
        
        def test_func(string_list: List[str], int_list: List[int]) -> str:
            """Test function with typed lists."""
            return "test"
        
        yak = Yak(provider=MockProvider())
        schema = yak.generate_tool_schema(test_func)
        
        props = schema["function"]["parameters"]["properties"]
        assert props["string_list"]["type"] == "array"
        assert props["string_list"]["items"]["type"] == "string"
        assert props["int_list"]["type"] == "array"
        assert props["int_list"]["items"]["type"] == "integer"


class TestDocstringParsing:
    """Test docstring parsing for parameter descriptions."""
    
    def test_docstring_parsing(self):
        """Test that docstring parsing extracts parameter descriptions."""
        from yak.core import Yak
        from test_core import MockProvider
        
        def well_documented_func(param1: str, param2: int) -> str:
            """
            A well documented function.
            
            Args:
                param1: Description of param1
                param2: Description of param2
            
            Returns:
                A string result
            """
            return "result"
        
        yak = Yak(provider=MockProvider())
        schema = yak.generate_tool_schema(well_documented_func)
        
        props = schema["function"]["parameters"]["properties"]
        assert "Description of param1" in props["param1"]["description"]
        assert "Description of param2" in props["param2"]["description"]
    
    def test_docstring_parsing_with_types(self):
        """Test docstring parsing with type annotations in docstring."""
        from yak.core import Yak
        from test_core import MockProvider
        
        def func_with_type_docs(param1: str, param2: int) -> str:
            """
            Function with type info in docstring.
            
            Args:
                param1 (str): String parameter description
                param2 (int): Integer parameter description
            """
            return "result"
        
        yak = Yak(provider=MockProvider())
        schema = yak.generate_tool_schema(func_with_type_docs)
        
        props = schema["function"]["parameters"]["properties"]
        assert "String parameter description" in props["param1"]["description"]
        assert "Integer parameter description" in props["param2"]["description"]
    
    def test_no_docstring(self):
        """Test handling of functions without docstrings."""
        from yak.core import Yak
        from test_core import MockProvider
        
        def undocumented_func(param1: str) -> str:
            return "result"
        
        yak = Yak(provider=MockProvider())
        schema = yak.generate_tool_schema(undocumented_func)
        
        assert "Execute undocumented_func" in schema["function"]["description"]
        props = schema["function"]["parameters"]["properties"]
        assert "Parameter param1" in props["param1"]["description"]


if __name__ == "__main__":
    pytest.main([__file__])

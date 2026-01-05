"""Tests for web server implementation."""

from __future__ import annotations

import io
import logging
import threading
import time
from unittest.mock import Mock, MagicMock, patch

import pytest

from aws_durable_execution_sdk_python_testing.exceptions import (
    IllegalStateException,
    InvalidParameterValueException,
    ResourceNotFoundException,
    SerializationError,
    UnknownRouteError,
)
from aws_durable_execution_sdk_python_testing.web.models import (
    HTTPRequest,
    HTTPResponse,
)
from aws_durable_execution_sdk_python_testing.web.routes import (
    BytesPayloadRoute,
    CallbackSuccessRoute,
    GetDurableExecutionRoute,
    HealthRoute,
    Router,
    StartExecutionRoute,
)
from aws_durable_execution_sdk_python_testing.web.server import (
    RequestHandler,
    WebServer,
    WebServiceConfig,
)


def test_web_service_config_default_values():
    """Test that default configuration values are correct."""

    config = WebServiceConfig()

    assert config.host == "localhost"
    assert config.port == 5000
    assert config.log_level == logging.INFO
    assert config.max_request_size == 10 * 1024 * 1024


def test_web_service_config_custom_values():
    """Test that custom configuration values are set correctly."""

    config = WebServiceConfig(
        host="127.0.0.1",
        port=9000,
        log_level=logging.DEBUG,
        max_request_size=5 * 1024 * 1024,
    )

    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.log_level == logging.DEBUG
    assert config.max_request_size == 5 * 1024 * 1024


def test_web_service_config_frozen_dataclass():
    """Test that WebServiceConfig is immutable."""
    config = WebServiceConfig()

    with pytest.raises(AttributeError):
        config.port = 9000


def test_web_server_initialization():
    """Test that WebServer initializes correctly."""
    config = WebServiceConfig(port=0)  # Use port 0 for testing
    executor = Mock()

    with WebServer(config, executor) as server:
        assert server.config == config
        assert server.executor == executor


def test_web_server_context_manager():
    """Test that WebServer works as a context manager."""
    config = WebServiceConfig(port=0)
    executor = Mock()

    # Test context manager entry and exit
    with WebServer(config, executor) as server:
        assert isinstance(server, WebServer)
        assert server.config == config
        assert server.executor == executor

    # Server should be cleaned up after context exit


def test_web_server_background_usage():
    """Test that server can be used in background thread for testing."""
    config = WebServiceConfig(port=0)
    executor = Mock()

    with WebServer(config, executor) as server:
        # Start server in background thread
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        # Give it a moment to start
        time.sleep(0.1)
        assert server_thread.is_alive()

        # Stop the server
        server.shutdown()

        # Give it a moment to shutdown
        time.sleep(0.1)
        server_thread.join(timeout=1)
        assert not server_thread.is_alive()


def test_web_server_has_executor_reference():
    """Test that WebServer stores executor reference correctly."""
    config = WebServiceConfig(port=0)
    executor = Mock()

    with WebServer(config, executor) as server:
        # Verify server has executor reference
        assert server.executor == executor

        # Verify RequestHandler class is set correctly

        assert server.RequestHandlerClass == RequestHandler


def test_web_server_has_router_and_handlers():
    """Test that WebServer creates router and handlers correctly."""

    executor = Mock()
    config = WebServiceConfig(port=0)  # Use port 0 to get any available port

    with WebServer(config, executor) as server:
        # Verify router is created
        assert server.router is not None
        assert isinstance(server.router, Router)

        # Verify handlers are created
        assert server.endpoint_handlers is not None
        assert len(server.endpoint_handlers) > 0

        # Verify specific handlers exist
        assert StartExecutionRoute in server.endpoint_handlers
        assert HealthRoute in server.endpoint_handlers

        # Verify handlers have executor reference
        start_handler = server.endpoint_handlers[StartExecutionRoute]
        assert start_handler.executor is executor


def test_web_server_all_routes_have_handlers():
    """Test that all routes in the router have corresponding handlers."""
    executor = Mock()
    config = WebServiceConfig(port=0)  # Use port 0 to get any available port

    with WebServer(config, executor) as server:
        # Test that router can find routes for all handler types
        handler_route_types = set(server.endpoint_handlers.keys())

        # Test a sample of routes to verify router functionality
        test_routes = [
            ("/start-durable-execution", "POST", StartExecutionRoute),
            ("/health", "GET", HealthRoute),
            (
                "/2025-12-01/durable-executions/test-arn",
                "GET",
                GetDurableExecutionRoute,
            ),
        ]

        for path, method, expected_route_type in test_routes:
            # Verify router can find the route (tests public API)
            found_route = server.router.find_route(path, method)
            assert isinstance(found_route, expected_route_type)

            # Verify handler exists for this route type
            assert expected_route_type in handler_route_types


def test_request_handler_exception_mapping():
    """Test that RequestHandler has proper exception handling capabilities."""

    # Verify that all the required exception types are available for import
    assert SerializationError is not None
    assert InvalidParameterValueException is not None
    assert ResourceNotFoundException is not None
    assert IllegalStateException is not None
    assert UnknownRouteError is not None


def test_http_response_create_error_from_exception():
    """Test HTTPResponse.create_error_from_exception method directly."""
    test_exception = InvalidParameterValueException("Test error message")
    response = HTTPResponse.create_error_from_exception(test_exception)

    assert response.status_code == 400
    assert response.headers["Content-Type"] == "application/json"

    # AWS-compliant format without wrapper
    expected_body = {
        "Type": "InvalidParameterValueException",
        "message": "Test error message",
    }
    assert response.body == expected_body


def test_request_handler_error_response_through_public_api():
    """Test error response handling through public do_POST method."""
    import io
    from unittest.mock import MagicMock

    # Create a mock request handler with minimal setup
    mock_server = MagicMock()
    mock_server.executor = Mock()
    mock_server.router = Mock()
    mock_server.endpoint_handlers = {}

    # Mock the router to raise an exception
    mock_server.router.find_route.side_effect = InvalidParameterValueException(
        "Test error message"
    )

    # Create handler instance
    with patch.object(RequestHandler, "__init__", return_value=None):
        handler = RequestHandler.__new__(RequestHandler)
        handler.executor = mock_server.executor
        handler.router = mock_server.router
        handler.endpoint_handlers = mock_server.endpoint_handlers
        handler.path = "/test-path"
        handler.headers = {"Content-Length": "0"}
        handler.rfile = io.BytesIO(b"")

    # Mock the response sending
    with patch.object(handler, "_send_response") as mock_send_response:
        # Call the public method that should trigger error handling
        handler.do_POST()

        # Verify _send_response was called with correct error response
        mock_send_response.assert_called_once()
        response = mock_send_response.call_args[0][0]

        assert response.status_code == 400
        assert response.headers["Content-Type"] == "application/json"

        # AWS-compliant format without wrapper
        expected_body = {
            "Type": "InvalidParameterValueException",
            "message": "Test error message",
        }
        assert response.body == expected_body


def test_handle_request_successful_route_parsing():
    """Test _handle_request successfully parses route and calls handler."""
    mock_route = Mock(spec=StartExecutionRoute)
    mock_handler = Mock()
    mock_response = HTTPResponse(status_code=200, headers={}, body={})
    mock_handler.handle.return_value = mock_response

    handler = _create_mock_request_handler()
    handler.router.find_route.return_value = mock_route
    handler.endpoint_handlers = {type(mock_route): mock_handler}
    handler.path = "/test-path"
    handler.headers = {"Content-Length": "10"}
    handler.rfile = io.BytesIO(b'{"test":1}')

    with patch.object(handler, "_send_response") as mock_send:
        handler._handle_request("POST")

    handler.router.find_route.assert_called_once_with("/test-path", "POST")
    mock_handler.handle.assert_called_once()
    mock_send.assert_called_once_with(mock_response)


def test_handle_request_bytes_payload_route():
    """Test _handle_request handles BytesPayloadRoute with raw bytes."""
    mock_route = Mock(spec=BytesPayloadRoute)
    mock_handler = Mock()
    mock_response = HTTPResponse(status_code=200, headers={}, body={})
    mock_handler.handle.return_value = mock_response

    handler = _create_mock_request_handler()
    handler.router.find_route.return_value = mock_route
    handler.endpoint_handlers = {type(mock_route): mock_handler}
    handler.path = "/callback?param=value"
    handler.headers = {"Content-Length": "5"}
    handler.rfile = io.BytesIO(b"bytes")

    with (
        patch.object(HTTPRequest, "from_raw_bytes") as mock_from_raw,
        patch.object(handler, "_send_response"),
    ):
        mock_from_raw.return_value = Mock()
        handler._handle_request("POST")

    mock_from_raw.assert_called_once_with(
        body_bytes=b"bytes",
        method="POST",
        path=mock_route,
        headers=dict(handler.headers),
        query_params={"param": ["value"]},
    )


def test_handle_request_regular_route():
    """Test _handle_request handles regular route with parsed body."""
    mock_route = Mock(spec=StartExecutionRoute)
    mock_handler = Mock()
    mock_response = HTTPResponse(status_code=200, headers={}, body={})
    mock_handler.handle.return_value = mock_response

    handler = _create_mock_request_handler()
    handler.router.find_route.return_value = mock_route
    handler.endpoint_handlers = {type(mock_route): mock_handler}
    handler.path = "/start?timeout=30"
    handler.headers = {"Content-Length": "0"}
    handler.rfile = io.BytesIO(b"")

    with (
        patch.object(HTTPRequest, "from_bytes") as mock_from_bytes,
        patch.object(handler, "_send_response"),
    ):
        mock_from_bytes.return_value = Mock()
        handler._handle_request("POST")

    mock_from_bytes.assert_called_once_with(
        body_bytes=b"",
        operation_name=None,
        method="POST",
        path=mock_route,
        headers=dict(handler.headers),
        query_params={"timeout": ["30"]},
    )


def test_handle_request_exception_handling():
    """Test _handle_request handles exceptions and sends error response."""
    handler = _create_mock_request_handler()
    handler.router.find_route.side_effect = ValueError("Test error")
    handler.path = "/error-path"

    with patch.object(handler, "_send_response") as mock_send:
        handler._handle_request("GET")

    mock_send.assert_called_once()
    response = mock_send.call_args[0][0]
    assert response.status_code == 500


def _create_mock_request_handler():
    """Helper to create a mock RequestHandler for testing."""
    handler = Mock(spec=RequestHandler)
    handler.router = Mock()
    handler.endpoint_handlers = {}
    handler.headers = {}
    handler._handle_request = RequestHandler._handle_request.__get__(handler)
    handler._send_response = Mock()
    return handler

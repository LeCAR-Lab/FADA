"""
Unit tests for holosoma_inference.utils.clock module.
"""

from unittest import mock

import zmq

from holosoma_inference.utils.clock import ClockPub, ClockSub


class TestClockPub:
    """Test cases for ClockPub class."""

    @mock.patch("zmq.Context")
    @mock.patch("time.time")
    def test_start_success(self, mock_time, mock_context):
        """Test successful start of ClockPub."""
        mock_time.return_value = 123.456
        mock_ctx = mock.Mock()
        mock_socket = mock.Mock()
        mock_context.return_value = mock_ctx
        mock_ctx.socket.return_value = mock_socket

        clock_pub = ClockPub(port=5556)
        clock_pub.start()

        assert clock_pub.enabled is True
        assert clock_pub.start_time == 123.456
        mock_socket.bind.assert_called_once_with("tcp://*:5556")

    def test_publish_success(self):
        """Test successful publish."""
        mock_socket = mock.Mock()

        clock_pub = ClockPub()
        clock_pub.enabled = True
        clock_pub.socket = mock_socket
        clock_pub.publish(1.5)
        mock_socket.send_string.assert_called_once_with("1500", zmq.NOBLOCK)

    def test_close(self):
        """Test ClockPub close functionality."""
        mock_socket = mock.Mock()
        mock_context = mock.Mock()

        clock_pub = ClockPub()
        clock_pub.socket = mock_socket
        clock_pub.context = mock_context
        clock_pub.enabled = True

        clock_pub.close()

        mock_socket.close.assert_called_once()
        mock_context.term.assert_called_once()
        assert clock_pub.enabled is False


class TestClockSub:
    """Test cases for ClockSub class."""

    def test_init(self):
        """Test ClockSub initialization."""
        clock_sub = ClockSub(port=5556)
        assert clock_sub.port == 5556
        assert clock_sub.last_clock == 0
        assert clock_sub._offset == 0

    @mock.patch("zmq.Context")
    def test_start(self, mock_context):
        """Test ClockSub start functionality."""
        mock_ctx = mock.Mock()
        mock_socket = mock.Mock()
        mock_context.return_value = mock_ctx
        mock_ctx.socket.return_value = mock_socket

        clock_sub = ClockSub(port=5556)
        clock_sub.start()

        mock_ctx.socket.assert_called_once_with(zmq.SUB)
        mock_socket.connect.assert_called_once_with("tcp://localhost:5556")

    def test_get_clock_with_messages(self):
        """Test get_clock with available messages."""
        mock_socket = mock.Mock()
        mock_socket.recv_string.side_effect = ["2500", zmq.Again()]

        clock_sub = ClockSub()
        clock_sub.socket = mock_socket
        clock_sub.last_clock = 1000

        result = clock_sub.get_clock()

        assert result == 2500
        assert clock_sub.last_clock == 2500

    def test_get_clock_no_messages(self):
        """Test get_clock when no messages are available."""
        mock_socket = mock.Mock()
        mock_socket.recv_string.side_effect = zmq.Again()

        clock_sub = ClockSub()
        clock_sub.socket = mock_socket
        clock_sub.last_clock = 1000

        result = clock_sub.get_clock()

        assert result == 1000

    def test_reset_origin(self):
        """Clock origin reset should zero the adjusted time."""
        mock_socket = mock.Mock()
        mock_socket.recv_string.side_effect = ["5000", zmq.Again()]

        clock_sub = ClockSub()
        clock_sub.socket = mock_socket

        clock_sub.reset_origin()
        assert clock_sub._offset == 5000

        mock_socket.recv_string.side_effect = ["6500", zmq.Again()]
        assert clock_sub.get_clock() == 1500

    def test_get_clock_handles_restart(self):
        """Clock subscriber should detect simulator restarts without explicit reset."""
        mock_socket = mock.Mock()
        clock_sub = ClockSub()
        clock_sub.socket = mock_socket

        # Simulate having previously seen a large timestamp with a matching offset
        clock_sub.last_clock = 4000
        clock_sub._offset = 4000

        # Now the simulator restarts and emits a small timestamp
        mock_socket.recv_string.side_effect = ["100", zmq.Again()]
        assert clock_sub.get_clock() == 0
        assert clock_sub._offset == 100

        # Subsequent timestamps should advance from the new origin
        mock_socket.recv_string.side_effect = ["350", zmq.Again()]
        assert clock_sub.get_clock() == 250

"""Exceptions for randomization system."""


class RandomizerNotSupportedError(NotImplementedError):
    """Raised when a randomizer doesn't support the current simulator."""

    def __init__(self, message: str):
        """Initialize the exception with a helpful message.

        Args:
            message: The base error message
        """
        full_message = (
            f"{message}\nTo continue without unsupported randomizations, set '--randomization.ignore_unsupported=True'"
        )
        super().__init__(full_message)

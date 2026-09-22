"""Exceptions that carry a message safe to show the user."""


class RagError(Exception):
    """Base class for failures the UI can report verbatim."""


class UnsupportedFileTypeError(RagError):
    pass


class FileTooLargeError(RagError):
    pass


class EmptyDocumentError(RagError):
    pass


class GenerationError(RagError):
    """The retrieval succeeded but the language model could not answer."""

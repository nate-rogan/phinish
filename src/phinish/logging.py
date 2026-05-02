"""Structlog configuration — call ``configure()`` once at package init."""

import os

import structlog


def configure() -> None:
    """Set up structlog: JSON in CI, colored console locally."""
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if os.environ.get("CI")
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

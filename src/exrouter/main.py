#!/usr/bin/env python3
"""EXRouter main entry point."""

import argparse
import asyncio
import logging
import sys
import os

from .config import Config, ServerConfig
from .proxy import LockProxy


def setup_logging(level: str = "info") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Propagate level to all exrouter sub-loggers
    for name in ("exrouter", "exrouter.proxy", "exrouter.lifecycle",
                 "exrouter.hooks", "exrouter.remapper"):
        logging.getLogger(name).setLevel(numeric_level)


def main():
    parser = argparse.ArgumentParser(description="EXRouter - Declarative backend proxy with global locking and remapping")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to config file (required)")
    args = parser.parse_args()

    log_level = os.environ.get("EXROUTER_LOG_LEVEL", "info")

    try:
        setup_logging(log_level)
        logger = logging.getLogger("exrouter")

        # Load config from file (server: section is optional)
        config = Config.from_file(args.config)
        
        # Apply ENV variables (EXROUTER_HOST, EXROUTER_PORT) - overrides config file
        config.server = ServerConfig.from_env()
        
        logger.info("EXRouter v1.1.0 starting up (with request remapping)")
        logger.info(f"Domains: {list(config.domains.keys())}")
        logger.info(f"Starting on {config.server.host}:{config.server.port}")

        proxy = LockProxy(config)
        asyncio.run(proxy.run())

    except Exception as e:
        logger = logging.getLogger("exrouter")
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

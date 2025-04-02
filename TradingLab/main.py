# Entry point dell'applicazione

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TradingLab - Application for technical analysis and algorithmic trading.
Main entry point to run the application.
"""

import sys
import os
import logging
from pathlib import Path
import argparse

# Add the parent directory to sys.path if running as script
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

# PyQt6 imports
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings

# Import from TradingLab modules
from TradingLab.config import (
    APP_NAME, APP_VERSION, APP_AUTHOR,
    create_directories, LOG_LEVEL
)
from TradingLab.gui.main_window import MainWindow
from TradingLab.gui.styles import style_manager, Theme
from TradingLab.utils import (
    app_logger, configure_root_logger, silence_loggers,
    log_exception, clean_temp_files
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=f"{APP_NAME} v{APP_VERSION}")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--theme", choices=["light", "dark"], default=None, help="Set UI theme")
    parser.add_argument("--clean", action="store_true", help="Clean temporary files on startup")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default=None, help="Set logging level")
    return parser.parse_args()


def setup_application():
    """Setup the application environment."""
    app_logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    
    # Create necessary directories
    create_directories()
    app_logger.debug("Directories created")
    
    # Load application settings
    settings = QSettings(APP_NAME, "settings")
    
    return settings


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging based on arguments
    if args.log_level:
        log_level = getattr(logging, args.log_level)
    else:
        log_level = LOG_LEVEL
    
    configure_root_logger(level=log_level)
    
    # Clean temporary files if requested
    if args.clean:
        cleaned_files = clean_temp_files()
        app_logger.info(f"Cleaned {cleaned_files} temporary files")
    
    # Setup application
    settings = setup_application()
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_AUTHOR)
    
    # Set theme
    if args.theme:
        theme = Theme.DARK if args.theme.lower() == "dark" else Theme.LIGHT
    else:
        # Load from settings or use default
        theme_setting = settings.value("theme", "light")
        theme = Theme.DARK if theme_setting == "dark" else Theme.LIGHT
    
    style_manager.set_theme(theme)
    app.setStyleSheet(style_manager.get_app_stylesheet())
    
    # Create and show main window
    try:
        main_window = MainWindow()
        main_window.show()
        
        if args.debug:
            app_logger.info("Debug mode enabled")
            main_window.setWindowTitle(f"{main_window.windowTitle()} [DEBUG]")
        
        # Execute application
        return app.exec()
    
    except Exception as e:
        log_exception(app_logger, e, "Fatal error during application startup:")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log_exception(app_logger, e, "Unhandled exception:")
        sys.exit(1)
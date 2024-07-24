import argparse
import os
import sys

from PyQt5.QtWebEngineWidgets import QWebEngineSettings
from PyQt5.QtWidgets import QApplication

from drcell.DrCELLQWindow import DrCELLQWindow

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Bokeh server with custom app.")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Bokeh server")
    parser.add_argument("--port-image", type=int, default=8000, help="Port for the image server")
    parser.add_argument("--app-path", type=str, default=None, help="Path to the Bokeh application script")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # Enable PyQt WebEngine
    QWebEngineSettings.globalSettings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
    QWebEngineSettings.globalSettings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)

    main_window = DrCELLQWindow(q_application=app, port=args.port, port_image=args.port_image, app_path=args.app_path)

    sys.exit(app.exec_())

import argparse
import sys
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
from threading import Thread
import startBokehServer
import util


# PyQt window class
class BokehWindow(QMainWindow):

    def __init__(self, port=5000, port_image=8000, app_path="main.py"):
        super().__init__()
        self.port = port
        self.port_image = port_image
        self.app_path = app_path
        while not util.is_port_available(self.port):
            print(f"Server port {self.port} is not available")
            self.port += 1
        while not util.is_port_available(self.port_image):
            print(f"Image server port {self.port_image} is not available")
            self.port_image += 1

        print(f"Server port: {port}")
        print(f"Image server port: {port_image}")
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Dimensional reduction Cluster Exploration and Labeling Library')
        kwargs = {'port': self.port, 'port_image': self.port_image, 'app_path': self.app_path, "skip_port_check": True}
        # Start Bokeh server in a separate thread
        self.bokeh_thread = Thread(target=startBokehServer.run_server, kwargs=kwargs)
        self.bokeh_thread.start()

        # Create a web view widget
        self.web_view = QWebEngineView()

        # Set the web view widget to display the Bokeh application
        self.web_view.setUrl(QUrl(f'http://localhost:{self.port}/'))

        # Create a layout and set the web view widget as its central widget
        layout = QVBoxLayout()
        layout.addWidget(self.web_view)

        # Create a container widget and set the layout
        container = QWidget()
        container.setLayout(layout)

        # Set the container widget as the central widget of the main window
        self.setCentralWidget(container)

        # Set window size and show the window
        self.setGeometry(100, 100, 1200, 900)
        self.show()

    def closeEvent(self, event):
        # Stop the Bokeh server thread
        startBokehServer.stop_server()
        self.bokeh_thread.join()  # Wait for the thread to finish
        app.quit()  # Exit the application
        event.accept()


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Bokeh server with custom app.")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Bokeh server")
    parser.add_argument("--port-image", type=int, default=8000, help="Port for the image server")
    parser.add_argument("--app-path", type=str, default="main.py", help="Path to the Bokeh application script")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    # Enable PyQt WebEngine
    QWebEngineSettings.globalSettings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
    QWebEngineSettings.globalSettings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)

    main_window = BokehWindow(port=args.port, port_image=args.port_image, app_path=args.app_path)

    sys.exit(app.exec_())

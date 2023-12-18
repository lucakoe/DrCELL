import sys

from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from threading import Thread
import startBokehServer

# Enable PyQt WebEngine
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings

# PyQt window class
class BokehWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Bokeh Embedded in PyQt')

        # Start Bokeh server in a separate thread
        bokeh_thread = Thread(target=startBokehServer.run_server)
        bokeh_thread.start()

        # Create a web view widget
        web_view = QWebEngineView()

        # Set the web view widget to display the Bokeh application
        web_view.setUrl(QUrl('http://localhost:5000/'))

        # Create a layout and set the web view widget as its central widget
        layout = QVBoxLayout()
        layout.addWidget(web_view)

        # Create a container widget and set the layout
        container = QWidget()
        container.setLayout(layout)

        # Set the container widget as the central widget of the main window
        self.setCentralWidget(container)

        # Set window size and show the window
        self.setGeometry(100, 100, 800, 600)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    QWebEngineSettings.globalSettings().setAttribute(QWebEngineSettings.PluginsEnabled, True)
    QWebEngineSettings.globalSettings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)

    main_window = BokehWindow()

    sys.exit(app.exec_())
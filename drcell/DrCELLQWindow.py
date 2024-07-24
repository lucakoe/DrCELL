from threading import Thread

from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QApplication

from drcell.server import DrCELLBokehServer


# PyQt window class
class DrCELLQWindow(QMainWindow):

    def __init__(self, q_application: QApplication, port=5000, port_image=8000, app_path=None):
        super().__init__()
        self.port = port
        self.port_image = port_image
        self.q_app = q_application
        self.app_path = app_path
        self.dr_cell_server = DrCELLBokehServer(port=self.port, port_image=self.port_image,
                                                app_path=self.app_path)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Dimensional reduction Cluster Exploration and Labeling Library')
        # Start Bokeh server in a separate thread
        self.bokeh_thread = Thread(target=self.dr_cell_server.start_server)
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
        # TODO fix bug where window does not close correctly
        # Stop the Bokeh server thread
        self.dr_cell_server.stop_server()
        self.bokeh_thread.join()  # Wait for the thread to finish
        self.q_app.quit()  # Exit the application
        event.accept()

import glob
import importlib
import os
import sys

from bokeh.application import Application
from bokeh.application.handlers import ScriptHandler
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

import drcell.util


class DrCELLBokehServer:
    def __init__(self, file_or_folder_path: str = sys.argv[0], port=5000, port_image=8000, app_path=None):
        self.initial_port = port
        self.port = port
        self.port_image = port_image
        self.app_path = app_path
        if self.app_path is None:
            # Load the script from the specified package and module
            package_name = "drcell"
            module_name = "drCELLBrokehApplication"

            spec = importlib.util.find_spec(f"{package_name}.{module_name}")
            if spec is None:
                raise ImportError(f"Module {package_name}.{module_name} not found")
            self.app_path = spec.origin
        self.dr_cell_file_paths = []
        if os.path.isdir(file_or_folder_path):
            for path in glob.glob(os.path.join(file_or_folder_path, '*.h5')):
                self.dr_cell_file_paths.append(os.path.abspath(path))
        elif os.path.isfile(file_or_folder_path):
            self.dr_cell_file_paths = [file_or_folder_path]
        self.server_instance = None

    def start_server(self):
        argv = ['--port-image', str(self.port_image), '--dr_cell_file_paths']
        argv.extend(self.dr_cell_file_paths)
        # Create a Bokeh application
        bokeh_app = Application(ScriptHandler(filename=self.app_path, argv=argv))
        self.port = self.initial_port
        while not drcell.util.generalUtil.is_port_available(self.port):
            print(f"Server port {self.port} is not available")
            self.port += 1
        print(f"Server port: {self.port}")
        # Create a Bokeh server
        self.server_instance = Server({'/': bokeh_app}, io_loop=IOLoop(), port=self.port)

        # Start the Bokeh server
        self.server_instance.start()
        print("Server started at localhost:" + str(self.server_instance.port))

        # Run the IOLoop to keep the server running
        self.server_instance.io_loop.start()

    def stop_server(self):
        if self.server_instance is not None:
            # Stop the Bokeh server
            self.server_instance.stop()
            self.server_instance.io_loop.stop()
            print("Server stopped")

import argparse
import importlib

from bokeh.application import Application
from bokeh.application.handlers.script import ScriptHandler
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

import drcell.util.generalUtil


class DrCELLBokehServer:
    def __init__(self, port=5000, port_image=8000, app_path=None):
        self.initial_port = port
        self.port = port
        self.port_image = port_image
        self.app_path = app_path
        if self.app_path is None:
            # Load the script from the specified package and module
            package_name = "drcell"
            module_name = "main"

            spec = importlib.util.find_spec(f"{package_name}.{module_name}")
            if spec is None:
                raise ImportError(f"Module {package_name}.{module_name} not found")
            self.app_path = spec.origin
        self.server_instance = None

    def start_server(self):
        argv = [str(self.port_image)]

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


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Bokeh server with custom app.")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Bokeh server")
    parser.add_argument("--port-image", type=int, default=8000, help="Port for the image server")
    parser.add_argument("--app-path", type=str, default=None, help="Path to the Bokeh application script")
    args = parser.parse_args()
    drcell_server = DrCELLBokehServer(port=args.port, port_image=args.port_image, app_path=args.app_path)
    drcell_server.start_server()
    input("Press Enter to stop the server...")
    drcell_server.stop_server()

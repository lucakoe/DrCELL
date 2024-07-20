from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.script import ScriptHandler
from tornado.ioloop import IOLoop
import argparse

import drcell.util.generalUtil

# Global variable to hold the server instance
server_instance = None


def run_server(port=5000, port_image=8000, app_path='main.py', skip_port_check=False):
    global server_instance
    if not skip_port_check:
        while not drcell.util.generalUtil.is_port_available(port):
            print(f"Server port {port} is not available")
            port += 1
        while not drcell.util.generalUtil.is_port_available(port_image):
            print(f"Image server port {port_image} is not available")
            port_image += 1

    print(f"Server port: {port}")
    print(f"Image server port: {port_image}")
    argv = [str(port_image)]

    # Create a Bokeh application
    bokeh_app = Application(ScriptHandler(filename=app_path, argv=argv))

    # Create a Bokeh server
    server_instance = Server({'/': bokeh_app}, io_loop=IOLoop(), port=port)

    # Start the Bokeh server
    server_instance.start()
    print("Server started at localhost:" + str(server_instance.port))

    # Run the IOLoop to keep the server running
    server_instance.io_loop.start()


def stop_server():
    global server_instance

    if server_instance is not None:
        # Stop the Bokeh server
        server_instance.stop()
        server_instance.io_loop.stop()
        print("Server stopped")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Bokeh server with custom app.")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Bokeh server")
    parser.add_argument("--port-image", type=int, default=8000, help="Port for the image server")
    parser.add_argument("--app-path", type=str, default="main.py", help="Path to the Bokeh application script")
    args = parser.parse_args()
    run_server(port=args.port, port_image=args.port_image, app_path=args.app_path)

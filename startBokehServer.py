from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.script import ScriptHandler
from tornado.ioloop import IOLoop
import argparse

# Global variable to hold the server instance
server_instance = None


def run_server(port=5000, app_path='main.py'):
    global server_instance

    # Create a Bokeh application
    bokeh_app = Application(ScriptHandler(filename=app_path))

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
    parser.add_argument("--app-path", type=str, default="main.py", help="Path to the Bokeh application script")
    args = parser.parse_args()
    run_server(port=args.port, app_path=args.app_path)

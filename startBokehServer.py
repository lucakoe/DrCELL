from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.script import ScriptHandler
from tornado.ioloop import IOLoop
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run Bokeh server with custom app.")
parser.add_argument("--port", type=int, default=5000, help="Port for the Bokeh server")
parser.add_argument("--app-path", type=str, default="main.py", help="Path to the Bokeh application script")
args = parser.parse_args()
# Create a Bokeh application
bokeh_app = Application(ScriptHandler(filename=args.app_path))

# Create a Bokeh server
server = Server({'/': bokeh_app}, io_loop=IOLoop(), port=args.port)

# Start the Bokeh server
server.start()
print("Server started at localhost:" + str(server.port))

# Run the IOLoop to keep the server running
server.io_loop.start()

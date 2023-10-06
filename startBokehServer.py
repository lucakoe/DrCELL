from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.server.server import Server
import misc


if __name__ == '__main__':



    app = Application(FunctionHandler())
    server = Server({'/': app}, num_procs=1, allow_websocket_origin=["localhost:5000"])
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()

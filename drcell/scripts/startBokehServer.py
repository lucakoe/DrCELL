import argparse

from drcell import DrCELLBokehServer

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

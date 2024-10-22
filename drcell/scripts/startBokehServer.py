import argparse
import sys

from drcell import DrCELLBokehServer


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Bokeh server with custom app.")
    parser.add_argument("--port", type=int, default=5000, help="Port for the Bokeh server")
    parser.add_argument("file_or_folder_path", type=str, default=sys.argv[0],
                        help="Path to the DrCELL file or folder containing the DrCELL files.")
    parser.add_argument("--port-image", type=int, default=8000, help="Port for the image server")
    parser.add_argument("--app-path", type=str, default=None, help="Path to the Bokeh application script")
    args = parser.parse_args()
    drcell_server = DrCELLBokehServer(file_or_folder_path=args.file_or_folder_path, port=args.port,
                                      port_image=args.port_image, app_path=args.app_path)
    drcell_server.start_server()
    input("Press Enter to stop the server...")
    drcell_server.stop_server()


if __name__ == '__main__':
    main()

# for development
import argparse

from drcell.server.ImageServer import ImageServer


def main():
    parser = argparse.ArgumentParser(description='Start an image generation server.')
    parser.add_argument('--port', type=int, default=8000, help='Port number for the server (default: 8000)')
    args = parser.parse_args()

    image_server_instance = ImageServer(port=args.port)
    image_server_instance.start_server()

    input("Press Enter to stop the server...")
    image_server_instance.stop_server()


if __name__ == '__main__':
    main()

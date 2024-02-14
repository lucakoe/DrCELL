import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from PIL import Image, ImageDraw
import io
import threading

import plotting


class RequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.image_cache = {}
        self.current_dataset = None
        super().__init__(*args, **kwargs)

    def do_GET(self):
        try:
            # Parse the URL to extract query parameters
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            # Check if the 'generate' parameter is present
            if 'generate' in query_params:
                # Get the value of the 'generate' parameter
                generate_param = query_params['generate'][0]

                # Generate or retrieve the image based on the parameter
                extend_plot = False
                # if query_params['extend-plot'][0] is "True":
                #     extend_plot =True
                image_content = self.get_or_generate_image(generate_param, extend_plot)

                # Specify the content type as image/jpeg
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()

                # Send the generated image in the response
                self.wfile.write(image_content)
            elif self.path == '/clear_cache':
                # Handle clearing the cache
                self.clear_cache()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'Cache cleared successfully')
            else:
                self.send_error(400, 'Bad Request: Missing generate parameter')

        except Exception as e:
            self.send_error(500, 'Internal Server Error: {}'.format(str(e)))

    def get_or_generate_image(self, parameter, extend_plot=False):
        # Check if the image for the given parameter is already generated
        if parameter in self.image_cache:
            return self.image_cache[parameter + str(extend_plot)]

        # Generate the image based on the parameter
        image_content = self.generate_image(parameter, extend_plot)

        # Cache the generated image
        self.image_cache[parameter + str(extend_plot)] = image_content

        return image_content

    def generate_image(self, parameter, extend_plot):
        # Split the parameter string into integers
        parameter = [int(x.strip()) for x in parameter.split(',')]
        # # Example: Generate a simple image with bars based on the parameter
        # width, height = 300, 200
        # image = Image.new('RGB', (width, height), color='white')
        # draw = ImageDraw.Draw(image)
        #

        #
        # # Draw bars based on the values
        # num_values = len(values)
        # bar_width = width // num_values
        # max_value = max(values)
        # for i, value in enumerate(values):
        #     bar_height = int(height * value / max_value)
        #     x0 = i * bar_width
        #     y0 = height - bar_height
        #     x1 = (i + 1) * bar_width - 1
        #     y1 = height - 1
        #     draw.rectangle([x0, y0, x1, y1], fill='blue')

        # Save the image to a bytes buffer
        image_bytes = io.BytesIO()
        # Save the plot to the BytesIO object as a JPEG image
        plt = plotting.get_plot_for_indices_of_current_dataset(parameter, fps=30, number_consecutive_recordings=6,
                                                               extend_plot=extend_plot)
        plt.savefig(image_bytes, format='jpg')
        plt.close('all')
        image_content = image_bytes.getvalue()

        return image_content

    def clear_cache(self):
        # Clear the image cache
        self.image_cache.clear()

    def update_dataset(self, new_dataset):
        # Update the current dataset
        self.current_dataset = new_dataset


def start_server(port):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f'Starting server on port {port}')

    # Start the server in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True  # So the thread will be terminated when the main program exits
    server_thread.start()

    # Return the server object to allow manipulation of the server if needed
    return httpd


def stop_server(httpd):
    if httpd:
        print('Stopping server...')
        httpd.shutdown()
        httpd.server_close()
        print('Server stopped')
    else:
        print('Server is not running')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start an image generation server.')
    parser.add_argument('--port', type=int, default=8000, help='Port number for the server (default: 8000)')
    args = parser.parse_args()

    httpd = None
    server_thread = threading.Thread(target=start_server, args=(args.port,))
    server_thread.start()

    input("Press Enter to stop the server...")
    stop_server()

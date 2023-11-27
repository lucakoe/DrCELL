from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import os

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Parse the URL to extract query parameters
            parsed_url = urlparse(self.path)
            query_params = parse_qs(parsed_url.query)

            # Check if the 'image' parameter is present
            if 'image' in query_params:
                image_name = query_params['image'][0]
                image_path = os.path.normpath(r'C:\Users\koenig\OneDrive - Students RWTH Aachen University\Bachelorarbeit\GitHub\twoP\Playground\Luca\PlaygoundProject\data\Umap_2530_2532MedianChoiceStim30trials_Array\plot_images\{}'.format(image_name))

                # Check if the specified image file exists
                if os.path.exists(image_path):
                    # Specify the content type based on the image file extension
                    content_type = 'image/jpeg' if image_name.endswith('.jpg') else 'image/png'

                    # Send the response with the specified image
                    self.send_response(200)
                    self.send_header('Content-type', content_type)
                    self.end_headers()

                    with open(image_path, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    self.send_error(404, 'File Not Found: {}'.format(image_path))
            else:
                self.send_error(400, 'Bad Request: Missing image parameter')

        except Exception as e:
            self.send_error(500, 'Internal Server Error: {}'.format(str(e)))

if __name__ == '__main__':
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, RequestHandler)
    print('Starting server on port 8000')
    httpd.serve_forever()
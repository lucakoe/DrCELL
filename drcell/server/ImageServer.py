import io
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from drcell.util.generalUtil import is_port_available
from drcell.util.plottingUtil import get_pca_plot_for_indices, get_plot_for_indices


class ImageServer:
    def __init__(self, port=8000, current_dataset=None, current_pca_preprocessed_dataset=None):
        self.port = port
        while not is_port_available(self.port):
            print(f"Image Server port {self.port} is not available")
            self.port += 1
        print(f"Image server port: {self.port}")
        server_address = ('', self.port)
        # Inject the outer class instance into the request handler
        self.request_handler = self.RequestHandler
        self.RequestHandler.server_instance = self
        self.server = HTTPServer(server_address, self.request_handler)
        self.current_dataset = current_dataset
        self.current_pca_preprocessed_dataset = current_pca_preprocessed_dataset
        self.server_thread = None

    class RequestHandler(BaseHTTPRequestHandler):
        image_cache = {}
        server_instance = None

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
                    pca_preprocessing = False
                    recording_type = "None"
                    if 'extend-plot' in query_params and str(query_params['extend-plot'][0]) == 'True':
                        extend_plot = True

                    if 'pca-preprocessing' in query_params and str(query_params['pca-preprocessing'][0]) == 'True':
                        pca_preprocessing = True

                    if 'recording-type' in query_params:
                        recording_type = query_params['recording-type'][0]

                    image_content = self.get_or_generate_image(generate_param, extend_plot,
                                                               pca_preprocessing=pca_preprocessing,
                                                               recording_type=recording_type)

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

        def get_or_generate_image(self, parameter, extend_plot=False, pca_preprocessing=False, recording_type=None):
            # Check if the image for the given parameter is already generated
            if parameter in self.image_cache:
                return self.image_cache[parameter + str(extend_plot) + str(pca_preprocessing) + str(recording_type)]

            # Generate the image based on the parameter
            image_content = self.generate_image(parameter, extend_plot, pca_preprocessing=pca_preprocessing,
                                                recording_type=recording_type)

            # Cache the generated image
            self.image_cache[
                parameter + str(extend_plot) + str(pca_preprocessing) + str(recording_type)] = image_content

            return image_content

        def generate_image(self, parameter, extend_plot, pca_preprocessing=False, recording_type=None):
            # Split the parameter string into integers
            parameter = [int(x.strip()) for x in parameter.split(',')]

            # Save the image to a bytes buffer
            image_bytes = io.BytesIO()
            # Save the plot to the BytesIO object as a JPEG image
            # TODO adjust pca plotting accordingly with correct axis etc.
            # for alice 12 recordings and 10 fps
            plt = self.server_instance.get_plot_for_indices_of_current_dataset(parameter, fps=30,
                                                                               number_consecutive_recordings=1,
                                                                               extend_plot=extend_plot,
                                                                               pca_preprocessing=pca_preprocessing,
                                                                               recording_type=recording_type)
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

    def update_pca_preprocessed_dataset(self, new_dataset):
        self.current_pca_preprocessed_dataset = new_dataset

    def start_server(self):
        print(f'Starting server on port {self.port}')

        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

    def stop_server(self):
        if self.server:
            print('Stopping server...')
            self.server.shutdown()
            self.server.server_close()
            print('Server stopped')
        else:
            print('Server is not running')

    def get_plot_for_indices_of_current_dataset(self, indices, fps=30, number_consecutive_recordings=1,
                                                extend_plot=False,
                                                pca_preprocessing=False, recording_type=None):
        dataset = self.current_dataset
        if pca_preprocessing:
            dataset = self.current_pca_preprocessed_dataset
            return get_pca_plot_for_indices(dataset, indices,
                                            extend_plot=extend_plot)

            # TODO adjust pca plotting accordingly with correct axis etc.
        return get_plot_for_indices(dataset, indices, fps=fps,
                                    number_consecutive_recordings=number_consecutive_recordings,
                                    extend_plot=extend_plot,
                                    recording_type=recording_type)

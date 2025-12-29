import ngrok
import time
import argh
import os


class Forwarder:

    @argh.arg('--port', type=int, required=True, help='local port to forward')
    def run(self, port: int = -1):
        token = os.environ['NGROK_AUTHTOKEN']
        password = os.environ['NGROK_PASSWORD']
        domain = os.environ.get('NGROK_DOMAIN', None)

        listener = ngrok.forward(
            port,
            authtoken=token,
            domain=domain,
            basic_auth=f'test:{password}',
        )

        print(f'ingress established at {listener.url()}')

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print('exiting')
            ngrok.disconnect()


if __name__ == '__main__':
    forwarder = Forwarder()

    parser = argh.ArghParser()
    argh.add_commands(parser, [forwarder.run])

    argh.dispatch(parser)

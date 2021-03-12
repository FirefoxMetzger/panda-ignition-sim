import zmq
import subprocess
import socket
import os
import pwd

class IgnSubscriber(object):
    def __init__(self, topic, *, context=zmq.Context()):
        self.socket = context.socket(zmq.SUB)
        self.topic = topic

        # this could also be streamlined by speaking the ign-transport discovery protcol
        host_name = socket.gethostname()
        user_name = pwd.getpwuid(os.getuid())[0]
        self.socket.subscribe(f"@/{host_name}:{user_name}@{topic}")

    def recv(self, *args):
        return self.socket.recv_multipart(*args)

    def __enter__(self):
        # weird hack to encourage ign-transport to actually publish camera messages
        # start an echo subscriber and print the messages into the void to make ign 
        # realize that something is listening to the topic
        # tracking issue: https://github.com/ignitionrobotics/ign-transport/issues/225
        self.echo_subscriber = subprocess.Popen(
            ["ign", "topic", "-e", "-t", self.topic], 
            stdout=open(os.devnull, 'w')
        )

        # this is a bad hack and should be implemented
        # by talking the ign-transport discovery protocol
        result = subprocess.check_output(f"ign topic -i -t {self.topic}", shell=True)
        self.address = result.decode("utf-8").split("\n")[1].split(",")[0].replace("\t", "").replace(" ", "")
        self.socket.connect(self.address)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.socket.disconnect(self.address)
        self.echo_subscriber.terminate()

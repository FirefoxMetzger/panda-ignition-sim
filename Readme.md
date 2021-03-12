# Simulation Environment for Legibility with Panda

## Installation

1. Install `gym-ignition` following the [official instructions](https://github.com/robotology/gym-ignition#setup)
2. `pip install pyzmq numpy matplotlib`
3. Compile python bindings for ignition's protobuf messages (see below)


## Execution

1. `python3 simulator.py`

The script will launch the simulator, register listeners for sensor data, and run a small simulation.

## Python Bindings for Ignition Protobuf Messages

Ignitions (ign-transport) communicates via zmq using protobuf messages. Creating python bindings
for the protobuf objects allows listening to ignition (and in particular sensor data) directly from python.

1. `git clone https://github.com/ignitionrobotics/ign-msgs.git`
    - Make sure you are in a different directory than this one (don't stack git projects within each other)
2. `cd ign-msgs/proto`
3. `protoc --python_out=<path/to/panda-python-sim> *.proto`

This should create a folder `ignition/msgs` inside `panda-python-sim` containing a lot of `<msg>_pb2.py` files.
Each file allows passing `<msg>`.
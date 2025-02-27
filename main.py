import os
import simulator
import json

from datetime import datetime
from dateutil import tz
from argparse import ArgumentParser
from log import init_logs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config_path", type=str, help="Configuration file path")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--out_path", type=str, help="Name of the output path folder")
    parser.add_argument(
        "--gen_type", type=str, help="Type of traffic generation (dynamic|static|progressive)", default="dynamic"
    )
    parser.add_argument("--days", type=int, help="Number of days to simulate", default=30)
    arguments = parser.parse_args()

    with open(arguments.config_path, "r") as f:
        parameters = json.load(f)

    timestamp = datetime.now(tz.tzlocal()).strftime("%Y_%m_%d_%H_%M_%S")

    if arguments.out_path:
        timestamp += "_" + arguments.out_path

    output_path = os.path.join("simulations", timestamp)

    os.makedirs(output_path)

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(parameters, f, indent=4)

    init_logs(os.path.join(output_path, "log.txt"), "at")

    print(arguments)

    simulator = simulator.Simulator(parameters, output_path, arguments.gen_type, arguments.model_path)
    simulator.run(arguments.days * 1440.00000001)

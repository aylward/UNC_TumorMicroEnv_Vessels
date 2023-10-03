import sys
import time
import argparse
import traceback

from usangio_vessels import usangio_vessels


def usangio_vessels_func(
    input_data,
    level_of_detail=1,
):
    app = usangio_vessels()
    app.import_matlab_data(input_data)
    app.run(level_of_detail)

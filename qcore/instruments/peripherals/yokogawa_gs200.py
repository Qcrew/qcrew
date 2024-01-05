""" """

import time

import numpy as np
import pyvisa

from qcore.instruments.instrument import Instrument, ConnectionError


class GS200(Instrument):
    """Python driver to use Yokogawa GS200 as a current source"""

    # GS200 takes 10ms to change source level and 20ms for output relay to stabilize
    WAIT_TIME = 0.1  # we set it to 0.1s to be safe

    def __init__(
        self,
        name: str,
        id: str,
        current: float = 0.0,  # Ampere
        output: bool = False,
    ) -> None:
        """ """
        self._handle: pyvisa.resources.Resource = None
        super().__init__(id=id, name=name, current=current, output=output)

        # assume qcrew uses GS200 as a current source, so we hard code this for now
        self._handle.write("source:function current")

    def connect(self) -> None:
        """ """
        if self._handle is not None:
            self.disconnect()
        resource_name = f"USB::0xB21::0x39::{self.id}::INSTR"
        try:
            self._handle = pyvisa.ResourceManager().open_resource(resource_name)
        except pyvisa.errors.VisaIOError as err:
            details = f"{err.abbreviation} : {err.description}"
            raise ConnectionError(f"Failed to connect {self}, {details = }") from None

    def disconnect(self) -> None:
        """ """
        self._handle.close()

    @property
    def status(self) -> bool:
        """ """
        try:
            self._handle.query("*IDN?")
        except (pyvisa.errors.VisaIOError, pyvisa.errors.InvalidSession):
            return False
        else:
            return True

    @property
    def current(self) -> float:
        """ """
        return float(self._handle.query(":source:level?"))

    @current.setter
    def current(self, value: float) -> None:
        """ """
        self._handle.write(f":source:level:auto {value}")
        time.sleep(GS200.WAIT_TIME)

    @property
    def output(self) -> bool:
        """ """
        return bool(int(self._handle.query(":output?")))

    @output.setter
    def output(self, value: bool) -> None:
        """ """
        self._handle.write(f"output {int(bool(value))}")
        time.sleep(GS200.WAIT_TIME)

    def ramp(self, stop, start=None, step=1e-4) -> None:
        """ """
        if start is None:  # start from the current level set right now
            start = self.current

        if start > stop:  # ramp down
            points = np.arange(stop, start + step / 2, step)[::-1]  # include endpoint
        else:  # ramp up
            points = np.arange(start, stop + step / 2, step)

        for point in points:
            self.current = point
            time.sleep(GS200.WAIT_TIME)

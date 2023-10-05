import time

import numpy as np
import pyvisa

from qcore.instruments.instrument import Instrument, ConnectionError


class APUASYN20(Instrument):
    # Without the Fast Switching (FS) addon, the worse case switching speed is 500us. The typical speed is 200us.
    WAIT_TIME = 500e-6

    def __init__(
        self,
        id: str,
        channel: int = 0,
        frequency: float = 0.0,
        phase: float = 0.0,
        power: float = 0.0,
        output: bool = False,
    ) -> None:
        """ """
        self._handle: pyvisa.resource.Resource = None
        super().__init__(
            id=id,
            channel=channel,
            frequency=frequency,
            phase=phase,
            power=power,
            output=output,
        )

    def connect(self) -> None:
        """ """
        if self._handle is not None:
            self.disconnect()
        resource_name = f"TCPIP0::{self.id}::inst0::INSTR"
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
    def channel(self) -> int:
        """Returns the current active channel"""
        return int(self._handle.query(f":SEL?"))

    @channel.setter
    def channel(self, value: int) -> None:
        """Sets active channel"""
        self._handle.write(f":SEL {value}")

    @property
    def frequency(self) -> float:
        """Returns freq in Hz"""
        return float(self._handle.query(f":FREQ:CW?"))

    @frequency.setter
    def frequency(self, value: float) -> None:
        """Writes frequency in Hz"""
        self._handle.write(f":FREQ:CW {value}")
        time.sleep(APUASYN20.WAIT_TIME)

    @property
    def phase(self) -> float:
        """Returns phase in rad"""
        return float(self._handle.query(f":PHAS:ADJ?"))

    @phase.setter
    def phase(self, value: float):
        """Writes phase in rad"""
        self._handle.write(f":PHAS:ADJ {value}")
        time.sleep(APUASYN20.WAIT_TIME)

    @property
    def power(self) -> float:
        """Returns power in dBm"""
        return float(self._handle.query(f":POW?"))

    @power.setter
    def power(self, value: float):
        """Sets power in dBm"""
        self._handle.write(f":POW {value}")
        time.sleep(APUASYN20.WAIT_TIME)

    @property
    def output(self) -> bool:
        """ """
        return bool(self._handle.query(f"OUTP?"))

    @output.setter
    def output(self, value: bool) -> None:
        """ """
        if value:
            self._handle.write(f"OUTP ON")
        else:
            self._handle.write(f"OUTP OFF")

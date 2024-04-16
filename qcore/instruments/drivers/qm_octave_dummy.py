import os
from qm.octave import QmOctaveConfig

from qcore.instruments.instrument import Instrument
from qcore.variables.parameter import Parameter


class Octave(Instrument):
    """Dummy instrument containing relevant information for connecting to an Octave."""

    settings: dict = Parameter()
    # uses_opx_plus: bool = Parameter()
    calibration_db_path: str = Parameter()
    port: int = Parameter()

    def __init__(
        self,
        settings: dict,
        # uses_opx_plus: bool,
        calibration_db_path: str,
        port: int,
        id: str,
        **parameters
    ):
        self._settings = settings
        # self._uses_opx_plus = uses_opx_plus
        self._calibration_db_path = calibration_db_path
        self._port = port
        super().__init__(id, **parameters)

    @property
    def status(self) -> bool:
        return True

    @calibration_db_path.setter
    def calibration_db_path(self, value: str) -> None:
        """ """
        self.calibration_db_path = value

    @calibration_db_path.getter
    def calibration_db_path(self) -> bool:
        """ """
        return self._calibration_db_path

    # @uses_opx_plus.getter
    # def uses_opx_plus(self) -> bool:
    #     """ """
    #     return self._uses_opx_plus

    # @uses_opx_plus.setter
    # def uses_opx_plus(self, value: bool) -> None:
    #     """ """
    #     self._uses_opx_plus = value

    @settings.getter
    def settings(self) -> dict:
        """ """
        return self._settings

    @settings.setter
    def settings(self, value: dict) -> None:
        """ """
        self._settings = value

    @port.getter
    def port(self) -> int:
        """ """
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        """ """
        self._port = value

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

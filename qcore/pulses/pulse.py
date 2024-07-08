""" """

from typing import Any, Union

from qcore.helpers.logger import logger

from qcore.pulses.digital_waveform import DigitalWaveform
from qcore.resource import Resource

class Pulse(Resource):
    """ """

    BASE_AMP = 0.2  # in V
    CLOCK_CYCLE = 4  # in ns

    def __init__(
        self,
        name: str,
        length: Union[None, int],
        I_ampx: float,
        Q_ampx: Union[None, float],  # set None for single waveform pulses
        pad: int,
        digital_marker: Union[DigitalWaveform, None] = None,
        **parameters,
    ) -> None:
        """ """        
        self.length: int = length
        self.pad: int = pad

        self.I_ampx: float = I_ampx
        self.Q_ampx: float = Q_ampx

        self._digital_marker = digital_marker

        super().__init__(name=name, **parameters)

    @property
    def total_length(self) -> int:
        """ """
        return self.length + self.pad

    def has_mixed_waveforms(self) -> bool:
        """ """
        return self.I_ampx is not None and self.Q_ampx is not None

    def sample(
        self,
    ) -> Union[tuple[float, Union[float, None]], tuple[list, Union[list, None]]]:
        """ """
        raise NotImplementedError("Subclasses must implement 'sample()'.")

    @property
    def digital_marker(self) -> Union[DigitalWaveform, None]:
        """ """
        return self._digital_marker

    @digital_marker.setter
    def digital_marker(self, value: Union[DigitalWaveform, None]) -> None:
        """ """
        if value is not None and not isinstance(value, DigitalWaveform):
            raise ValueError(f"Invalid {value = }, must be {DigitalWaveform}.")
        self._digital_marker = value
        logger.debug(f"Set {self} digital marker to {value}.")

    def snapshot(self, flatten=False) -> dict[str, Any]:
        """ """
        snapshot = super().snapshot()
        if flatten:
            digital_marker = snapshot["digital_marker"]
            if digital_marker is not None:
                snapshot["digital_marker"] = digital_marker.snapshot()
        return snapshot

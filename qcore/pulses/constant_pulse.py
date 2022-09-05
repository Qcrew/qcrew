""" """

import numpy as np

from qcore.pulses.pulse import Pulse


class ConstantPulse(Pulse):
    """ """

    def __init__(
        self,
        name: str = "constant_pulse",
        length: int = 1000,  # in ns
        I_ampx: float = 1.0,
        Q_ampx: None | float = 0.0,
        pad: int = 0,
    ) -> None:
        """ """
        super().__init__(name, length=length, I_ampx=I_ampx, Q_ampx=Q_ampx, pad=pad)

    @property
    def total_I_amp(self) -> float:
        """ """
        return Pulse.BASE_AMP * self.I_ampx

    def sample(self) -> tuple[float, float | None] | tuple[list, list | None]:
        """ """
        has_constant_waveform = not self.pad
        if has_constant_waveform:
            return self._sample_constant_waveform()
        else:
            return self._sample_arbitrary_waveform()

    def _sample_constant_waveform(self) -> tuple[float, float | None]:
        """ """
        total_amp = self.total_I_amp
        return (total_amp, 0.0) if self.has_mixed_waveforms() else (total_amp, None)

    def _sample_arbitrary_waveform(self) -> tuple[list, list | None]:
        """ """
        samples = np.ones(self.length)
        pad = np.zeros(self.pad) if self.pad else []

        i_wave = np.concatenate((samples, pad)) * self.total_I_amp
        q_wave = np.zeros(self.total_length)

        return (i_wave, q_wave) if self.has_mixed_waveforms() else (i_wave, None)
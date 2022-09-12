""" """

from qcore.elements.mode import Mode
from qcore.pulses.ramped_constant_pulse import ConstantPulse
from qcore.pulses.readout_pulse import ConstantReadoutPulse


class Readout(Mode):
    """ """

    PORTS_KEYS = (*Mode.PORTS_KEYS, "out")
    OFFSETS_KEYS = (*Mode.OFFSETS_KEYS, "out")

    def __init__(self, tof: int = 180, smearing: int = 0, **parameters) -> None:
        """ """
        self.tof: int = tof
        self.smearing: int = smearing

        if "operations" not in parameters:
            default_operations = [
                ConstantPulse("constant_pulse"),
                ConstantReadoutPulse("readout_pulse"),
            ]
            parameters["operations"] = default_operations

        super().__init__(**parameters)

    def measure(self) -> None:
        """ """
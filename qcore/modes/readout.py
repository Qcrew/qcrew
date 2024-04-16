""" """

from qcore.modes.mode import Mode
from qcore.pulses.digital_waveform import DigitalWaveform
from qcore.pulses.pulse import Pulse
from qcore.pulses.ramped_constant_pulse import ConstantPulse
from qcore.pulses.readout_pulse import ConstantReadoutPulse
from qm import qua
from qcore.helpers.logger import logger


class Readout(Mode):
    """ """

    # out1 and out2 have been added to enable I AND Q input to the OPX
    PORTS_KEYS = (*Mode.PORTS_KEYS, "out", "out1", "out2")
    OFFSETS_KEYS = (*Mode.OFFSETS_KEYS, "out", "out1", "out2")

    DEMOD_METHOD_MAP = {
        "sliced": qua.demod.sliced,
        "accumulated": qua.demod.accumulated,
        "window": qua.demod.moving_window,
    }

    def __init__(self, tof: int = 180, smearing: int = 0, **parameters) -> None:
        """ """
        self.tof: int = tof
        self.smearing: int = smearing

        if "operations" not in parameters:
            default_operations = [
                ConstantPulse("constant_pulse"),
                ConstantReadoutPulse(
                    "readout_pulse", digital_marker=DigitalWaveform("ADC_ON")
                ),
            ]
            parameters["operations"] = default_operations

        super().__init__(**parameters)

    def measure(
        self,
        pulse: Pulse,
        targets: tuple = None,
        ampx=1.0,
        stream: str = None,
        demod_type: str = "full",
        demod_args: tuple = None,
    ) -> None:
        """ """
        op_name = self._pulse_op_map[pulse.name]
        try:
            num_ampxs = len(ampx)
            if num_ampxs != 4:
                logger.error("Ampx must be a sequence of 4 values")
                raise ValueError(f"Invalid ampx value count, expect 4, got {num_ampxs}")
        except TypeError:
            num_ampxs = 1

        if stream is not None:
            if num_ampxs == 1:
                qua.measure(op_name * qua.amp(ampx), self.name, stream)
            else:
                qua.measure(op_name * qua.amp(*ampx), self.name, stream)

        elif targets is not None:
            var_i, var_q = targets

            if demod_type == "full":
                output_i, output_q = ("cos", var_i), ("sin", var_q)
                demod_i, demod_q = qua.demod.full(*output_i), qua.demod.full(*output_q)
            elif demod_type == "dual":
                demod_i = qua.dual_demod.full("cos", "out1", "sin", "out2", var_i)
                demod_q = qua.dual_demod.full("minus_sin", "out1", "cos", "out2", var_q)
            else:
                try:
                    demod_method = self.DEMOD_METHOD_MAP[demod_type]
                except KeyError:
                    logger.error(f"Unrecognized demod type '{demod_type}'")
                    raise
                else:
                    output_i = ("cos", var_i, *demod_args)
                    output_q = ("sin", var_q, *demod_args)
                    demod_i, demod_q = demod_method(*output_i), demod_method(*output_q)

            if num_ampxs == 1:
                qua.measure(op_name * qua.amp(ampx), self.name, None, demod_i, demod_q)
            else:
                qua.measure(op_name * qua.amp(*ampx), self.name, None, demod_i, demod_q)

    def iw_training_measure(
        self,
        pulse: Pulse,
        division_length: int,
        targets: tuple = None,
        ampx=1.0,
    ) -> None:

        op_name = self._pulse_op_map[pulse.name]
        var_ii, var_iq, var_qi, var_qq = targets

        demod_ii = qua.demod.sliced("cos", var_ii, division_length, "out1")
        demod_iq = qua.demod.sliced("sin", var_iq, division_length, "out2")
        demod_qi = qua.demod.sliced("minus_sin", var_qi, division_length, "out1")
        demod_qq = qua.demod.sliced("cos", var_qq, division_length, "out2")

        qua.measure(
            op_name * qua.amp(ampx),
            self.name,
            None,
            demod_ii,
            demod_iq,
            demod_qi,
            demod_qq,
        )

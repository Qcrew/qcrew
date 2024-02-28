""" Python driver for Anritsu VNA MS46522B """

import pyvisa

from qcore.instruments.instrument import Instrument, ConnectionError
from qcore.helpers.logger import logger


class MS46522B(Instrument):
    """ """

    HEADER_LENGTH = 11  # number of characters to ignore when querying data
    MIN_SWEEP_POINTS, MAX_SWEEP_POINTS = 2, 20001
    MIN_SWEEP_DELAY, MAX_SWEEP_DELAY = 0, 100
    MIN_BANDWIDTH, MAX_BANDWIDTH = 1, 500000
    MIN_TRACES, MAX_TRACES = 1, 16
    VALID_S_PARAMETERS = ("s11", "s12", "s21", "s22")
    # we use a restricted set of trace formats, expand the set if needed in future
    VALID_TRACE_FORMATS = ("imag", "mlog", "phase", "real")
    DEFAULT_TRACE_LAYOUT = {
        1: "R1C1",
        2: "R1C2",
        **dict.fromkeys([3, 4], "R2C2"),
        **dict.fromkeys([5, 6], "R2C3"),
        **dict.fromkeys([7, 8], "R2C4"),
        9: "R3C3",
        10: "R2C5",
        **dict.fromkeys([11, 12], "R4C3"),
        **dict.fromkeys([13, 14, 15, 16], "R4C4"),
    }

    def __init__(self, id: str, name: str = "VNA", **parameters) -> None:
        """ """
        self._handle = None
        super().__init__(id=id, name=name, **parameters)

        self._traces: list[tuple[str, str]] = None
        self.traces = [("s21", "real"), ("s21", "imag")]
        self._handle.timeout = None  # better to have no timeout for very long sweeps
        self._handle.write(":sense:average:type sweepbysweep")  # enforce default
        self._handle.write(":sweep:type linear")  # for now, only support linear sweeps
        self._handle.write(":sense:sweep:delay:state 1")  # toggle per point sweep delay
        # for now, we disable averaging so sweep() returns consistent results
        self._handle.write(":sense:average:clear")
        self._handle.write(f":sense:average:state 0")

    def connect(self) -> None:
        """ """
        if self._handle is not None:
            self.disconnect()
        resource_name = f"TCPIP0::{self.id}::INSTR"
        try:
            self._handle = pyvisa.ResourceManager().open_resource(resource_name)
        except pyvisa.errors.VisaIOError as err:
            details = f"{err.abbreviation} : {err.description}"
            raise ConnectionError(f"Failed to connect {self}, {details = }") from None

    def disconnect(self) -> None:
        """ """
        self._handle.close()

    def configure(self, **config) -> None:
        """ """
        self.hold()

        for parameter, value in config.items():
            if hasattr(self, parameter):
                setattr(self, parameter, value)

    @property
    def datakeys(self):
        """ """
        return [f"{s_param}_{trace_format}" for s_param, trace_format in self._traces]

    @property
    def frequencies(self) -> list:
        """ """
        freq_str = self._handle.query(":sense:frequency:data?")[
            MS46522B.HEADER_LENGTH :
        ]
        return [float(freq) for freq in freq_str.split()]

    @property
    def status(self) -> bool:
        """ """
        try:
            self._handle.query("*IDN?")
        except (pyvisa.errors.VisaIOError, pyvisa.errors.InvalidSession):
            return False
        else:
            return True

    def sweep(self) -> tuple[list[float], dict[str, list[float]]]:
        """ """
        logger.info("Doing sweep ...")
        self._handle.write(":trigger:single")  # trigger single sweep
        logger.info("Rescaling traces...")
        self._handle.write(":display:window:y:auto")  # auto-scale all traces
        self.hold()
        logger.info("Done with sweep, retrieving info...")

        slc = MS46522B.HEADER_LENGTH  # start of data slice
        # freqstr = self._handle.query(":sense:frequency:data?")[slc:]
        # freqs = [float(freq) for freq in freqstr.split()]

        datakeys = [f"{s_param}_{trace_fmt}" for s_param, trace_fmt in self._traces]
        data = dict.fromkeys(datakeys)
        for count, key in enumerate(datakeys, start=1):
            logger.info(f"Setting param to {count}...")
            self._handle.write(f":calculate:parameter{count}:select")
            logger.info("Querying data...")
            datastr = self._handle.query(":calculate:data:fdata?")[slc:]
            data[key] = [float(value) for value in datastr.split()]

        return data

    def hold(self) -> None:
        """ """
        self._handle.write(":sense:hold:function hold")

    def disconnect(self) -> None:
        """ """
        self._handle.close()

    def _check_bounds(self, value: float, min: float, max: float, key: str) -> None:
        """ """
        if not min <= value <= max:
            raise ValueError(f"{key} {value = } out of bounds: [{min}, {max}].")

    def clear_avg(self) -> None:
        self._handle.write(":sense:average:clear")

    @property
    def avg_type(self) -> str:
        return str(self._handle.query(":sense:average:type?"))

    @avg_type.setter
    def avg_type(self, type: str):
        if type == "sweep":
            self._handle.write(":sense:average:type sweepbysweep")
        elif type == "point":
            self._handle.write(":sense:average:type pointbypoint")
        else:
            raise ValueError(
                f'Unknown averaging type {type}. Averaging type can only be by "sweep" or "point".'
            )

    @property
    def is_averaging(self) -> bool:
        return bool(self._handle.query(":sense:average:state?"))

    @is_averaging.setter
    def is_averaging(self, turn_on: bool):
        if turn_on:
            self._handle.write(":sense:average:state 1")
        else:
            self._handle.write(":sense:average:state 0")

    @property
    def avg_count(self) -> int:
        return int(self._handle.query(":sense:average:count?"))

    @avg_count.setter
    def avg_count(self, count: int):
        self._handle.write(f":sense:average:count {count}")

    @property
    def fcenter(self) -> float:
        """ """
        return float(self._handle.query(":sense:frequency:center?"))

    @fcenter.setter
    def fcenter(self, value: float) -> None:
        """ """
        self._handle.write(f":sense:frequency:center {value}")

    @property
    def fspan(self) -> float:
        """ """
        return float(self._handle.query(":sense:frequency:span?"))

    @fspan.setter
    def fspan(self, value: float) -> None:
        """ """
        self._handle.write(f":sense:frequency:span {value}")

    @property
    def fstart(self) -> float:
        """ """
        return float(self._handle.query(":sense:frequency:start?"))

    @fstart.setter
    def fstart(self, value: float) -> None:
        """ """
        self._handle.write(f":sense:frequency:start {value}")

    @property
    def fstop(self) -> float:
        """ """
        return float(self._handle.query(":sense:frequency:stop?"))

    @fstop.setter
    def fstop(self, value: float) -> None:
        """ """
        self._handle.write(f":sense:frequency:stop {value}")

    @property
    def bandwidth(self) -> float:
        """ """
        return float(self._handle.query(":sense:bandwidth?"))

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        """ """
        min, max = MS46522B.MIN_BANDWIDTH, MS46522B.MAX_BANDWIDTH
        self._check_bounds(value, min, max, "Bandwidth")
        self._handle.write(f":sense:bandwidth {value}")

    @property
    def sweep_delay(self) -> float:
        """ """
        return float(self._handle.query(":sense:sweep:delay?"))

    @sweep_delay.setter
    def sweep_delay(self, value: float) -> None:
        """ """
        min, max = MS46522B.MIN_SWEEP_DELAY, MS46522B.MAX_SWEEP_DELAY
        self._check_bounds(value, min, max, "Sweep delay")
        self._handle.write(f":sense:sweep:delay {value}")

    @property
    def sweep_points(self) -> int:
        """ """
        return int(self._handle.query(":sense:sweep:point?"))

    @sweep_points.setter
    def sweep_points(self, value: int) -> None:
        """ """
        min, max = MS46522B.MIN_SWEEP_POINTS, MS46522B.MAX_SWEEP_POINTS
        self._check_bounds(value, min, max, "Sweep points")
        self._handle.write(f":sense:sweep:point {value}")

    @property
    def powers(self) -> tuple[float, float]:
        """ """
        port1_power = float(self._handle.query(":source:power:port1?"))
        port2_power = float(self._handle.query(":source:power:port2?"))
        return (port1_power, port2_power)

    @powers.setter
    def powers(self, value: tuple[float, float]) -> None:
        try:
            port1_power, port2_power = value
        except (TypeError, ValueError):
            raise ValueError("Powers setter expects (float, float)") from None
        else:
            self._handle.write(f":source:power:port1 {port1_power}")
            self._handle.write(f":source:power:port2 {port2_power}")

    @property
    def traces(self) -> list[tuple[str, str]]:
        """ """
        return self._traces.copy()

    @traces.setter
    def traces(self, value: list[tuple[str, str]]) -> None:
        """ """
        try:
            num_traces = len(value)
        except (TypeError, ValueError):
            raise ValueError("Traces setter expects list[tuple[str, str]]") from None
        else:
            min, max = MS46522B.MIN_TRACES, MS46522B.MAX_TRACES
            self._check_bounds(num_traces, min, max, "Number of traces")
            self._handle.write(f":calculate:parameter:count {num_traces}")

            valid_s_params = MS46522B.VALID_S_PARAMETERS
            valid_trace_fmts = MS46522B.VALID_TRACE_FORMATS
            for count, trace_info in enumerate(value, start=1):
                s_param, trace_fmt = trace_info
                if s_param.lower() not in valid_s_params:
                    raise ValueError(f"Invalid '{s_param = }', {valid_s_params = }.")
                if trace_fmt.lower() not in valid_trace_fmts:
                    raise ValueError(f"Invalid {trace_fmt = }, {valid_trace_fmts = }.")
                self._handle.write(f":calculate:parameter{count}:define {s_param}")
                self._handle.write(f":calculate:parameter{count}:format {trace_fmt}")

            self._traces = value

            # adjust trace display on shockline app
            layout = MS46522B.DEFAULT_TRACE_LAYOUT[num_traces]
            self._handle.write(f":display:window:split {layout}")

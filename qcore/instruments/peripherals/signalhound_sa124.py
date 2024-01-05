""" """

from ctypes import CDLL, byref, c_char_p, c_double, c_int
from pathlib import Path

import numpy as np

from qcore.helpers.logger import logger
from qcore.instruments.instrument import Instrument, ConnectionError

SA = CDLL(str(Path(__file__).parent / "signalhound_sa124.dll"))

SA.saGetErrorString.restype = c_char_p
SA.saGetSweep_64f.argtypes = [
    c_int,
    np.ctypeslib.ndpointer(np.float64, ndim=1, flags="C"),
    np.ctypeslib.ndpointer(np.float64, ndim=1, flags="C"),
]


class SA124(Instrument):
    """Python driver to use SA124 USB spectrum analyzer in frequency sweep mode.

    Sweep parameters:
    1. "center": frequency sweep center in Hz
    2. "span": frequency sweep span in Hz
    3. "rbw": resolution bandwidth in Hz. The amplitude value for each frequency bin
    represents total energy from rbw / 2 below and above the bin's center. Available
    values are [0.1Hz-100kHz], 250kHz, 6MHz. See `_is_valid_rbw()` for exceptions to
    available values.
    4. "power": reference power level of the device in dBm. To achieve the best
    results, ensure gain and attenuation are set to AUTO and reference level is set
    at or slightly above expected input power for best sensitivity."""

    DETECTOR: int = 1
    """ Specify if overlapping results from signal processing should be averaged (1)
    or if minimum and maximum values should be maintained (0). """

    SCALE: int = 0
    """ Change units of returned amplitudes.
    0: log scale (dBm)
    1: lin scale (mV)
    2: log full scale input
    3: lin full scale input """

    RBW_SHAPE: int = 1
    """ Specify the RBW filter shape applied by changing the window function.
    1: custom bandwidth flat-top window measured at the 3dB cutoff point is used.
    2: Gaussian window with zero-padding measured at the 6dB cutoff point is used. """

    VID_PROC_UNITS: int = 0
    """ Specify units for video processing.
    0: log units, use this emulate a traditional spectrum analyzer.
    1: volt units , use this for cleaning up an amplitude modulated signal.
    2: power units, use this for "average" power measurements.
    3: bypass, use this to minimize processing power and bypass video processing. """

    REJECT_IMAGE: int = 1
    """ Determine whether software image reject will be performed. Generally, set
    reject to true (1) for continuous signals, and false (0) to catch short duration
    signals at a known frequency. """

    TIMEBASE: int = 2
    """ Set to (2) to use an external 10 MHz reference or (1) to use an internal
    clock reference. """

    MAX_CENTER: float = 13e9  # Hz
    MIN_CENTER: float = 100e3  # Hz
    MIN_SPAN: float = 1.0  # Hz
    MAX_POWER: float = 20.0  # dBm
    DEFAULT_RBW: float = 250e3  # Hz

    # device modes
    IDLE: int = -1
    SWEEPING: int = 0

    def __init__(
        self,
        name: str,
        id: str,
        center: float = 5e9,
        span: float = 500e6,
        rbw: float = DEFAULT_RBW,
        power: float = 0.0,
    ) -> None:
        """ """
        self._handle = None
        self._status = False  # set by connect() and _errorcheck()
        self._is_sweep_configured: bool = False  # to set sweep parameters on device
        self._freqs: list[float] = None  # to save sweep frequencies for quick access

        # these sweep parameters are set by the user to configure sweeps
        self._center: float = center
        self._span: float = span
        self._rbw: float = rbw
        self._power: float = power

        # these parameters are queried from the device after a sweep has been requested
        # they are set to None when the center, span, or rbw changes prior to a sweep
        self._sweep_length: int = None
        self._start_frequency: float = None
        self._bin_size: float = None

        super().__init__(id, name=name, center=center, span=span, rbw=rbw, power=power)

        # initialize SA124 with presets
        SA.saSetTimebase(self._handle, SA124.TIMEBASE)  # use external 10MHz reference
        SA.saConfigAcquisition(self._handle, SA124.DETECTOR, SA124.SCALE)
        SA.saConfigRBWShape(self._handle, SA124.RBW_SHAPE)
        SA.saConfigProcUnits(self._handle, SA124.VID_PROC_UNITS)

    def _errorcheck(self, errorcode: int) -> None:
        """ """
        if errorcode:  # non-zero values indicate errors
            details = SA.saGetErrorString(errorcode).decode()
            message = f"{self} got {errorcode = }, {details = }."
            if errorcode < 0:
                self._status = False
                raise ConnectionError(message)
            else:
                logger.warning(message)

    def connect(self) -> None:
        """ """
        if self.status:  # close any existing connections
            self.disconnect()

        device = c_int(-1)
        self._errorcheck(SA.saOpenDeviceBySerialNumber(byref(device), int(self.id)))
        self._handle = device.value
        self._status = True
        self._configure_sweep()  # to ensure device is ready to sweep upon connection

    def disconnect(self) -> None:
        """ """
        self._errorcheck(SA.saCloseDevice(self._handle))
        self._status = False

    @property
    def status(self) -> bool:
        """ """
        return self._status

    def sweep(self) -> tuple[list[float], list[float]]:
        """ """
        if not self._is_sweep_configured:
            self._configure_sweep()  # updates self._freqs
        sweep_min = np.zeros(len(self._freqs)).astype(np.float64)
        sweep_max = np.zeros(len(self._freqs)).astype(np.float64)
        self._errorcheck(SA.saGetSweep_64f(self._handle, sweep_min, sweep_max))
        # as SA124.DETECTOR = 1, returning sweep_max is okay
        return self._freqs, sweep_max.tolist()

    def single_sweep(
        self, center: float = None, averages: int = 1, configure: bool = False
    ) -> float:
        """ """
        if configure:
            center = self.center if center is None else center
            self.configure(center=center, rbw=250e3, span=250e3)

        ys = []
        for _ in range(averages):
            _, y = self.sweep()
            ys.append(y[len(y) // 2])

        ys = 10 ** (np.array(ys) / 10.0)
        y = np.mean(ys)
        return 10 * np.log10(y)

    def _configure_sweep(self) -> None:
        """ """
        handle, reject_image = self._handle, SA124.REJECT_IMAGE
        center, span = c_double(self.center), c_double(self.span)
        rbw, power = c_double(self.rbw), c_double(self.power)
        # SA124 must be set on idle mode before configuring sweep settings
        self._errorcheck(SA.saInitiate(handle, SA124.IDLE, 0))
        self._errorcheck(SA.saConfigCenterSpan(handle, center, span))
        self._errorcheck(SA.saConfigSweepCoupling(handle, rbw, rbw, reject_image))
        self._errorcheck(SA.saConfigLevel(handle, power))
        # set SA124 to sweep mode, we are now ready to sweep
        self._errorcheck(SA.saInitiate(self._handle, SA124.SWEEPING, 0))

        # get sweep information
        sweep_length, start_frequency, bin_size = c_int(-1), c_double(-1), c_double(-1)
        args = (byref(arg) for arg in (sweep_length, start_frequency, bin_size))
        self._errorcheck(SA.saQuerySweepInfo(handle, *args))
        self._sweep_length = sweep_length = sweep_length.value
        self._start_frequency = start_frequency = start_frequency.value
        self._bin_size = bin_size = bin_size.value
        self._freqs = [start_frequency + i * bin_size for i in range(sweep_length)]
        self._is_sweep_configured = True

    @property
    def sweep_length(self) -> int:
        """ """
        return self._sweep_length

    @property
    def start_frequency(self) -> float:
        """ """
        return self._start_frequency

    @property
    def bin_size(self) -> float:
        """ """
        return self._bin_size

    @property
    def center(self) -> float:
        """ """
        return self._center

    @center.setter
    def center(self, value: float) -> None:
        """ """
        try:
            in_bounds = SA124.MIN_CENTER <= value <= SA124.MAX_CENTER
        except TypeError:
            message = f"Expect center of {float}, got {value = } of {type(value)}"
            raise ValueError(message)
        else:
            if not in_bounds:
                bounds = f"[{SA124.MIN_CENTER:.2E}, {SA124.MAX_CENTER:.2E}]"
                raise ValueError(f"Center {value = :E} out of {bounds = }")
            self._center = value
            self._is_sweep_configured = False
            self._start_frequency, self._sweep_length, self._bin_size = None, None, None

    @property
    def span(self) -> float:
        """ """
        return self._span

    @span.setter
    def span(self, value: float) -> None:
        """ """
        max_freq, min_freq = self._center + value, self._center - value
        try:
            out_of_bounds = max_freq > SA124.MAX_CENTER or min_freq < SA124.MIN_CENTER
        except TypeError:
            message = f"Expect span of {float}, got {value = } of {type(value)}"
            raise ValueError(message)
        else:
            if out_of_bounds:
                bounds = f"[{SA124.MIN_CENTER:.2E}, {SA124.MAX_CENTER:.2E}]"
                raise ValueError(f"Sweep range out of {bounds = } for span = {value}.")
            self._span = value if value >= SA124.MIN_SPAN else SA124.MIN_SPAN
            self._is_sweep_configured = False
            self._start_frequency, self._sweep_length, self._bin_size = None, None, None

    @property
    def rbw(self) -> float:
        """ """
        return self._rbw

    @rbw.setter
    def rbw(self, value: float) -> None:
        """ """
        start, span = self._center - (self._span / 2), self._span
        is_valid = False  # conditions are obtained from the SA124 API manual
        try:
            if (value == 6e6 and start >= 2e8 and span >= 2e8) or value == 250e3:
                is_valid = True
            elif 0.1 <= value <= 1e5:
                is_valid = True
                if (span >= 1e8 or (span > 2e5 and start < 16e6)) and value < 6.5e3:
                    is_valid = False
        except TypeError:
            message = f"Expect rbw of {float}, got {value = } of {type(value)}"
            raise ValueError(message)
        else:
            if not is_valid:
                message = f"Invalid rbw {value = }, set default {SA124.DEFAULT_RBW}Hz."
                logger.warning(message)
                self._rbw = SA124.DEFAULT_RBW
            else:
                self._rbw = value
            self._is_sweep_configured = False
            self._start_frequency, self._sweep_length, self._bin_size = None, None, None

    @property
    def power(self) -> float:
        """ """
        return self._power

    @power.setter
    def power(self, value: float) -> None:
        """ """
        try:
            self._power = value if value <= SA124.MAX_POWER else SA124.MAX_POWER
        except TypeError:
            message = f"Expect power of {float}, got {value = } of {type(value)}"
            raise ValueError(message)
        else:
            self._is_sweep_configured = False

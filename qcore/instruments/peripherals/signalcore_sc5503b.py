""" """

from ctypes import (
    CDLL,
    Structure,
    POINTER,
    byref,
    c_float,
    c_ubyte,
    c_ulonglong,
    c_void_p,
    c_char_p,
    c_uint8,
)
from pathlib import Path

from qcore.instruments.instrument import Instrument


class RFParams(Structure):
    _fields_ = [("frequency", c_ulonglong), ("powerLevel", c_float),] + [
        (name, c_ubyte)
        for name in (
            "rfEnable",
            "alcOpen",
            "autoLevelEnable",
            "fastTune",
            "tuneStep",
            "referenceSetting",
        )
    ]


class DeviceStatus(Structure):
    _fields_ = [
        (name, c_ubyte)
        for name in (
            "tcxoPllLock",
            "vcxoPllLock",
            "finePllLock",
            "coarsePllLock",
            "sumPllLock",
            "extRefDetected",
            "refClkOutEnable",
            "extRefLockEnable",
            "alcOpen",
            "fastTuneEnable",
            "standbyEnable",
            "rfEnable",
            "pxiClkEnable",
        )
    ]


SC = CDLL(str(Path(__file__).parent / "signalcore_sc5503b.dll"))

SC.sc5503b_OpenDevice.argtypes = [c_char_p, POINTER(c_void_p)]
SC.sc5503b_CloseDevice.argtypes = [c_void_p]
SC.sc5503b_SetFrequency.argtypes = [c_void_p, c_ulonglong]
SC.sc5503b_SetPowerLevel.argtypes = [c_void_p, c_float]
SC.sc5503b_SetRfOutput.argtypes = [c_void_p, c_ubyte]
SC.sc5503b_GetRfParameters.argtypes = [c_void_p, POINTER(RFParams)]
SC.sc5503b_GetDeviceStatus.argtypes = [c_void_p, POINTER(DeviceStatus)]
SC.sc5503b_SetClockReference.argtypes = [c_void_p, c_uint8, c_uint8, c_uint8, c_uint8]


class SC5503B(Instrument):
    """ """

    def __init__(
        self,
        name: str,
        id: str,
        frequency: float = 6e9,
        power: float = 0.0,
        output: bool = False,
    ):
        """ """
        self._handle = None
        super().__init__(id, name=name, frequency=frequency, power=power, output=output)

        SC.sc5503b_SetClockReference(self._handle, 1, 0, 0, 0)

    def connect(self):
        """ """
        if self.status or self._handle is not None:
            self.disconnect()

        self._handle = c_void_p()
        SC.sc5503b_OpenDevice(self.id.encode(), byref(self._handle))

    def disconnect(self):
        """ """
        SC.sc5503b_CloseDevice(self._handle)
        self._handle = None

    @property
    def status(self) -> bool:
        """ """
        try:
            return bool(self.frequency)  # frequency == 0.0 when SC5503B is disconnected
        except OSError:
            self._handle = None
            return False

    def _get_rf_params(self) -> RFParams:
        """ """
        rf_params = RFParams()
        SC.sc5503b_GetRfParameters(self._handle, rf_params)
        return rf_params

    def _get_status(self) -> DeviceStatus:
        status = DeviceStatus()
        SC.sc5503b_GetDeviceStatus(self._handle, status)
        return status

    @property
    def clocked(self) -> bool:
        """ """
        return bool(self._get_status().extRefDetected)

    @property
    def output(self) -> bool:
        """ """
        return bool(self._get_status().rfEnable)

    @output.setter
    def output(self, value: bool) -> None:
        """ """
        SC.sc5503b_SetRfOutput(self._handle, int(bool(value)))

    @property
    def frequency(self) -> float:
        """ """
        return float(self._get_rf_params().frequency)

    @frequency.setter
    def frequency(self, value: float) -> None:
        """ """
        SC.sc5503b_SetFrequency(self._handle, int(value))

    @property
    def power(self) -> float:
        """ """
        return self._get_rf_params().powerLevel

    @power.setter
    def power(self, value: float) -> None:
        """ """
        SC.sc5503b_SetPowerLevel(self._handle, value)

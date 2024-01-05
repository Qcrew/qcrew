""" """

from ctypes import (
    c_char_p,
    c_void_p,
    c_ulonglong,
    c_float,
    CDLL,
    c_ubyte,
    POINTER,
    Structure,
    c_uint,
    c_ushort,
)
from pathlib import Path

from qcore.instruments.instrument import Instrument


class RFParams(Structure):
    _fields_ = [
        ("rf1_freq", c_ulonglong),
        ("start_freq", c_ulonglong),
        ("stop_freq", c_ulonglong),
        ("step_freq", c_ulonglong),
        ("sweep_dwell_time", c_uint),
        ("sweep_cycles", c_uint),
        ("buffer_points", c_uint),
        ("rf_level", c_float),
        ("rf2_freq", c_ushort),
    ]


class DeviceStatus(Structure):
    class ListMode(Structure):
        _fields_ = [
            (name, c_ubyte)
            for name in (
                "sss_mode",
                "sweep_dir",
                "tri_waveform",
                "hw_trigger",
                "step_on_hw_trig",
                "return_to_start",
                "trig_out_enable",
                "trig_out_on_cycle",
            )
        ]

    class PLLStatus(Structure):
        _fields_ = [
            (name, c_ubyte)
            for name in (
                "sum_pll_ld",
                "crs_pll_ld",
                "fine_pll_ld",
                "crs_ref_pll_ld",
                "crs_aux_pll_ld",
                "ref_100_pll_ld",
                "ref_10_pll_ld",
                "rf2_pll_ld",
            )
        ]

    class OperateStatus(Structure):
        _fields_ = [
            (name, c_ubyte)
            for name in (
                "rf1_lock_mode",
                "rf1_loop_gain",
                "device_access",
                "rf2_standby",
                "rf1_standby",
                "auto_pwr_disable",
                "alc_mode",
                "rf1_out_enable",
                "ext_ref_lock_enable",
                "ext_ref_detect",
                "ref_out_select",
                "list_mode_running",
                "rf1_mode",
                "over_temp",
                "harmonic_ss",
            )
        ]

    _fields_ = [
        ("list_mode", ListMode),
        ("operate_status", OperateStatus),
        ("pll_status", PLLStatus),
    ]


SC = CDLL(str(Path(__file__).parent / "signalcore_sc5511a.dll"))

SC.sc5511a_open_device.argtypes = [c_char_p]
SC.sc5511a_open_device.restype = c_void_p
SC.sc5511a_close_device.argtypes = [c_void_p]
SC.sc5511a_set_freq.argtypes = [c_void_p, c_ulonglong]
SC.sc5511a_set_level.argtypes = [c_void_p, c_float]
SC.sc5511a_set_output.argtypes = [c_void_p, c_ubyte]
SC.sc5511a_set_clock_reference.argtypes = [c_void_p, c_ubyte, c_ubyte]
SC.sc5511a_get_rf_parameters.argtypes = [c_void_p, POINTER(RFParams)]
SC.sc5511a_get_device_status.argtypes = [c_void_p, POINTER(DeviceStatus)]


class SC5511A(Instrument):
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

        # 0 = 10 MHz ref out signal, 1 = locks to external reference
        SC.sc5511a_set_clock_reference(self._handle, 0, 1)

    def connect(self):
        """ """
        if self.status or self._handle is not None:
            self.disconnect()
        self._handle = SC.sc5511a_open_device(self.id.encode())

    def disconnect(self):
        """ """
        SC.sc5511a_close_device(self._handle)
        self._handle = None

    @property
    def status(self) -> bool:
        try:
            return bool(self._get_status().operate_status.device_access)
        except OSError:
            self._handle = None
            return False

    def _get_status(self) -> DeviceStatus:
        """ """
        status = DeviceStatus()
        SC.sc5511a_get_device_status(self._handle, status)
        return status

    def _get_rf_params(self) -> RFParams:
        """ """
        rf_params = RFParams()
        SC.sc5511a_get_rf_parameters(self._handle, rf_params)
        return rf_params

    @property
    def clocked(self) -> bool:
        """ """
        return bool(self._get_status().operate_status.ext_ref_detect)

    @property
    def output(self) -> bool:
        """ """
        return bool(self._get_status().operate_status.rf1_out_enable)

    @output.setter
    def output(self, value: bool) -> None:
        """ """
        SC.sc5511a_set_output(self._handle, int(bool(value)))

    @property
    def frequency(self) -> float:
        """ """
        return float(self._get_rf_params().rf1_freq)

    @frequency.setter
    def frequency(self, value: float) -> None:
        """ """
        SC.sc5511a_set_freq(self._handle, int(value))

    @property
    def power(self) -> float:
        """ """
        return self._get_rf_params().rf_level

    @power.setter
    def power(self, value: float) -> None:
        """ """
        SC.sc5511a_set_level(self._handle, value)

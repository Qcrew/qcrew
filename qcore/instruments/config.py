""" QCREW's instrument config, a catalogue of all programmable instruments in our lab and their types and serial numbers """

from qcore.instruments.peripherals import *
from qcore.instruments.instrument import DummyInstrument


class InstrumentConfig(dict):
    """ """

    def __init__(self) -> None:
        """
        key: Instrument class
        value: list of ids corresponding to the instruments qcrew has of the given class, each id must be a string
        """
        self[MS46522B] = ["VNA1"]
        self[SC5503B] = ["10002656"]
        self[SC5511A] = ["10002657"]
        self[SA124] = ["19184645", "20234154"]
        self[LMS] = [str(id) for id in range(25330, 25338)]
        self[GS200] = ["90X823743", "91X336839"]
        self[DummyInstrument] = ["0", "1", "2", "3", "4", "5", "6", "7"]

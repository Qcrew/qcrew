import os
from qm.octave import QmOctaveConfig

from qcore.instruments.instrument import Instrument
from qcore.variables.parameter import Parameter


class DummyOctave(Instrument):

    settings: dict = Parameter()

    def __init__(self, settings: dict, id: str, **parameters):
        self._settings = settings
        super().__init__(id, **parameters)

    @property
    def status(self) -> bool:
        return True

    @settings.getter
    def settings(self) -> dict:
        """ """
        return self._settings

    @settings.setter
    def settings(self, value: dict) -> None:
        """ """
        self._settings = value

    def connect(self) -> None:
        # # Note: port 80 should be replaced if Octave is on external network
        # #  i.e., replaced with 11XXX where XXX are the last 3 digits of the
        # #  octave IP address behind the router. e.g. if the octave is
        # #  XXX.XXX.XX.50, the port would be 11050.
        # self._octave = OctaveUnit(self.name, self.id, port=80, con="con1")
        # # Add the octaves
        # octaves = [self._octave]
        # # Configure the Octaves
        # self._config = octave_declaration(octaves)
        pass


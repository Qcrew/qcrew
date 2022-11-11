from qm.qua import fixed, program, stream_processing

from qcore.elements import Readout
from qcore.experiment import Experiment
from qcore.expvariable import ExpVar
from qcore.sweep import Sweep
from qcore.sequences.rr_spec import generate_rr_spec
from qcore.sequences.constructors import construct_sweep
from qcore.instruments import QM
from qcore.dataset import Dataset
import numpy as np


class RR_Spec(Experiment):
    def __init__(self, rr: Readout, **kwargs):
        self.rr = rr

        super().__init__(**kwargs)

    def init_variables(self):

        # set control variables
        self.wait_time = 20000  # ns

        # set sweep variables
        self.N = Sweep(
            name="N",
            var_type=int,
            start=0,
            stop=50000,
            step=1,
        )

        self.freq = Sweep(
            start=int(-60e6),
            stop=int(-40e6),
            step=int(0.5e6),
            var_type=int,
            include_endpoint=True,
            name="freq",
            units="Hz",
        )

        # define datasets
        self.I = Dataset(axes=[self.N, self.freq], name="I", var_type=fixed)
        self.Q = Dataset(axes=[self.N, self.freq], name="Q", var_type=fixed)

        # for derived datasets, include initial data for calculating running total
        init_data = np.zeros(self.freq.length)
        self.iq_avg = Dataset(axes=[self.freq], name="IQ_AVG", data=init_data)
        self.magnitude = Dataset(
            axes=[self.freq], name="MAGNITUDE", units="dB", data=init_data
        )
        self.phase = Dataset(
            axes=[self.freq], name="PHASE", units="rad", data=init_data
        )

        self.arg_mapping = {"freq": self.freq.q_var}

    def process_streams(self):

        self.I.process_stream()
        self.Q.process_stream()
        self.freq.process_stream(save_all=False)
        self.N.process_stream(save_all=False)

    def construct_pulse_sequence(self):

        with program() as qua_program:
            self.init_variables()

            rr_spec = generate_rr_spec(self.I, self.Q, self.rr)
            ordered_sweep_list = [self.N, self.freq]

            construct_sweep(
                ordered_sweep_list=ordered_sweep_list,
                pulse_sequence=rr_spec,
                arg_mapping=self.arg_mapping,
                wait_time=self.wait_time,
                wait_elem=self.rr,
            )

            with stream_processing:
                self.process_streams()

        # add with_stream context here
        return qua_program

    def process_data(self, datasaver, data, current_count, last_count):
        """this is INSIDE the fetch loop!!! use for live processing only!!!"""
        # while saving, assume sweep order and dimension and dataset axes order and dimension is consistent

        # save frequency data
        datasaver.save_data(self.freq, data["freq"])

        # save raw I and Q data at the correct position
        pos = slice(last_count, current_count)
        datasaver.save_data(self.I, data["I"], index=(pos, ...))
        datasaver.save_data(self.Q, data["Q"], index=(pos, ...))

        # calculate and store derived datasets for plotting
        num_batches = current_count - last_count  # for weighted averages
        i_avg, q_avg = np.average(data["I"], axis=0), np.average(data["Q"], axis=0)
        iq_avg = i_avg + 1j * q_avg
        self.iq_avg.data = (
            last_count * self.iq_avg.data + num_batches * iq_avg
        ) / current_count

        phase = np.unwrap(np.angle(data["I"] + 1j * data["Q"]))
        phase_avg = np.average(phase, axis=0)
        self.phase.data = (
            last_count * self.phase.data + num_batches * phase_avg
        ) / current_count

        magnitude = 20 * np.log10(np.abs(data["I"] + 1j * data["Q"]))
        magnitude_avg = np.average(magnitude, axis=0)
        self.magnitude.data = (
            last_count * self.magnitude.data + num_batches * magnitude_avg
        ) / current_count

        # live plot datasets
        self.plotter.plot()
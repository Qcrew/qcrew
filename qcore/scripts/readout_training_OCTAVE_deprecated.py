""" """

import matplotlib.pyplot as plt
import numpy as np
from qm import qua as qm_qua
from scipy import signal, special
from scipy.optimize import curve_fit
from qm.qua import AnalogMeasureProcess, fixed
from qcore import qua
from qcore.helpers.logger import logger
from qcore.modes.mode import Mode
from qcore.libs.fit_fns import gaussian2d_symmetric

ADC_TO_VOLTS = 2**-12
TS = 1e-9  # Sampling time of the OPX in seconds
T_OFS = 36.46


# class AccumulationMethod:
#     def __init__(self):
#         self.loc = ""
#         self.return_func: Type[AnalogMeasureProcess] = None

#     def _full_target(
#         self, target: QuaVariableType
#     ) -> AnalogMeasureProcess.ScalarProcessTarget:
#         return AnalogMeasureProcess.ScalarProcessTarget(self.loc, target)

#     def _sliced_target(
#         self, target: QuaVariableType, samples_per_chunk: int
#     ) -> AnalogMeasureProcess.VectorProcessTarget:
#         analog_time_division = AnalogMeasureProcess.SlicedAnalogTimeDivision(
#             self.loc, samples_per_chunk
#         )
#         return AnalogMeasureProcess.VectorProcessTarget(
#             self.loc, target, analog_time_division
#         )

#     def _accumulated_target(
#         self, target: QuaVariableType, samples_per_chunk: int
#     ) -> AnalogMeasureProcess.VectorProcessTarget:
#         analog_time_division = AnalogMeasureProcess.AccumulatedAnalogTimeDivision(
#             self.loc, samples_per_chunk
#         )
#         return AnalogMeasureProcess.VectorProcessTarget(
#             self.loc, target, analog_time_division
#         )

#     def _moving_window_target(
#         self, target: QuaVariableType, samples_per_chunk: int, chunks_per_window: int
#     ) -> AnalogMeasureProcess.VectorProcessTarget:
#         analog_time_division = AnalogMeasureProcess.MovingWindowAnalogTimeDivision(
#             self.loc, samples_per_chunk, chunks_per_window
#         )
#         return AnalogMeasureProcess.VectorProcessTarget(
#             self.loc, target, analog_time_division
#         )


# class RealAccumulationMethod(AccumulationMethod):
#     """A base class for specifying the integration and demodulation processes in the [measure][qm.qua._dsl.measure]
#     statement.
#     These are the options which can be used inside the measure command as part of the ``demod`` and ``integration``
#     processes.
#     """

#     def __init__(self):
#         super().__init__()

#     def __new__(cls):
#         if cls is AccumulationMethod:
#             raise TypeError("base class may not be instantiated")
#         return object.__new__(cls)

#     def full(self, iw: str, target: QuaVariableType, element_output: str = ""):
#         """Perform an ordinary demodulation/integration. See [Full demodulation](../../../Guides/features/#full-demodulation).

#         Args:
#             iw (str): integration weights
#             target (QUA variable): variable to which demod result is
#                 saved
#             element_output: (optional) the output of an element from
#                 which to get ADC results
#         """
#         return self.return_func(self.loc, element_output, iw, self._full_target(target))

#     def sliced(
#         self,
#         iw: str,
#         target: QuaVariableType,
#         samples_per_chunk: int,
#         element_output: str = "",
#     ):
#         """Perform a demodulation/integration in which the demodulation/integration process is split into chunks
#         and the value of each chunk is saved in an array cell. See [Sliced demodulation](../../../Guides/features/#sliced-demodulation).

#         Args:
#             iw (str): integration weights
#             target (QUA array): variable to which demod result is saved
#             samples_per_chunk (int): The number of ADC samples to be
#                 used for each chunk is this number times 4.
#             element_output: (optional) the output of an element from
#                 which to get ADC results
#         """
#         return self.return_func(
#             self.loc, element_output, iw, self._sliced_target(target, samples_per_chunk)
#         )

#     def accumulated(
#         self,
#         iw: str,
#         target: QuaVariableType,
#         samples_per_chunk: int,
#         element_output: str = "",
#     ):
#         """Same as ``sliced()``, however the accumulated result of the demodulation/integration
#         is saved in each array cell. See [Accumulated demodulation](../../../Guides/features/#accumulated-demodulation).

#         Args:
#             iw (str): integration weights
#             target (QUA array): variable to which demod result is saved
#             samples_per_chunk (int): The number of ADC samples to be
#                 used for each chunk is this number times 4.
#             element_output: (optional) the output of an element from
#                 which to get ADC results
#         """
#         return self.return_func(
#             self.loc,
#             element_output,
#             iw,
#             self._accumulated_target(target, samples_per_chunk),
#         )

#     def moving_window(
#         self,
#         iw: str,
#         target: QuaVariableType,
#         samples_per_chunk: int,
#         chunks_per_window: int,
#         element_output: str = "",
#     ):
#         """Same as ``sliced()``, however the several chunks are accumulated and saved to each array cell.
#         See [Moving window demodulation](../../../Guides/features/#moving-window-demodulation).

#         Args:
#             iw (str): integration weights
#             target (QUA array): variable to which demod result is saved
#             samples_per_chunk (int): The number of ADC samples to be
#                 used for each chunk is this number times 4.
#             chunks_per_window (int): The number of chunks to use in the
#                 moving window
#             element_output: (optional) the output of an element from
#                 which to get ADC results
#         """
#         return self.return_func(
#             self.loc,
#             element_output,
#             iw,
#             self._moving_window_target(target, samples_per_chunk, chunks_per_window),
#         )


# class _Demod(RealAccumulationMethod):
#     def __init__(self):
#         super().__init__()
#         self.loc = ""
#         self.return_func = AnalogMeasureProcess.DemodIntegration


# demod = _Demod()


class ReadoutTrainerOctave:
    """ """

    def __init__(
        self,
        rr: Mode,
        qubit: Mode,
        qm,
        reps,
        wait_time,
        readout_pulse,
        qubit_pi_pulse,
        ddrop_params=None,
        weights_file_path=None,
    ):
        """ """
        self._rr: Mode = rr
        self._qubit: Mode = qubit
        self._qm = qm
        self.modes = [rr, qubit]
        self.mode_names = [mode.name for mode in self.modes]
        self.reps = reps
        self.wait_time = wait_time
        self.readout_pulse = readout_pulse
        self.qubit_pi_pulse = qubit_pi_pulse
        self.ddrop_params = ddrop_params
        self.weights_file_path = weights_file_path

        logger.info(f"Initialized ReadoutTrainer with {self._rr} and {self._qubit}")

    def train_weights(self) -> None:
        """
        Obtain integration weights of rr given the excited and ground states of qubit and update rr mode.
        """
        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)

        # Start with constant integration weights. Not really necessary
        self._reset_weights()

        division_length = 10  # Size of each demodulation slice in clock cycles

        # Get traces and average envelope when qubit in ground state
        env_g, env_e = self._acquire_traces(self._qm, division_length)

        # Get difference between average envelopes
        subtracted_trace = env_e - env_g

        norm_subtracted_trace = normalize_complex_array(
            subtracted_trace
        )  # <- these are the optimal weights :)

        # Update readout with optimal weights
        # weights = self._update_weights(squeezed_diff)

        # Plot envelopes
        # Time axis for the plots at the end

        x_plot = np.arange(
            division_length * 4,
            readout_pulse.length + readout_pulse.pad + 1,
            division_length * 4,
        )

        plot_three_complex_arrays(x_plot, env_g, env_e, norm_subtracted_trace)

        return env_g, env_e

    def normalize_complex_array(arr):
        # Calculate the simple norm of the complex array
        norm = np.sqrt(np.sum(np.abs(arr) ** 2))

        # Normalize the complex array by dividing it by the norm
        normalized_arr = arr / norm

        # Rescale the normalized array so that the maximum value is 1
        max_val = np.max(np.abs(normalized_arr))
        rescaled_arr = normalized_arr / max_val

        return rescaled_arr

    def plot_three_complex_arrays(x, arr1, arr2, arr3):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.plot(x, arr1.real, label="real")
        ax1.plot(x, arr1.imag, label="imag")
        ax1.set_title("ground state")
        ax1.set_xlabel("Readout time [ns]")
        ax1.set_ylabel("demod traces [a.u.]")
        ax1.legend()
        ax2.plot(x, arr2.real, label="real")
        ax2.plot(x, arr2.imag, label="imag")
        ax2.set_title("excited state")
        ax2.set_xlabel("Readout time [ns]")
        ax2.set_ylabel("demod traces [a.u.]")
        ax2.legend()
        ax3.plot(x, arr3.real, label="real")
        ax3.plot(x, arr3.imag, label="imag")
        ax3.set_title("SNR")
        ax3.set_xlabel("Readout time [ns]")
        ax3.set_ylabel("subtracted traces [a.u.]")
        ax3.legend()
        plt.tight_layout()
        plt.show()

    def divide_array_in_half(arr):
        split_index = len(arr) // 2
        arr1 = arr[:split_index]
        arr2 = arr[split_index:]
        return arr1, arr2

    def _reset_weights(self):
        """
        Start the pulse with constant integration weights
        """
        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)
        readout_pulse.weights = (1.0, 0.0, 0.0, 1.0)

    def _acquire_traces(self, qm, division_length) -> tuple[list]:
        """
        Run QUA program to obtain traces of the readout pulse.
        """

        # Execute script
        qua_program = self._get_QUA_trace_acquisition(division_length)
        job = self._qm.execute(qua_program)

        # Fetch and reshape the data
        res_handles = job.result_handles
        res_handles.wait_for_all_values()

        IIg, IIe = divide_array_in_half(res_handles.get("II").fetch_all())
        IQg, IQe = divide_array_in_half(res_handles.get("IQ").fetch_all())
        QIg, QIe = divide_array_in_half(res_handles.get("QI").fetch_all())
        QQg, QQe = divide_array_in_half(res_handles.get("QQ").fetch_all())
        # Sum the quadrature to fully demodulate the traces
        Ie = IIe + IQe
        Ig = IIg + IQg
        Qe = QIe + QQe
        Qg = QIg + QQg
        # Derive and normalize the ground and excited traces
        ground_trace = Ig + 1j * Qg
        excited_trace = Ie + 1j * Qe

        return ground_trace, excited_trace

    def _get_QUA_trace_acquisition(self, division_length):
        """ """
        reps = self.reps
        wait_time = self.wait_time
        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)
        (qubit_pi_pulse,) = self._qubit.get_operations(self.qubit_pi_pulse)

        with qm_qua.program() as acquire_traces:
            number_of_divisions = int(
                (readout_pulse.length + readout_pulse.pad) / 4 / division_length
            )

            ind = qm_qua.declare(int)
            n = qm_qua.declare(int)
            II = qm_qua.declare(fixed, size=number_of_divisions)
            IQ = qm_qua.declare(fixed, size=number_of_divisions)
            QI = qm_qua.declare(fixed, size=number_of_divisions)
            QQ = qm_qua.declare(fixed, size=number_of_divisions)
            n_st = qm_qua.declare_stream()
            II_st = qm_qua.declare_stream()
            IQ_st = qm_qua.declare_stream()
            QI_st = qm_qua.declare_stream()
            QQ_st = qm_qua.declare_stream()

            # division_length is the size of each demodulation slice in clock cycles

            with qm_qua.for_(n, 0, n < reps, n + 1):

                qm_qua.measure(
                    readout_pulse,
                    self._rr.name,
                    None,
                    AnalogMeasureProcess.DemodIntegration(
                        "", "out1", "cos", self._sliced_target(II, division_length)
                    ),
                    AnalogMeasureProcess.DemodIntegration(
                        "", "out2", "sin", self._sliced_target(IQ, division_length)
                    ),
                    AnalogMeasureProcess.DemodIntegration(
                        "",
                        "out1",
                        "minus_sin",
                        self._sliced_target(QI, division_length)
                    ),
                    AnalogMeasureProcess.DemodIntegration(
                        "", "out2", "cos", self._sliced_target(QQ, division_length)
                    ),
                )
                # demod.sliced("cos", II, division_length, "out1"),
                # demod.sliced("sin", IQ, division_length, "out2"),
                # demod.sliced("minus_sin", QI, division_length, "out1"),
                # demod.sliced("cos", QQ, division_length, "out2"),

                qua.wait(wait_time, self._rr)

                with qm_qua.for_(ind, 0, ind < number_of_divisions, ind + 1):
                    qm_qua.save(II[ind], II_st)
                    qm_qua.save(IQ[ind], IQ_st)
                    qm_qua.save(QI[ind], QI_st)
                    qm_qua.save(QQ[ind], QQ_st)

                qua.align()

                self._qubit.play(qubit_pi_pulse)
                qua.align(self._rr, self._qubit)
                qm_qua.measure(
                    readout_pulse,
                    self._rr.name,
                    None,
                    AnalogMeasureProcess.DemodIntegration(
                        "", "out1", "cos", self._sliced_target(II, division_length)
                    ),
                    AnalogMeasureProcess.DemodIntegration(
                        "", "out2", "sin", self._sliced_target(IQ, division_length)
                    ),
                    AnalogMeasureProcess.DemodIntegration(
                        "",
                        "out1",
                        "minus_sin",
                        self._sliced_target(QI, division_length),
                    ),
                    AnalogMeasureProcess.DemodIntegration(
                        "", "out2", "cos", self._sliced_target(QQ, division_length)
                    ),
                )
                # demod.sliced("cos", II, division_length, "out1"),
                # demod.sliced("sin", IQ, division_length, "out2"),
                # demod.sliced("minus_sin", QI, division_length, "out1"),
                # demod.sliced("cos", QQ, division_length, "out2"),

                qua.wait(wait_time, self._rr)

                with qm_qua.for_(ind, 0, ind < number_of_divisions, ind + 1):
                    qm_qua.save(II[ind], II_st)
                    qm_qua.save(IQ[ind], IQ_st)
                    qm_qua.save(QI[ind], QI_st)
                    qm_qua.save(QQ[ind], QQ_st)
                qm_qua.save(n, n_st)

            with qm_qua.stream_processing():
                n_st.save("iteration")
                II_st.buffer(2 * number_of_divisions).average().save("II")
                IQ_st.buffer(2 * number_of_divisions).average().save("IQ")
                QI_st.buffer(2 * number_of_divisions).average().save("QI")
                QQ_st.buffer(2 * number_of_divisions).average().save("QQ")

        return acquire_traces

    def _sliced_target(
        self, target, samples_per_chunk: int
    ) -> AnalogMeasureProcess.VectorProcessTarget:
        analog_time_division = AnalogMeasureProcess.SlicedAnalogTimeDivision(
            "", samples_per_chunk
        )
        return AnalogMeasureProcess.VectorProcessTarget(
            "", target, analog_time_division
        )

    def _calc_average_envelope(self, trace_list, timestamps, t_ofs):
        int_freq = np.abs(self._rr.int_freq)

        # demodulate
        s = trace_list * np.exp(1j * 2 * np.pi * int_freq * TS * (timestamps - t_ofs))

        # filter 2*omega_IF using hann filter
        hann = signal.hann(int(2 / TS / int_freq), sym=True)
        hann = hann / np.sum(hann)
        s_filtered = np.array([np.convolve(s_single, hann, "same") for s_single in s])

        # adjust envelope
        env = 2 * s_filtered.conj()

        # get average envelope
        avg_env = np.average(env, axis=0)

        return avg_env

    def _squeeze_array(self, s):
        """
        Split the array in bins of 4 values and average them. QM requires the weights to have 1/4th of the length of the readout pulse.
        """
        return np.average(np.reshape(s, (-1, 4)), axis=1)

    def _update_weights(self, squeezed_diff):
        weights = {}
        weights["I"] = np.array(
            [np.real(squeezed_diff).tolist(), (np.imag(-squeezed_diff)).tolist()]
        )
        weights["Q"] = np.array(
            [np.imag(-squeezed_diff).tolist(), np.real(-squeezed_diff).tolist()]
        )

        path = self.weights_file_path

        # Save weights to npz file
        np.savez(path, **weights)

        # Update the readout pulse with the npz file path
        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)
        readout_pulse.weights = str(path)

        return weights

    def _fit_hist_double_gaussian(self, guess, data_g):
        """
        The E population is estimated from the amplitudes of the two gaussians
        fitted from the G state blob.
        """

        p0 = [
            guess["x0"],
            guess["x1"],
            guess["a0"],
            guess["a1"],
            guess["ofs"],
            guess["sigma"],
        ]

        popt, _ = curve_fit(double_gaussian, data_g["xs"], data_g["ys"], p0=p0)
        return popt

    def calculate_threshold(self):
        # Get IQ for qubit in ground state
        IQ_acquisition_program = self._get_QUA_IQ_acquisition()
        job = self._qm.execute(IQ_acquisition_program)
        handle = job.result_handles
        handle.wait_for_all_values()
        Ig_list = handle.get("I").fetch_all()["value"]
        Qg_list = handle.get("Q").fetch_all()["value"]

        # Get IQ for qubit in excited state
        IQ_acquisition_program = self._get_QUA_IQ_acquisition(excite_qubit=True)
        job = self._qm.execute(IQ_acquisition_program)
        handle = job.result_handles
        handle.wait_for_all_values()
        Ie_list = handle.get("I").fetch_all()["value"]
        Qe_list = handle.get("Q").fetch_all()["value"]

        # Fit each blob to a 2D gaussian and retrieve the center
        params_g, data_g = self._fit_IQ_blob(Ig_list, Qg_list)
        params_e, data_e = self._fit_IQ_blob(Ie_list, Qe_list)

        IQ_center_g = (params_g["x0"], params_g["y0"])  # G blob center
        IQ_center_e = (params_e["x0"], params_e["y0"])  # E blob center

        # Calculate threshold
        threshold = (IQ_center_g[0] + IQ_center_e[0]) / 2

        # Update readout with optimal threshold
        self._update_threshold(threshold)

        # Calculates the confusion matrix of the readout
        conf_matrix = self._calculate_confusion_matrix(Ig_list, Ie_list, threshold)

        # Plot scatter and contour of each blob
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_aspect("equal")
        ax.scatter(Ig_list, Qg_list, label="|g>", s=5)
        ax.scatter(Ie_list, Qe_list, label="|e>", s=5)
        ax.contour(
            data_g["I_grid"],
            data_g["Q_grid"],
            data_g["counts_fit"],
            levels=5,
            cmap="winter",
        )
        ax.contour(
            data_e["I_grid"],
            data_e["Q_grid"],
            data_e["counts_fit"],
            levels=5,
            cmap="autumn",
        )
        ax.plot(
            [threshold, threshold],
            [np.min(data_g["Q_grid"]), np.max(data_g["Q_grid"])],
            label="threshold",
            c="k",
            linestyle="--",
        )

        ax.set_title("IQ blobs for each qubit state")
        ax.set_ylabel("Q")
        ax.set_xlabel("I")
        ax.legend()
        plt.show()

        # Plot I histogram
        fig, ax = plt.subplots(figsize=(7, 4))
        n_g, bins_g, _ = ax.hist(Ig_list, bins=50, alpha=1)
        n_e, bins_e, _ = ax.hist(Ie_list, bins=50, alpha=0.8)

        # Estimate excited state population from G blob double gaussian fit

        pge = conf_matrix["pge"]  # first estimate of excited state population
        guess = {
            "x0": params_g["x0"],
            "x1": params_e["x0"],
            "a0": max(n_g),
            "a1": max(n_g) * pge / (1 - pge),
            "ofs": 0.0,
            "sigma": params_g["sigma"],
        }
        data_hist_g = {"xs": (bins_g[1:] + bins_g[:-1]) / 2, "ys": n_g}

        popt = self._fit_hist_double_gaussian(guess, data_hist_g)
        print(popt)
        a0 = popt[2]
        a1 = popt[3]
        e_population = a1 / (a1 + a0)
        print("Excited state population: ", e_population)

        ax.plot(bins_g, [double_gaussian(x, *popt) for x in bins_g])

        ax.set_title("Projection of the IQ blobs onto the I axis")
        ax.set_ylabel("counts")
        ax.set_xlabel("I")
        ax.legend()
        plt.show()

        # Organize the raw I and Q data for each G and E measurement
        data = {
            "Ig": Ig_list,
            "Qg": Qg_list,
            "Ie": Ie_list,
            "Qe": Qe_list,
        }

        # Plot manual threshold selectiveness on |g>
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title("Is qubit really in ground state if state = 0?")
        ax.set_ylabel("Certainty")
        ax.set_xlabel("Threshold")
        popt
        p_pass_if_g = lambda t: 0.5 * special.erfc((t - popt[0]) / 2**0.5 / popt[-1])
        p_pass_if_e = lambda t: 0.5 * special.erfc((t - popt[1]) / 2**0.5 / popt[-1])
        p_g = popt[2] / (popt[2] + popt[3])
        p_e = popt[3] / (popt[2] + popt[3])
        certainty = (
            lambda t: p_pass_if_g(t)
            * p_g
            / (p_pass_if_e(t) * p_e + p_pass_if_g(t) * p_g)
        )
        t_rng = np.linspace(bins_g[0], bins_g[-1], 1000)
        ax.plot(t_rng, [certainty(t) for t in t_rng])
        ax.plot(
            [threshold, threshold],
            [0.4, 1.1],
            linestyle="--",
            label="calculated threshold",
        )
        ax.legend()
        plt.show()

        return threshold, data

    def _fit_IQ_blob(self, I_list, Q_list):
        fit_fn = "gaussian2d_symmetric"

        # Make ground IQ blob in a 2D histogram
        zs, xs, ys = np.histogram2d(I_list, Q_list, bins=50)

        # Replace "bin edge" by "bin center"
        dx = xs[1] - xs[0]
        xs = (xs - dx / 2)[1:]
        dy = ys[1] - ys[0]
        ys = (ys - dy / 2)[1:]

        # Get fit to 2D gaussian
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        best_fit, fit_params = gaussian2d_symmetric(zs, ys_grid.T, xs_grid.T)
        best_fit = best_fit.T

        data = {
            "I_grid": xs_grid,
            "Q_grid": ys_grid,
            "counts": zs,
            "counts_fit": best_fit,
        }

        return fit_params, data

    def _get_QUA_IQ_acquisition(self, excite_qubit: bool = False):
        """ """
        reps = self.reps
        wait_time = self.wait_time

        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)
        (qubit_pi_pulse,) = self._qubit.get_operations(self.qubit_pi_pulse)

        with qm_qua.program() as acquire_IQ:
            I = qm_qua.declare(qm_qua.fixed)
            Q = qm_qua.declare(qm_qua.fixed)
            n = qm_qua.declare(int)

            with qm_qua.for_(n, 0, n < reps, n + 1):
                # qua.play("predist_square_plusminus_pulse" * qua.amp(-0.3), "FLUX")
                # qua.wait(int(2500 // 4), self._qubit.name, self._rr.name)

                if self.ddrop_params:
                    self._macro_DDROP_reset()

                if excite_qubit:
                    # qua.align(self._rr.name, self._qubit.name)
                    self._qubit.play(qubit_pi_pulse)
                    qua.align(self._rr, self._qubit)

                self._rr.measure(readout_pulse, (I, Q), demod_type="dual")
                qm_qua.save(I, "I")
                qm_qua.save(Q, "Q")
                qua.wait(wait_time, self._rr)

        return acquire_IQ

    def _update_threshold(self, threshold):
        (readout_pulse,) = self._rr.get_operations(self.readout_pulse)
        readout_pulse.threshold = threshold

    def _calculate_confusion_matrix(self, Ig_list, Ie_list, threshold):
        pgg = 100 * round((np.sum(Ig_list > threshold) / len(Ig_list)), 3)
        pge = 100 * round((np.sum(Ig_list < threshold) / len(Ig_list)), 3)
        pee = 100 * round((np.sum(Ie_list < threshold) / len(Ie_list)), 3)
        peg = 100 * round((np.sum(Ie_list > threshold) / len(Ie_list)), 3)
        print("\nState prepared in |g>")
        print(f"   Measured in |g>: {pgg}%")
        print(f"   Measured in |e>: {pge}%")
        print("State prepared in |e>")
        print(f"   Measured in |e>: {pee}%")
        print(f"   Measured in |g>: {peg}%")
        return {"pgg": pgg, "pge": pge, "pee": pee, "peg": peg}

    def _macro_DDROP_reset(self):
        rr_ddrop_freq = self.ddrop_params["rr_ddrop_freq"]
        rr_ddrop = self.ddrop_params["rr_ddrop"]
        qubit_ddrop = self.ddrop_params["qubit_ddrop"]
        steady_state_wait = self.ddrop_params["steady_state_wait"]
        qubit_ef = self.ddrop_params["qubit_ef_mode"]

        qua.align(self._qubit, self._rr, qubit_ef)  # wait qubit pulse to end
        qua.update_frequency(self._rr, rr_ddrop_freq)
        qm_qua.play(self._rr.name, rr_ddrop)  # play rr ddrop excitation
        qua.wait(
            steady_state_wait, self._qubit, qubit_ef
        )  # wait resonator in steady state
        qm_qua.play(self._qubit.name, qubit_ddrop)  # play qubit ddrop excitation
        qm_qua.play(qubit_ef.name, "ddrop_pulse")  # play qubit ddrop excitation
        qua.wait(
            steady_state_wait, self._qubit, qubit_ef
        )  # wait resonator in steady state
        qua.align(self._qubit, self._rr, qubit_ef)  # wait qubit pulse to end
        qua.update_frequency(self._rr, self._rr.int_freq)


def double_gaussian(xs, x0, x1, a0, a1, ofs, sigma):
    """
    Gaussian defined by it's area <area>, sigma <s>, position <x0> and
    y-offset <ofs>.
    """
    r0 = (xs - x0) ** 2
    r1 = (xs - x1) ** 2
    ys = ofs + a0 * np.exp(-0.5 * r0 / sigma**2) + a1 * np.exp(-0.5 * r1 / sigma**2)
    return ys

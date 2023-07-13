""" """

import math
import time
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from qcore.instruments import QM, SA124, LMS
from qcore.modes.mode import Mode
from qcore.pulses import ConstantPulse
from qcore.helpers import logger, Stage

from qm import _Program
from qm.qua import infinite_loop_, program
from qm.QuantumMachine import QuantumMachine


class Parameter:
    """ """

    def __init__(self, name, value, vrange, minstep=0):
        self.name = name
        self.value = value
        self.vrange = vrange
        self.minstep = minstep


class Minimizer:
    """
    A brute-force parameter minimizer obtained from on Yale-qrlab implementation by Reiner Heeres, all credits to them.

    Straight-forward parameter optimizer.

    Optimization strategy:
    - Repeat <n_it> times (default: 5):
        - For each parameter:
            - sweep parameter from (value - range/2) to (value + range/2) in
              <n_eval> steps (default: 6) and evaluate function.
            - determine best parameter value.
            - reduce range by a factor <range_div> (default: 2.5).

    Specify optimization function <func>, it's arguments <args> and keyword
    arguments <kwargs>. The function should accept a dictionary of Parameter
    objects as it's first argument. It should return a scalar.
    """

    def __init__(
        self,
        func,
        args=(),
        kwargs={},
        n_eval=6,
        n_it=5,
        range_div=2.5,
        verbose=False,
        plot=False,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.params = {}
        self.n_eval = n_eval
        self.n_it = n_it
        self.verbose = verbose
        self.range_div = range_div
        self.plot = plot

    def add_parameter(self, p):
        self.params[p.name] = p

    def minimize(self, min_step=None):
        if self.plot:
            fig, [axes_list, min_list] = plt.subplots(
                2, len(self.params), sharex=False, sharey=False
            )

        for i_it in range(self.n_it):  # for n iterations
            # for each param to minimize
            for i_p, (pname, p) in enumerate(self.params.items()):
                # obtain initial value from parameter
                p_val0 = p.value

                # by default, sweep points are generated by np.linspace
                p_vals = np.linspace(
                    p_val0 - p.vrange / 2, p_val0 + p.vrange / 2, self.n_eval
                )

                # but if the step size is too small, ignore n_eval set by user and use p.minstep to decide sweep points with np.arange
                # include endpoint (that's what the +0.00001 is doing)
                if np.abs(p_vals[1] - p_vals[0]) < p.minstep:
                    p_vals = np.arange(p_vals[0], p_vals[-1] + 0.00001, p.minstep)
                # trivial case, nothing to minimize here
                if len(p_vals) == 1:
                    break

                vs = []
                # evaluate each value with the objective function and store it in a list
                for p_val in p_vals:
                    p.value = p_val
                    vs.append(self.func(self.params, *self.args, **self.kwargs))
                # find the minimum value evaluated
                vs = np.array(vs)
                imin = np.argmin(vs)
                p.value = p_vals[imin]
                # print additional info if verbose flag is on
                if self.verbose:
                    print(
                        f"Iteration: {i_it}, Param: {pname}, Minimized value = {vs[imin]:.3} at param value {p.value:.3}, Step size: {p_vals[1] - p_vals[0]:.3}"
                    )
                if self.plot:
                    axes_list[i_p].plot(p_vals, vs)

                    min_list[i_p].plot(p_vals, vs)
                    min_list[i_p].set_ylim(vs.min(), vs.max())
                    min_list[i_p].set_xlim(p_vals.min(), p_vals.max())

                p.vrange /= self.range_div  # reduce sweep range for next iteration

        # If live optimizing, set final parameters
        self.func(self.params, *self.args, **self.kwargs)

        return self.params


class MixerTuner:
    """
    Use tune_lo or tune_sb to minimize local oscillator leakage and upper sideband leakage respectively. Both methods require the mode object as well as a method key - either "BF" (default) to run Brute-Force minimizer or "NM" to run Nelder-Mead minimizer. The BF minimizer allows more control over the minimization but is slower. The NM minimizer is faster but its inner workings cannot be finely controlled.

    Note that the BF minimizer is minimizing the amplitude level whereas the NM minimizer is minimizing the contrast between signal peak and floor (to lower than the 3dB threshold).
    """

    # parameters common to both Nelder-Mead and Brute-Force minimization
    threshold: float = 3.0  # in dBm
    ref_power: float = 0.0

    # Nelder-Mead (NM) parameters
    simplex: np.ndarray = np.array([[0.0, 0.0], [0.0, 0.1], [0.1, 0.0]])
    maxiter: int = 100
    span: float = 2e6
    rbw: float = 50e3

    def __init__(self, sa: SA124) -> None:
        """ """
        self._sa: SA124 = sa
        self._qm: QuantumMachine = None
        self._mode: Mode = None
        self._mode_lo: LMS = None

        self._qm_job = None  # will be set and halted by the tune methods
        self._initial_contrast = None  # set during tuning check

    def _update_mixer_offsets(self, offsets) -> None:
        """ """
        self._mode.mixer_offsets = {**self._mode.mixer_offsets, **offsets}

    def tune_lo(self, mode: Mode, method: str = "BF", **kwargs):
        """ """
        self._setup(mode)  # will also set self._mode = mode
        # these three return values are only needed for NM, BF sets them to None
        is_tuned, center_idx, floor = self._check_tuning(method, "LO")
        i_offset, q_offset = None, None

        if method == "BF":
            i_offset, q_offset = self._tune_lo_bf(**kwargs)
        elif method == "NM":
            if is_tuned:
                logger.success(f"LO already tuned to within {self.threshold} dBm!")
                return
            i_offset, q_offset = self._tune_lo_nm(center_idx, floor)

        if i_offset is not None and q_offset is not None:
            self._update_mixer_offsets({"I": i_offset, "Q": q_offset})

        self._qm_job.halt()

    def tune_sb(self, mode: Mode, method: str = "BF", **kwargs):
        """ """
        self._setup(mode)  # will also set self._mode = mode
        # these three return values are only needed for NM, BF sets them to None
        is_tuned, center_idx, floor = self._check_tuning(method, "SB")
        g_offset, p_offset = None, None

        if method == "BF":
            g_offset, p_offset = self._tune_sb_bf(**kwargs)
        elif method == "NM":
            if is_tuned:
                logger.success("SB already tuned to within {self.threshold}dBm!")
                return
            g_offset, p_offset = self._tune_sb_nm(center_idx, floor)

        if g_offset is not None and p_offset is not None:
            self._update_mixer_offsets({"G": g_offset, "P": p_offset})

        self._qm_job.halt()

    def _setup(self, mode: Mode):
        """Play carrier frequency and intermediate frequency to mode and check current tuning"""
        self._mode = mode
        self._mode.add_operations(ConstantPulse("mixer_tuning_constant_pulse"))
        with Stage(remote=True) as stage:
            (self._mode_lo,) = stage.get(self._mode.lo_name)
        self._mode_lo.output = True
        self._qm = QM(modes=(self._mode,), oscillators=(self._mode_lo,))
        self._qm_job = self._qm.execute(self._get_qua_program(self._mode))

    def _check_tuning(self, method, key):
        """ """
        center = None
        if key == "LO":
            center = self._mode_lo.frequency
        elif key == "SB":
            center = self._mode_lo.frequency - self._mode.int_freq
        else:
            raise ValueError(f"{key = } is not valid, use 'LO' or 'SB'")

        if method == "BF":
            amp = self._sa.single_sweep(center=center, configure=True)
            logger.info(f"Initial amp of {key} leakage: {amp}")
            return None, None, None  # dummy values, they don't matter
        elif method == "NM":
            span, rbw, ref_pow = self.span, self.rbw, self.ref_power
            fs, amps = self._sa.sweep(
                center=center, span=span, rbw=rbw, ref_power=ref_pow
            )
            center_idx = math.ceil(self._sa.sweep_length / 2 + 1)
            stop, start = int(center_idx / 2), int(center_idx + (center_idx / 2))
            floor = (np.average(amps[:stop]) + np.average(amps[start:])) / 2
            contrast = amps[center_idx] - floor
            is_tuned = contrast < self.threshold
            real_center = fs[center_idx]
            logger.info(f"Tune check at {real_center:7E}: {contrast = :.5}dBm")
            self._initial_contrast = contrast
            return is_tuned, center_idx, floor
        else:
            raise ValueError(f"{method = } is not a valid key, use 'BF' or 'NM'")

    def _tune_lo_bf(self, offset_range, **kwargs):
        """ """
        range0, range1 = offset_range

        def objective_fn(params):
            i_offset, q_offset = params["I"].value, params["Q"].value
            self._qm.set_output_dc_offset_by_element(self._mode.name, "I", i_offset)
            self._qm.set_output_dc_offset_by_element(self._mode.name, "Q", q_offset)
            val = self._sa.single_sweep()
            logger.info(f"Measuring at I: {i_offset}, Q: {q_offset}, amp: {val}")
            return val

        mixer_offsets = self._mode.mixer_offsets
        i_offset, q_offset = mixer_offsets["I"], mixer_offsets["Q"]
        opt_params = self._minimize_bf(
            objective_fn, ("I", "Q"), i_offset, q_offset, range0, range1, **kwargs
        )
        return opt_params["I"].value, opt_params["Q"].value

    def _tune_lo_nm(self, center_idx, floor):
        """ """
        logger.info(f"Minimizing {self._mode} LO leakage...")

        def objective_fn(offsets: tuple[float]) -> float:
            i_offset, q_offset, mode_name = offsets[0], offsets[1], self._mode.name
            self._qm.set_output_dc_offset_by_element(mode_name, "I", i_offset)
            self._qm.set_output_dc_offset_by_element(mode_name, "Q", q_offset)
            contrast = self._get_contrast(center_idx, floor)
            logger.debug(f"Set I: {i_offset}, Q: {q_offset}. {contrast = }")
            return contrast

        result = self._minimize_nm(objective_fn, bounds=((-0.5, 0.5), (-0.5, 0.5)))
        return result if result is not None else (None, None)

    def _tune_sb_bf(self, offset_range, **kwargs):
        """ """
        range0, range1 = offset_range

        def objective_fn(params):
            g_offset, p_offset = params["G"].value, params["P"].value
            correction_matrix = self._qm._config.get_mixer_correction_matrix(
                g_offset, p_offset
            )
            self._qm_job.set_element_correction(self._mode.name, correction_matrix)
            val = self._sa.single_sweep()
            logger.info(f"Measuring at G: {g_offset}, P: {p_offset}, amp: {val}")
            return val

        mixer_offsets = self._mode.mixer_offsets
        g_offset, p_offset = mixer_offsets["G"], mixer_offsets["P"]
        opt_params = self._minimize_bf(
            objective_fn, ("G", "P"), g_offset, p_offset, range0, range1, **kwargs
        )
        return opt_params["G"].value, opt_params["P"].value

    def _tune_sb_nm(self, center_idx, floor):
        """ """
        logger.info(f"Minimizing {self._mode} SB leakage...")

        def objective_fn(offsets: tuple[float]) -> float:
            c_matrix = self._qm._config.get_mixer_correction_matrix(*offsets)
            if any(x < -2 or x > 2 for x in c_matrix):
                logger.info("Found out of bounds value in c-matrix")
                return self._initial_contrast
            self._qm_job.set_element_correction(self._mode.name, c_matrix)
            contrast = self._get_contrast(center_idx, floor)
            logger.debug(f"Set G: {offsets[0]}, P: {offsets[1]}. {contrast = }")
            return contrast

        result = self._minimize_nm(objective_fn, bounds=None)
        return result if result is not None else (None, None)

    def _minimize_bf(
        self,
        func,
        param_names,
        init0,
        init1,
        range0,
        range1,
        num_iterations=4,
        num_points=11,
        range_divider=4,
        plot=False,
        verbose=False,
    ):
        p0, p1 = param_names
        m = Minimizer(
            func,
            n_it=num_iterations,
            n_eval=num_points,
            range_div=range_divider,
            verbose=verbose,
            plot=plot,
        )
        m.add_parameter(Parameter(p0, value=float(init0), vrange=range0))
        m.add_parameter(Parameter(p1, value=float(init1), vrange=range1))
        m.minimize()
        return m.params

    def _minimize_nm(
        self,
        fn: Callable[[tuple[float]], float],
        bounds,
    ) -> tuple[float]:
        """ """
        start_time = time.perf_counter()
        opt = {
            "fatol": self.threshold,
            "initial_simplex": self.simplex,
            "maxiter": self.maxiter,
        }
        result = scipy.optimize.minimize(
            fn, [0, 0], method="Nelder-Mead", bounds=bounds, options=opt
        )
        if result.success:
            time_, contrast = time.perf_counter() - start_time, fn(result.x)
            logger.success(f"Minimized in {time_:.5}s with final {contrast = :.5}")
            if contrast > self.threshold:
                diff = contrast - self.threshold
                logger.warning(f"Final contrast exceeds threshold by {diff}dBm")
            return result.x
        else:
            logger.error(f"Minimization unsuccessful, details: {result.message}")

    def _get_qua_program(self, mode: Mode) -> _Program:
        """ """
        (pulse,) = mode.get_operations("mixer_tuning_constant_pulse")
        with program() as mixer_tuning:
            with infinite_loop_():
                mode.play(pulse)
        return mixer_tuning

    def _get_contrast(self, center_idx: int, floor: float) -> float:
        """ """
        _, amps = self._sa.sweep()
        return abs(amps[center_idx] - floor)

    def landscape(self, mode, key, xlim, ylim, points):
        """
        mode: the mode object
        key: string "LO" or " SB"
        xlim: tuple (min, max) for the landscape x axis
        ylim: tuple (min, max) for the landscape y axis
        points: number of points to sweep for x and y
        """
        # create landscape grid
        x = np.linspace(*xlim, points)
        y = np.linspace(*ylim, points)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros((points, points))

        self._setup(mode)

        # prepare SA for fast sweeps and prepare the objective functions
        func, center_idx = None, None
        if key == "LO":
            center = self._mode_lo.frequency
            # do this to set the SA
            self._sa.sweep(
                center=center, span=self.span, rbw=self.rbw, ref_power=self.ref_power
            )
            center_idx = math.ceil(self._sa.sweep_length / 2 + 1)

            def lo_fn(offsets: tuple[float]) -> float:
                self._qm.set_output_dc_offset_by_element(
                    self._mode.name, "I", offsets[0]
                )
                self._qm.set_output_dc_offset_by_element(
                    self._mode.name, "Q", offsets[1]
                )
                _, amps = self._sa.sweep()
                return amps[center_idx]

            func = lo_fn

        elif key == "SB":
            center = self._mode_lo.lo_freq - self._mode.int_freq  # upper sideband to suppress
            # do this to set the SA
            self._sa.sweep(
                center=center, span=self.span, rbw=self.rbw, ref_power=self.ref_pow
            )
            center_idx = math.ceil(self._sa.sweep_length / 2 + 1)

            def sb_fn(offsets: tuple[float]) -> float:
                c_matrix = self._qm._config.get_mixer_correction_matrix(*offsets)
                self._qm_job.set_element_correction(self._mode.name, c_matrix)
                _, amps = self._sa.sweep()
                return amps[center_idx]

            func = sb_fn

        logger.info(f"Finding {key} landscape for '{mode}'...")

        # evaluate the objective functions on the grid
        for i in range(points):
            for j in range(points):
                zz[i][j] = func((xx[i][j], yy[i][j]))
                logger.info(f"Set: {xx[i][j]}, {yy[i][j]}. Get {zz[i][j]}")

        # show final plot
        plt.pcolormesh(x, y, zz, shading="auto", cmap="viridis")
        plt.colorbar()

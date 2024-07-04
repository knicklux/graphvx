import os
import csv

import cvxpy as cp


def set_clarabel_accuracy(acc, settings):
    if acc < settings["tol_gap_abs"]:
        print("updating accuracy to: {}".format(acc))
    settings["tol_gap_abs"] = min(acc, settings["tol_gap_abs"])
    settings["tol_gap_rel"] = min(acc, settings["tol_gap_rel"])
    settings["tol_feas"] = min(acc, settings["tol_feas"])


def set_scs_accuracy(acc, settings):
    settings["eps_abs"] = min(acc, settings["eps_abs"])
    settings["eps_rel"] = min(acc, settings["eps_rel"])
    settings["eps_infeas"] = min(acc, settings["eps_infeas"])


def set_cvxopt_accuracy(acc, settings):
    settings["abstol"] = min(acc, settings["abstol"])
    settings["reltol"] = min(acc, settings["reltol"])


def set_sdpa_accuracy(acc, settings):
    settings["abstol"] = min(acc, settings["abstol"])
    settings["reltol"] = min(acc, settings["reltol"])


def set_solver_accuracy(solver, acc, settings):
    if solver == cp.SCS:
        set_scs_accuracy(acc, settings)
    elif solver == cp.CLARABEL:
        set_clarabel_accuracy(acc, settings)
    elif solver == cp.CVXOPT:
        set_cvxopt_accuracy(acc, settings)
    else:
        raise RuntimeError("automatic accuracy setting not implemented for solver")


def noop(*args, **kwargs):
    pass


def noop_factory():
    return lambda: noop


def constant_factory(value):
    return lambda: value


def append_hook(list):
    return lambda x: list.append(x)


def append_residuals_f(list_pri, list_dual):
    def append_residuals(info_dict):
        list_pri.append(info_dict["res_pri"])
        list_dual.append(info_dict["res_dual"])
    return lambda x: append_residuals(x)


def log_residuals_to_csv(path):
    def write_residuals(info_dict):
        primal_residual = info_dict["res_pri"]
        dual_residual = info_dict["res_dual"]
        iteration = info_dict["num_iterations"]

        # Check if the file exists to determine if we need to write the header
        file_exists = os.path.isfile(path)

        with open(path, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header if the file is new or if it is the first iteration
            if not file_exists or iteration == 1:
                writer.writerow(['iteration', 'res_pri', 'res_dual'])

            # Write the actual data
            writer.writerow([iteration, primal_residual, dual_residual])

    return lambda x: write_residuals(x)


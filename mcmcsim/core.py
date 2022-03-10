import typing
import os

import lmfit
import numpy as np
from emcee.backends.hdf import HDFBackend
from scipy import stats

np.random.seed(42)
import json

import corner
import emcee
import matplotlib.pyplot as plt
import pathlib
# from dataclasses_json import dataclass_json
import pydantic
from pydantic.dataclasses import dataclass
import dataclasses
from pydantic.json import pydantic_encoder

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

# Pydantic for numpy array (https://github.com/samuelcolvin/pydantic/issues/380#issuecomment-459352718)
class TypedArray(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        return np.array(val, dtype=cls.inner_type)

class ArrayMeta(type):
    def __getitem__(self, t):
        return type('Array', (TypedArray,), {'inner_type': t})

class Array(np.ndarray, metaclass=ArrayMeta):
    pass

@dataclass
class Variable():
    # name: str
    # values: Array[float]
    # sigma: typing.Union[Array[float], float]=None
    # vary: bool=True
    # user_arg: typing.Any=None
    def __init__(self,
        name: str,
        values: Array[float],
        sigma: typing.Union[Array[float], float]=None,
        vary: bool=True,
        user_arg: typing.Any=None,
    ) -> None:
        self.name = name
        self.values = np.squeeze(values)
        # self.sigma = np.squeeze(sigma)
        self.vary = vary
        self.user_arg = user_arg

        if sigma is not None:
            if np.isscalar(sigma):
                sigma_rsp = np.full_like(values, sigma)
            else:
                sigma_rsp = sigma.reshape(values.shape)
            self.sigma = sigma_rsp
        else:
            sigma_empty = np.empty_like(self.values)
            sigma_empty[:] = np.NaN
            self.sigma = sigma_empty

        return
class HDFBackendExt(HDFBackend):

    def get_outputs_history(self, flat=True, thin=1, discard=0):
        return self.get_blobs(flat=flat, thin=thin, discard=discard)


class MCMCRunner():

    def __init__(self,
        fn_calc_model: callable=None,
        parameters_ref: lmfit.Parameters=None,
        output_exp: typing.Dict[str, Variable]=None,
        inputs: typing.Dict[str, Variable]=None,
        kw_model_arg={},
        nwalkers=32,
        fpath='output_mcmc.h5',
        store_outputs=True,
        store_params_exp_and_inputs=True,
        initialize_from_h5=False,
        model_has_runner_ref=False,
        # sampler=None,
    ) -> None:
        self.fpath = fpath
        self.fn_calc_model = fn_calc_model
        self.kw_model_arg = kw_model_arg
        self.nwalkers = nwalkers
        self.store_outputs = store_outputs
        self.store_params_exp_and_inputs = store_params_exp_and_inputs
        self.parameters_ref = parameters_ref
        self.output_exp = output_exp
        self.inputs = inputs
        self.model_has_runner_ref = model_has_runner_ref

        # if store_params_exp_and_inputs:
        #     path = pathlib.Path(self.fpath).resolve()
        #     path_json = path.parent / '{}.json'.format(path.stem)
        #     # input_json = json.dumps(self.inputs, cls=NpEncoder)
        #     obj = {
        #         'inputs': inputs,
        #         'output_exp': output_exp,
        #         'parameters_ref': parameters_ref.dumps(),
        #         'nwalkers': self.nwalkers,
        #     }
        #     with open(path_json, 'w') as f:
        #         json.dump(obj, f, cls=NpEncoder)

        if initialize_from_h5:
            self.initialize_from_h5file()

        self.parameter_mod = self.parameters_ref.copy() if self.parameters_ref else None
        p_varies = [p for p in self.parameters_ref.values() if p.vary]
        self.labels = [p.name for p in p_varies]
        self.n_pars = len(p_varies)

        self.sampler = None

        if parameters_ref is None or output_exp is None:
            self.set_sampler(reset=False)

        if self.model_has_runner_ref:
            self.kw_model_arg = {'runner': self, **self.kw_model_arg}

        pass

    def initialize_from_h5file(self):
        # path = pathlib.Path(self.fpath).resolve()
        backend_ = emcee.backends.HDFBackend(self.fpath)

        # path_json = path.parent / '{}.json'.format(path.stem)
        with backend_.open('r') as f:
            g = f[backend_.name]
            vary_input_dic = json.loads(g['inputs'].attrs['vary'])
            vary_output_dic = json.loads(g['output_exp'].attrs['vary'])
            ds_inputs = g['inputs']
            rec_inputs = ds_inputs[0]
            ds_output_exp = g['output_exp']
            rec_output_exp = ds_output_exp[0]

            input_dic = record_h5_to_variable_dic(rec_inputs, vary_input_dic)
            output_dic = record_h5_to_variable_dic(rec_output_exp, vary_output_dic)
            self.inputs = input_dic
            self.output_exp = output_dic

            str_params = g.attrs['params_ref_json']

            param_load = lmfit.Parameters()
            param_load.loads(str_params)
            self.parameters_ref = param_load

    def set_parameters_ref(self, p):
        self.parameters_ref = p

    def set_sampler(self, sampler=None, reset=False):
        if sampler is None:
            # p_varies = [p for p in self.parameters_ref.values() if p.vary]
            # labels = [p.name for p in self.parameters_ref.values() if p.vary]
            # n_pars = len(p_varies)
            backend_ = HDFBackendExt(self.fpath)
            # backend_ = HDFBackendExtended(self.fpath, self.output_exp,
            #     store_params_exp_and_inputs=self.store_params_exp_and_inputs,
            #     # parameters_ref=self.parameters_ref,
            #     # inputs=self.inputs,
            # )

            if reset:
                backend_.reset(self.nwalkers, self.n_pars)
            # dtype = [("log_prior", float), ("mean", float)]
            if self.store_outputs:
                # dtypes = [(key, np.ndarray) for key in self.output_exp]
                dtypes = [(key, np.float, self.output_exp[key].values.size) for key in self.output_exp]
            else:
                dtypes = None
            # dtypes = [('out1', float)]
            sampler = emcee.EnsembleSampler(self.nwalkers, self.n_pars,
                self.log_probability_dict, backend=backend_, parameter_names=self.labels,
                blobs_dtype=dtypes,
            )

            # if self.sampler is None and self.store_params_exp_and_inputs:
            #     sampler.backend.save_params_ref_experiments_and_inputs()

            self.sampler = sampler

            # Save the input and experimental to hdf file
            if self.store_params_exp_and_inputs:
                input_as_array = dict_to_record_sigma_var(self.inputs)
                output_exp_as_array = dict_to_record_sigma_var(self.output_exp)

                with backend_.open("a") as f:
                    g = f[backend_.name]
                    if 'inputs' not in g:
                        g.create_dataset('inputs', data=input_as_array)
                    if 'output_exp' not in g:
                        g.create_dataset('output_exp', data=output_exp_as_array)
                    g.attrs['params_ref_json'] = self.parameters_ref.dumps()
                    input_vary_dic = {k: v.vary for k, v in self.inputs.items()}
                    output_vary_dic = {k: v.vary for k, v in self.output_exp.items()}
                    g['inputs'].attrs['vary'] = json.dumps(input_vary_dic)
                    g['output_exp'].attrs['vary'] = json.dumps(output_vary_dic)
                    input_userarg_dic = {k: v.user_arg for k, v in self.inputs.items()}
                    output_userarg_dic = {k: v.user_arg for k, v in self.inputs.items()}
                    g['inputs'].attrs['user_arg'] = json.dumps(input_userarg_dic)
                    g['output_exp'].attrs['user_arg'] = json.dumps(output_userarg_dic)


            return backend_
        else:
            self.sampler = sampler

    def run_mcmc(self, nsteps, load_backend=True,
        sigma_initial_pos=1e-4, fpath=None,
        **kw_ags_run):

        if fpath is not None:
            self.fpath = fpath
        if load_backend:
            backend = self.set_sampler(reset=False)
            # backend = emcee.backends.HDFBackend(fpath)
            print("Initial size: {0}".format(backend.iteration))
            if backend.iteration == 0:
                pos = self.create_initial_position(sigma_initial_pos)
            else:
                pos = None
            self.sampler.run_mcmc(pos, nsteps, **kw_ags_run)
        else:
            backend = self.set_sampler(reset=True)
            pos = self.create_initial_position(sigma_initial_pos)
            self.sampler.run_mcmc(pos, nsteps, **kw_ags_run)
        print("Final size: {0}".format(backend.iteration))
        return

    def residual(self, p) -> typing.Dict[str,np.ndarray]:
        "Calculate the residual"
        y_calc_dict = self.fn_calc_model(p, self.inputs, **self.kw_model_arg)
        res_dict = {
            key: (y_calc_dict[key] - self.output_exp[key].values)/self.output_exp[key].sigma
            for key in self.output_exp
            if self.output_exp[key].vary
        }

        if self.store_outputs and self.sampler:
            self.sampler.backend.output_calc = y_calc_dict

            # Save calculateds
            self.__y_calc_dict = y_calc_dict

        return res_dict

    def calc_least_sqr_sum(self, p) -> typing.Union[float, np.ndarray]:
        "Calculate least square error"
        errors_dict = self.residual(p)
        sum_sqs = np.sum([err**2 for err in errors_dict.values()])
        return sum_sqs

    def calc_maximum_loglikelihood(self, p) -> typing.Union[float, np.ndarray]:
        "Calculate loglikelihood"
        sum_sqs = self.calc_least_sqr_sum(p)
        sigma2_terms = np.array([np.log(2*np.pi*val.sigma**2)
            for val in self.output_exp.values()
            if val.vary
        ])
        log_likehood = -0.5 * (sum_sqs + np.sum(sigma2_terms))

        return log_likehood

    def calc_minus_loglikelihood(self, p):
        return -self.calc_maximum_loglikelihood(p)

    def calc_log_prior_uniform(self, params):
        "Calculate uniform log prior"
        n_pars = get_num_params(params)
        p_varies = [p for p in params.values() if p.vary]
        log_prior = np.empty(n_pars)
        for i, p in enumerate(p_varies):
            log_prior[i] = 0.0 if p.min < p.value < p.max else -np.inf
        return log_prior.sum()

    def log_probability(self, p):
        lp = self.calc_log_prior_uniform(p)
        # if not np.isfinite(lp):
        #     return -np.inf, -2.0
        ls_sum = self.calc_maximum_loglikelihood(p)
        ln_post = lp + ls_sum if np.isfinite(lp) else -np.inf

        if not self.store_outputs:
            return ln_post

        vals_outs = list(self.__y_calc_dict.values())
        out_lst = [ln_post, *vals_outs]
        out_tp = tuple(out_lst)
        # out_tp = ln_post, np.array([[-1, 3.0], [-4, 5.0]])
        # print(out_tp)
        # return ln_post, np.array([[-1, 3.0], [-4, 5.0]])
        # return (
        #     ln_post,
        #     np.random.randn(10),
        # )
        return out_tp

    def log_probability_dict(self, p_dic):
        for key in p_dic:
            self.parameter_mod[key].value = p_dic[key]
        llh = self.log_probability(self.parameter_mod)
        return llh

    def log_probability_array(self, arr):
        params_ = create_params_from_ref_updating_vals(self.parameters_ref, arr)
        llh = self.log_probability(params_)
        return llh

    def create_initial_position(self, sigma_initial_pos):
        opt_vals = get_pars_vals_vary(self.parameters_ref)
        pos = opt_vals + sigma_initial_pos*np.random.randn(self.nwalkers, self.n_pars)
        return pos

    def plot_corner(self, thin=1, discard=0, kw_corner={}):
        flat_samples = self.sampler.get_chain(flat=True, thin=thin, discard=discard)

        fig = corner.corner(
            flat_samples, labels=self.labels, **kw_corner,
        );
        return fig

    def plot_chain(self, thin=1, discard=0):
        fig, axes = plt.subplots(self.n_pars, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain(thin=thin, discard=discard)
        for i in range(self.n_pars):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        #     ax.set_xlim(0, 100)
        axes[-1].set_xlabel("step number");

        return

    def get_parameter_interval_from_marginalized(self, thin=1, discard=0,
        latex_formated=False, quantiles=[16, 50, 84]):
        samples = self.sampler.get_chain(flat=True, thin=thin, discard=discard)

        dic_intervals = {}
        for i in range(self.n_pars):
            mcmc = np.percentile(samples[:, i], quantiles)
            q = np.diff(mcmc)
            dic_intervals[self.labels[i]] = {
                'median': mcmc[1],
                'lower': q[0],
                'upper': q[1],
            }
            if latex_formated:
                txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
                txt = txt.format(mcmc[1], q[0], q[1], self.labels[i])
                dic_intervals[self.labels[i]]['latex'] = txt
            # display(Math(txt))

        return dic_intervals

    def plot_model_output(self, p, input_name=None, output_names=None,
        show_exp=True, kw_plots=None, kw_exps=None):
        "Auxialiry function to plot outputs. The input size should match the number of outputs"
        assert len(self.output_exp) < 10, 'This plot is limited for 9 outputs'
        map_subplot = {
            1: (1, 1),
            2: (1, 2),
            3: (1, 3),
            4: (2, 2),
            5: (2, 3),
            6: (2, 3),
            7: (3, 3),
            8: (3, 3),
            9: (3, 3),
        }

        if input_name is None:
            input_name = list(self.inputs.keys())[0]
        if output_names is None:
            output_names = list(self.output_exp.keys())
        if kw_plots is None:
            kw_plots = [{}] * len(output_names)
        if kw_plots is None:
            kw_exps = [{}] * len(output_names)

        y_calc_dict = self.fn_calc_model(p, self.inputs, **self.kw_model_arg)

        nsubs = map_subplot[len(output_names)]
        plt.figure()
        for i, kout in enumerate(output_names):
            plt.subplot(*nsubs, i+1)
            plt.plot(self.inputs[input_name].values, y_calc_dict[kout], **kw_plots[i])
            if show_exp:
                plt.plot(self.inputs[input_name].values, self.output_exp[kout].values, **kw_plots[i])
            plt.xlabel('{}'.format(input_name))
            plt.ylabel('{}'.format(kout))


    def plot_mcmc_outputs(self, input_name=None, output_names=None,
        thin=1, discard=0,
        show_exp=True, kw_plots=None, kw_exps=None, cmap=None,
        save_no_show=False, savefig_args=None, fig_kws={'figsize':(12,8)}):
        "Auxialiry function to plot outputs from many samples. The input size should match the number of outputs"
        assert len(self.output_exp) < 10, 'This plot is limited for 9 outputs'
        map_subplot = {
            1: (1, 1),
            2: (1, 2),
            3: (1, 3),
            4: (2, 2),
            5: (2, 3),
            6: (2, 3),
            7: (3, 3),
            8: (3, 3),
            9: (3, 3),
        }

        if input_name is None:
            input_name = list(self.inputs.keys())[0]
        if output_names is None:
            output_names = list(self.output_exp.keys())
        if kw_plots is None:
            kw_plots = [{}] * len(output_names)
        if kw_exps is None:
            kw_exps = [{'marker': 'o', 'linestyle': 'None'}] * len(output_names)


        # y_calc_dict = self.fn_calc_model(p, self.inputs, **self.kw_model_arg)
        # y_output_hist = self.get_outputs_history(flat=True, thin=thin, discard=discard)
        y_output_hist = self.sampler.get_blobs(flat=True, thin=thin, discard=discard)
        ln_Probs = self.sampler.get_log_prob(flat=True, thin=thin, discard=discard)
        ln_Probs_norm = (ln_Probs - ln_Probs.min()) / (ln_Probs.max() - ln_Probs.min())
        # probs = np.exp(ln_Probs)

        key_out0 = list(self.output_exp.keys())[0]
        size_iterations = y_output_hist[key_out0].shape[0]

        if cmap is None:
            cm = plt.get_cmap('gray_r')
            # colors = cm(np.linspace(0, 0.7, size_iterations))
            ln_Probs_norm_adj = ln_Probs_norm * 0.7
            colors = cm(ln_Probs_norm_adj)
            # colors = [str(c) for c in cm(np.linspace(0, 0.7, size_iterations))]


        nsubs = map_subplot[len(output_names)]
        fig, axes = plt.subplots(*nsubs, **fig_kws)
        axes = np.atleast_1d(axes)
        axes_flat = axes.reshape((-1,))

        for i, kout in enumerate(output_names):
            for k in range(size_iterations):
                # plt.subplot(*nsubs, i+1)
                axes_flat[i].plot(self.inputs[input_name].values, y_output_hist[kout][k],
                    color=colors[k],
                    **kw_plots[i])
            axes_flat[i].set_xlabel('{}'.format(input_name))
            axes_flat[i].set_ylabel('{}'.format(kout))

        if show_exp:
            for i, kout in enumerate(output_names):
                # plt.subplot(*nsubs, i+1)
                axes_flat[i].plot(self.inputs[input_name].values, self.output_exp[kout].values,
                    **kw_exps[i])

        if save_no_show:
            if savefig_args is None:
                savefig_args = dict(fname='mcmc_outputs.png', dpi=300)
            plt.savefig(**savefig_args)
            plt.close()
        return fig

    # def get_outputs_history(self, flat=False, thin=1, discard=0):
    #     return self.sampler.backend.get_outputs_history(flat=flat, thin=thin, discard=discard)
    def get_outputs_history(self, flat=True, thin=1, discard=0):
        return self.sampler.get_blobs(flat=flat, thin=thin, discard=discard)


def display_marginalized_intervals(runner, sample_kw):
    from IPython.display import display, Math
    params_intervals = runner.get_parameter_interval_from_marginalized(latex_formated=True, **sample_kw)
    latexes = [item['latex'] for item in params_intervals.values()]
    [display(Math(txt)) for txt in latexes];


def get_color_vector_based_on_prob(ln_Probs, cm=plt.get_cmap('gray_r'), upper_color=0.7):
    ln_Probs_norm = (ln_Probs - ln_Probs.min()) / (ln_Probs.max() - ln_Probs.min())
    ln_Probs_norm_adj = ln_Probs_norm * upper_color
    colors = cm(ln_Probs_norm_adj)
    return colors


#%%%%%%%%%%%%%%%%%%%%%%%% PRIVATE

def dict_to_record_sigma_var(d_var):
    dtypes_inputs = [(key, np.float, d_var[key].values.size) for key in d_var]
    dtypes_inputs = []
    input_h5_arr = []
    for key in d_var:
        dt_val = (key, np.float, d_var[key].values.size)
        sigma_val = ('sigma_{}'.format(key), np.float, d_var[key].values.size)
        dtypes_inputs += [dt_val]
        dtypes_inputs += [sigma_val]

        inp = d_var[key]
        input_h5_arr += [inp.values]
        input_h5_arr += [inp.sigma]

        input_h5_tp = tuple(input_h5_arr)
        input_h5_lst = [input_h5_tp]

    input_as_array = np.array(input_h5_lst, dtypes_inputs)
    return input_as_array

def record_h5_to_variable_dic(rec_inputs, vary_dict):
    inputs_names = rec_inputs.dtype.names[0::2]
    input_dic = {}
    for inp_key in inputs_names:
        v = rec_inputs[inp_key]
        s = rec_inputs['sigma_{}'.format(inp_key)]
        vary = vary_dict[inp_key]
        aux = Variable(inp_key, v, s, vary)
        input_dic[inp_key] = aux
    return input_dic



def get_dfree(params, runner):
    keys = list(runner.inputs.keys())
    vals = runner.inputs[keys[0]].values
    n_e = len(vals)
    n_y = len(runner.output_exp)
    npar = len([v for v in params.values() if v.vary])
    dfree = n_e * n_y - npar
    return dfree

























# FORMER

def residual(p, fn_model, args_model, y_exp, sigma2):
    "Calculate the residual"
    return (fn_model(p, *args_model) - y_exp)/np.sqrt(sigma2)

def calc_least_sqr_sum(p, fn_model, args_model, y_exp, sigma2):
    "Calculate least square error"
    errors = residual(p, fn_model, args_model, y_exp, sigma2)
    sum_sqs = np.sum(errors**2)
    return sum_sqs

def calc_maximum_loglikelihood(p, fn_model, args_model, y_exp, sigma2):
    "Calculate loglikelihood"
    sum_sqs = calc_least_sqr_sum(p, fn_model, args_model, y_exp, sigma2)
    log_likehood = -0.5 * (sum_sqs + np.sum(np.log(2*np.pi*sigma2)))
    return log_likehood

def calc_minimize_loglikelihood(p, fn_model, args_model, y_exp, sigma2):
    "Minimize loglikelihood function"
    return -calc_maximum_loglikelihood(p, fn_model, args_model, y_exp, sigma2)

def calc_log_prior_uniform(params):
    "Calculate log prior"
    n_pars = get_num_params(params)
    log_prior = np.empty(n_pars)
    for i, p in enumerate(params.values()):
        log_prior[i] = 0.0 if p.min < p.value < p.max else -np.inf
    return log_prior.sum()

def log_probability(p, fn_model, args_model, y_exp, sigma2):
    lp = calc_log_prior_uniform(p)
    if not np.isfinite(lp):
        return -np.inf
    ls_sum = calc_maximum_loglikelihood(p, fn_model, args_model, y_exp, sigma2)
    return lp + ls_sum

def log_probability_array(arr, fn_model, params_ref, args_model, y_exp, sigma2):
    params_ = create_params_from_ref_updating_vals(params_ref, arr)
    llh = log_probability(params_, fn_model, args_model, y_exp, sigma2)
    return llh

def calc_sqr_sum_from_log_likelihood(log_lh, sigma2):
    sqr = -2*log_lh - np.sum(np.log(2*np.pi*sigma2))
    return sqr



#-------------------------------------------------
#-------------------------------------------------
#    OLDER
#-------------------------------------------------
#-------------------------------------------------

# Generate random parameters set near optimal solution
def generate_random_params_values_near_opt(params_f, n_steps = 1000, sigma=0.1):
    num_p = len([p for p in params_f.values() if p.vary])
    means = np.array(list(params_f.valuesdict().values()))
    if np.isscalar(sigma):
        sigmas = np.full(len(means), sigma)
    else:
        sigmas = sigma

    sampling_randoms = np.random.normal(loc=means, scale=sigmas, size=(n_steps,num_p))

    return sampling_randoms

def calc_F_fisher(num_p, num_exps, alpha=0.975):
    dfree_fisher = (num_p, num_exps-num_p)
    F_fisher = stats.f.ppf(alpha,*dfree_fisher)
    return F_fisher

def calc_threshold_Fobj(Fobj_opt, num_p, num_exps, alpha=0.975):
    F_fisher = calc_F_fisher(num_p, num_exps, alpha=alpha)
    factor = (1 + num_p/(num_exps-num_p) * F_fisher)
    F_threshold = Fobj_opt * factor
    return F_threshold

def get_pars_vals_vary(params):
    return np.array([p for p in params.values() if p.vary])

def get_num_params(params):
    return len([p for p in params.values() if p.vary])

def corner_plot_add_cross_lines(figure, values, color='g'):
    num_p = len(values)
    axes = np.array(figure.axes).reshape((num_p, num_p))
    # Loop over the diagonal
    for i in range(num_p):
        ax = axes[i, i]
        ax.axvline(values[i], color=color)

    # Loop over the histograms
    for yi in range(num_p):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(values[xi], color=color)
            ax.axhline(values[yi], color=color)
            ax.plot(values[xi], values[yi], "s", color=color)
    return

def create_params_from_ref_updating_vals(params_ref, values):
    "Auxiliary function to create lmfit.Parameters"
#     params_new = deepcopy(params_ref)
    params_new = params_ref.copy()
    for i, key in enumerate(params_new):
        params_new[key].value = values[i]
    return params_new


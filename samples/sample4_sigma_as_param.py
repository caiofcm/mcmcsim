
import matplotlib.pyplot as plt
import numpy as np
import lmfit
import mcmcsim


def calc_model_new(p, inputs_dict, runner=None):
    x = inputs_dict['x'].values
    y = p['a1']*np.exp(-x/p['t1']) + p['a2']*np.exp(-(x-0.1) / p['t2'])

    # Settting sigma vector from parameters
    if runner:
        y_exp = runner.output_exp['y'].values
        runner.output_exp['y'].sigma = p['factor_sigma']*y_exp
    return {
        'y': y,
    }


def main():

    "Creating true parameters and generating fake data"
    p_true = lmfit.Parameters()
    p_true.add_many(('a1', 3.), ('a2', -5.), ('t1', 2.), ('t2', 10.))
    sigma = np.array([0.4])

    num_out = len(sigma)
    x = np.linspace(1, 10, 100)

    inputs_dict = {
        'x': mcmcsim.Variable('x', x),
    }

    error_y = sigma*np.random.randn(x.size, num_out)
    y_true_dict = calc_model_new(p_true, inputs_dict)
    y_true = np.column_stack([v for v in y_true_dict.values()])
    y_exp = y_true + error_y

    "Using the dict approach for experimental data and inputs"
    y_exp_dict = {
        'y': mcmcsim.Variable('y', y_exp[:,0], np.full_like(y_exp[:,0], sigma[0])),
    }

    "Setting test parameters for estimate"
    params0 = lmfit.Parameters()
    params0.add('a1', 4.0, min=-10.0, max=10.0)
    params0.add('a2', 4.0, min=-10.0, max=10.0)
    params0.add('t1', 3.0, min=0.1, max=10.0)
    params0.add('t2', 8.0, min=0.1, max=20.0)
    params0.add('factor_sigma', 0.05, min=0.0, max=1.0)

    y_out = calc_model_new(params0, inputs_dict)

    "Creating the auxiliary MCMC Runner"
    runner = mcmcsim.MCMCRunner(calc_model_new, params0, y_exp_dict, inputs_dict,
        fpath='output_mcmcs.h5', store_outputs=True, store_params_exp_and_inputs=True,
        model_has_runner_ref=True,
    )

    "Calculating the (-) log likelihood"
    f_ls = runner.calc_minus_loglikelihood(params0)
    print(f_ls)

    "Regular parameter estimation (using lmfit)"
    mini = lmfit.Minimizer(runner.calc_minus_loglikelihood, params0, nan_policy='propagate')

    out1 = mini.minimize(method='Nelder')
    lmfit.report_fit(out1)
    params_f = out1.params

    "Updating reference parameter in Runner (idealy, it will be passed in the constructor)"
    runner.set_parameters_ref(params_f)

    "Calculating the outputs using the estimated parameters"
    y_out_optimal = calc_model_new(params_f, inputs_dict)

    "View"
    plt.figure()
    plt.plot(x, y_true)
    plt.plot(x, y_exp)
    plt.plot(x, y_out['y'])
    plt.plot(x, y_out_optimal['y'])

    "Running MCMC"

    runner.run_mcmc(load_backend=False, nsteps=5000, progress=True)


    "Visualizing"

    runner.plot_corner(discard=500)

    runner.plot_chain()

    params_intervals = runner.get_parameter_interval_from_marginalized(latex_formated=True)
    print(params_intervals)

    runner.plot_model_output(params_f)

    "Visualize the outputs"

    runner.plot_mcmc_outputs(discard=500, thin=30, save_no_show=True)
    print('Image saved to file')

    plt.show()


if __name__ == "__main__":
    main()

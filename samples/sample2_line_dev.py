
import matplotlib.pyplot as plt
import numpy as np
import lmfit
import mcmcsim

def calc_model_line(p, inputs_dict, extra_arg_kw=None):
    x = inputs_dict['x'].values
    y = p['m'] * x + p['b']
    return {
        'y': y,
    }

def main():

    "Creating true parameters and generating fake data"
    p_true = lmfit.Parameters()
    p_true.add_many(('m', 3.), ('b', -5.))
    sigma = np.array([10.0])

    num_out = len(sigma)
    x = np.linspace(1, 10, 250)

    inputs_dict = {
        'x': mcmcsim.Variable('x', x),
    }

    error_y = sigma*np.random.randn(x.size, num_out)
    y_true_dict = calc_model_line(p_true, inputs_dict)
    y_true = np.column_stack([v for v in y_true_dict.values()])
    y_exp = y_true + error_y

    "Using the dict approach for experimental data and inputs"
    y_exp_dict = {
        'y': mcmcsim.Variable('y', y_exp[:,0], np.full_like(y_exp[:,0], sigma[0])),
    }

    "Setting test parameters for estimate"
    params0 = lmfit.Parameters()
    params0.add('m', 4.0, min=-10.0, max=10.0)
    params0.add('b', 4.0, min=-10.0, max=10.0)

    y_out = calc_model_line(params0, inputs_dict)

    "Creating the auxiliary MCMC Runner"
    runner = mcmcsim.MCMCRunner(calc_model_line, params0, y_exp_dict, inputs_dict,
        fpath='output_mcmcs.h5', store_outputs=True, store_params_exp_and_inputs=True,
    )

    "Calculating the (-) log likelihood"
    f_ls = runner.calc_minus_loglikelihood(params0)
    print(f_ls)

    "Regular parameter estimation (using lmfit)"
    mini = lmfit.Minimizer(runner.calc_minus_loglikelihood, params0, nan_policy='omit')

    out1 = mini.minimize(method='Nelder')
    lmfit.report_fit(out1)
    params_f = out1.params

    "Updating reference parameter in Runner (idealy, it will be passed in the constructor)"
    runner.set_parameters_ref(params_f)

    "Calculating the outputs using the estimated parameters"
    y_out_optimal = calc_model_line(params_f, inputs_dict)

    "View"
    plt.figure()
    plt.plot(x, y_true)
    plt.plot(x, y_exp)
    plt.plot(x, y_out['y'])
    plt.plot(x, y_out_optimal['y'])

    "First run MCMC"

    runner.run_mcmc(load_backend=False, nsteps=50, progress=True)

    "Second run MCMC"

    runner.run_mcmc(load_backend=True, nsteps=50, progress=True)

    "Visualizing"

    # runner.plot_corner(discard=0)

    sample_kw = {'thin': 10, 'discard': 10}

    "Visualize the outputs"

    runner.plot_mcmc_outputs(save_no_show=False, **sample_kw)
    print('Image saved to file')

    plt.show()

if __name__ == "__main__":
    main()

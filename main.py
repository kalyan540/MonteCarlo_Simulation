import monaco as mc
from scipy.stats import triang
import numpy as np
import matplotlib.pyplot as plt
from math import ceil


# Define preprocess function
def preprocess(case):
    # Sample inputs for each case (product price, sales, costs)
    unit_price = case.invals['Unit Price'].val
    unit_sales = case.invals['Unit Sales'].val
    variable_costs = case.invals['Variable Costs'].val
    fixed_costs = case.invals['Fixed Costs'].val

    # Return the inputs as a tuple for the run function
    return (unit_price, unit_sales, variable_costs, fixed_costs)


# Define run function to calculate earnings
def run(unit_price, unit_sales, variable_costs, fixed_costs):
    earnings = (unit_price * unit_sales) - (variable_costs + fixed_costs)
    return earnings  # Returning as tuple


# Define postprocess function
def postprocess(case, earnings):
    # You can process or format the earnings data here if needed
    case.addOutVal('Earnings', earnings)  # Storing the output  `


# Define the simulation setup
def product_earnings_monte_carlo_sim():
    # Monte Carlo simulation configuration
    fcns = {
        'preprocess': preprocess,
        'run': run,
        'postprocess': postprocess
    }

    # Define number of draws and seed
    ndraws = 1000  # Number of simulations
    seed = 12345678  # Random seed

    # Initialize the Monte Carlo Simulation
    sim = mc.Sim(name='product_earnings', ndraws=ndraws, fcns=fcns,
                 firstcaseismedian=True, seed=seed, singlethreaded=False,
                 verbose=True, debug=False)

    # Define the triangular distributions for input variables
    # Example: Unit price with a triangular distribution between 50 and 200
    sim.addInVar(name='Unit Price', dist=triang, distkwargs={'c': 0.5, 'loc': 50, 'scale': 150})

    # Example: Unit sales (random variable between 5000 and 15000)
    sim.addInVar(name='Unit Sales', dist=triang, distkwargs={'c': 0.5, 'loc': 5000, 'scale': 10000})

    # Example: Variable costs (random variable between 10 and 50 per unit)
    sim.addInVar(name='Variable Costs', dist=triang, distkwargs={'c': 0.5, 'loc': 10, 'scale': 40})

    # Example: Fixed costs (random variable between 100000 and 500000)
    sim.addInVar(name='Fixed Costs', dist=triang, distkwargs={'c': 0.5, 'loc': 100000, 'scale': 400000})

    # Run the simulation
    sim.runSim()
    print(sim.outvars.keys())

    # Add output variable 'Earnings' to the simulation and store results
    #sim.outvars['Earnings'] = mc.OutVar(name='Earnings', vals=sim.outvars['run'], ndraws=ndraws)

    # Add statistics to output variables (e.g., earnings)
    sim.outvars['Earnings'].addVarStat(stat='orderstatTI', statkwargs={'p': 0.5, 'c': 0.95, 'bound': '2-sided'})

    # Plot the earnings distribution
    fig, ax = mc.plot(sim.outvars['Earnings'], plotkwargs={'bins': 50})
    ax.set_autoscale_on(False)
    ax.set_title('Product Earnings Distribution')
    fig.set_size_inches(8.0, 4.5)
    plt.savefig('product_earnings_histogram.png', dpi=100)

    # Perform sensitivity analysis for the input parameters
    # Sensitivity analysis for 'Unit Price', 'Unit Sales', 'Variable Costs', 'Fixed Costs'
    ## Calculate and plot sensitivity indices
    sim.calcSensitivities('Earnings')
    fig_sens, ax = sim.outvars['Earnings'].plotSensitivities()
    fig_sens.set_size_inches(8.0, 4.5)
    fig_sens.savefig('sensitivity_analysis_input_params.png', dpi=100)

    # Optionally, print some statistics on the earnings
    earnings_mean = np.mean(sim.outvars['Earnings'].vals)
    earnings_5th_percentile = np.percentile(sim.outvars['Earnings'].vals, 5)
    earnings_95th_percentile = np.percentile(sim.outvars['Earnings'].vals, 95)
    print(f'Mean Earnings: ${earnings_mean:,.2f}')
    print(f'5th Percentile: ${earnings_5th_percentile:,.2f}')
    print(f'95th Percentile: ${earnings_95th_percentile:,.2f}')



    return sim


if __name__ == '__main__':
    # Run the Monte Carlo simulation
    sim = product_earnings_monte_carlo_sim()

    # Optionally, you can save the simulation data or analyze further
    sim.saveSimToFile()  # Save the simulation results to a .mcsim file

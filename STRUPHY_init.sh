# copy run and diagnostics files to first level
cp simulations/run_STRUPHY_cc_lin_6D.sh run_STRUPHY_cc_lin_6D.sh

cp simulations/diagnostics.sh diagnostics.sh
cp -r hylife/diagnostics/. my_diagnostics/

# create a default simulations folder (default: in the repos directory)
mkdir -p my_struphy_sims

# copy template 
cp -r simulations/template_slab my_struphy_sims/sim_1
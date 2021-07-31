# copy main code and run file to first level
cp simulations/STRUPHY_cc_lin_6D.py STRUPHY_cc_lin_6D.py
cp simulations/run_STRUPHY_cc_lin_6D.sh run_STRUPHY_cc_lin_6D.sh

# create a default simulations folder in the root repo
mkdir -p my_struphy_sims

# copy template 
cp -r simulations/template_slab my_struphy_sims/sim_1
def test_lin_mhd():
    import os

    os.system('struphy -r lin_mhd    -i parameters.yml      -o sim_2')
    os.system('struphy -r lin_mhd_MF -i parameters_u1p0.yml -o sim_3')
    os.system('struphy -r lin_mhd_MF -i parameters_u1p3.yml -o sim_4')
    os.system('struphy -r lin_mhd_MF -i parameters_u2p0.yml -o sim_5')
    os.system('struphy -r lin_mhd_MF -i parameters_u2p3.yml -o sim_6')


def test_cc_lin_mhd_6d():
    import os
    
    os.system('struphy -r cc_lin_mhd_6d_MF -i parameters_u1p0.yml')
    os.system('struphy -r cc_lin_mhd_6d_MF -i parameters_u1p3.yml')
    os.system('struphy -r cc_lin_mhd_6d_MF -i parameters_u2p0.yml')
    os.system('struphy -r cc_lin_mhd_6d_MF -i parameters_u2p3.yml')


if __name__ == '__main__':
    test_lin_mhd()
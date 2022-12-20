def test_lin_mhd():
    import os

    os.system('struphy run lin_mhd    -i parameters.yml      -o sim_2')
    os.system('struphy run lin_mhd_MF -i parameters_u1p0.yml -o sim_3')
    os.system('struphy run lin_mhd_MF -i parameters_u1p3.yml -o sim_4')
    os.system('struphy run lin_mhd_MF -i parameters_u2p0.yml -o sim_5')
    os.system('struphy run lin_mhd_MF -i parameters_u2p3.yml -o sim_6')


def test_cc_lin_mhd_6d_MF():
    import os
    
    os.system('struphy -run cc_lin_mhd_6d_MF -i parameters_u1p0.yml')
    os.system('struphy -run cc_lin_mhd_6d_MF -i parameters_u1p3.yml')
    os.system('struphy -run cc_lin_mhd_6d_MF -i parameters_u2p0.yml')
    os.system('struphy -run cc_lin_mhd_6d_MF -i parameters_u2p3.yml')

def test_pc_lin_mhd_6d_MF_full():
    import os
    
    os.system('struphy run pc_lin_mhd_6d_MF_full -i parameters_u1p0.yml')
    os.system('struphy run pc_lin_mhd_6d_MF_full -i parameters_u1p3.yml')
    os.system('struphy run pc_lin_mhd_6d_MF_full -i parameters_u2p0.yml')
    os.system('struphy run pc_lin_mhd_6d_MF_full -i parameters_u2p3.yml')

def test_pc_lin_mhd_6d_MF_perp():
    import os
    
    os.system('struphy run pc_lin_mhd_6d_MF_perp -i parameters_u1p0.yml')
    os.system('struphy run pc_lin_mhd_6d_MF_perp -i parameters_u1p3.yml')
    os.system('struphy run pc_lin_mhd_6d_MF_perp -i parameters_u2p0.yml')
    os.system('struphy run pc_lin_mhd_6d_MF_perp -i parameters_u2p3.yml')


if __name__ == '__main__':
    test_lin_mhd()
    test_cc_lin_mhd_6d_MF()
    test_pc_lin_mhd_6d_MF_full()
    test_pc_lin_mhd_6d_MF_perp()
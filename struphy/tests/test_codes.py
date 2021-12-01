def test_codes():
    import os

    os.system('struphy -r lin_mhd_MF -i parameters_u1p0.yml')
    os.system('struphy -r lin_mhd_MF -i parameters_u1p3.yml')
    os.system('struphy -r lin_mhd_MF -i parameters_u2p0.yml')
    os.system('struphy -r lin_mhd_MF -i parameters_u2p3.yml')
    
    os.system('struphy -r cc_lin_mhd_6d_MF -i parameters_u1p0.yml')
    os.system('struphy -r cc_lin_mhd_6d_MF -i parameters_u1p3.yml')
    os.system('struphy -r cc_lin_mhd_6d_MF -i parameters_u2p0.yml')
    os.system('struphy -r cc_lin_mhd_6d_MF -i parameters_u2p3.yml')

if __name__ == '__main__':
    test_codes()
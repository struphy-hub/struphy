# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. 
# Start the docker engine and run "docker login" with the current token from https://struphy.pages.mpcdf.de/struphy/sections/install.html#user-install, then:
#
# docker info
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_fedora_latest -f docker/fedora.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/struphy_fedora_latest

FROM fedora:latest

# install linux packages 
RUN dnf install -y wget yum-utils make openssl-devel bzip2-devel libffi-devel zlib-devel \
    && dnf update -y \
    && dnf install -y gcc \ 
    && dnf install -y gfortran \ 
    && dnf install -y blas-devel lapack-devel \ 
    && dnf install -y openmpi openmpi-devel \
    && dnf install -y libgomp \
    && dnf install -y git \
    && dnf install -y environment-modules \
    && dnf install -y python3-mpi4py-openmpi \
    && dnf install -y sqlite-devel \
    && dnf install -y pandoc \
    && wget https://www.python.org/ftp/python/3.12.8/Python-3.12.8.tgz \
    && tar xzf Python-3.12.8.tgz \
    && cd Python-3.12.8 \
    && ./configure --with-system-ffi --with-computed-gotos --enable-loadable-sqlite-extensions \
    && make -j ${nproc} \
    && make altinstall \
    && bash -c "echo 'alias python3=python3.12' >> ~/.bashrc" \
    && bash -c "source ~/.bashrc" \
    && mv /usr/local/lib/libpython3.12.a libpython3.12.a.bak \
    && echo "Done"

RUN alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 \
    && alternatives --set python3 /usr/local/bin/python3.12 \
    && python3 -m ensurepip \
    && python3 -V

    # create new working dir
WORKDIR /struphy_install/

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self

CMD ["python3"]
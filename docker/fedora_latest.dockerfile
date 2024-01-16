# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. Start the docker engine and then run:
#
# docker info
# docker login gitlab-registry.mpcdf.mpg.de -u docker_api -p glpat--z6kJtobeG-xM_LdL6k6
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/fedora -f docker/fedora.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/fedora

FROM fedora:latest

# install linux packages 
RUN dnf install -y python38 \
    && dnf install -y gcc \
    && dnf install -y gfortran \ 
    && dnf install -y blas-devel lapack-devel \ 
    && dnf install -y openmpi openmpi-devel \
    && dnf install -y libgomp \
    && dnf install -y git \
    && dnf install -y environment-modules \
    && dnf install -y python3-mpi4py-openmpi \
    && dnf install -y python3-devel \
    && dnf install -y pandoc \
    && dnf install -y sqlite

# create new working dir
WORKDIR /fedora_latest_py_3.8/

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self

RUN alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

RUN python3.8 -m ensurepip
RUN python3.8 -m pip install --upgrade pip
RUN bash -c ". /etc/profile.d/modules.sh && module load mpi/openmpi-$(arch) && module list && python3.8 -m pip install sympy==1.5 gvec_to_python" 

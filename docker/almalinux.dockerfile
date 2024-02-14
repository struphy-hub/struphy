# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. Start the docker engine and then run:
#
# docker info
# docker login gitlab-registry.mpcdf.mpg.de -u docker_api -p glpat--z6kJtobeG-xM_LdL6k6
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/almalinux -f docker/almalinux.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/almalinux

FROM almalinux:latest

# install linux packages
RUN yum update -y && yum clean all \
    && yum install -y gcc \
    && yum install -y gfortran \ 
    && yum install -y openmpi openmpi-devel \
    && export PATH=$PATH:/usr/lib64/openmpi/bin \
    && yum install -y libgomp \
    && yum install -y git \
    && yum install -y environment-modules \
    && yum install -y python3-devel \
    && yum install -y sqlite

# create new working dir
WORKDIR /almalinux_latest/

# allow mpirun as root
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
ENV OMPI_MCA_pml=ob1
ENV OMPI_MCA_btl=tcp,self

ENV PATH="/usr/lib64/openmpi/bin:$PATH"
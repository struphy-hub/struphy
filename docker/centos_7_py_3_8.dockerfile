# Here is how to build the image and upload it to the mpcdf gitlab registry:
#
# We suppose you are in the struphy repo directory. Start the docker engine and then run:
#
# docker info
# docker login gitlab-registry.mpcdf.mpg.de -u docker_api -p glpat--z6kJtobeG-xM_LdL6k6
# docker build -t gitlab-registry.mpcdf.mpg.de/struphy/struphy/centos_7_py_3_8 -f docker/centos_7_py_3_8.dockerfile .
# docker push gitlab-registry.mpcdf.mpg.de/struphy/struphy/centos_7_py_3_8

FROM centos:7

RUN yum update -y \
    && yum install -y centos-release-scl \
    && yum install -y devtoolset-8-gcc devtoolset-8-gcc-c++ devtoolset-8-gcc-gfortran \
    && yum install -y lapack-devel openmpi-devel blas-devel libgomp \
    && yum install -y git pandoc \
    && yum groupinstall -y "Development Tools" \
    && yum install -y wget openssl-devel bzip2-devel libffi-devel libxml2-devel libxslt-devel \
    && yum clean all

RUN yum install -y sqlite-devel

# Enable new version of GCC from SCL
SHELL ["/usr/bin/scl", "enable", "devtoolset-8"]

# Install OpenSSL
RUN cd /usr/local/src \
    && curl -O https://www.openssl.org/source/openssl-1.1.1k.tar.gz \
    && tar -xzf openssl-1.1.1k.tar.gz \
    && cd openssl-1.1.1k \
    && ./config --prefix=/usr/local/ssl --openssldir=/usr/local/ssl shared zlib \
    && make \
    && make install

# Configure environment for OpenSSL
RUN echo "/usr/local/ssl/lib" > /etc/ld.so.conf.d/openssl-1.1.1k.conf \
    && ldconfig \
    && echo 'export PATH=/usr/local/ssl/bin:$PATH' >> /etc/profile

# Install Python 3.8
RUN cd /usr/local/src \
    && wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tar.xz \
    && tar -xf Python-3.8.0.tar.xz \
    && cd Python-3.8.0 \
    && ./configure --enable-shared CFLAGS="-fPIC" --with-openssl=/usr/local/ssl --enable-loadable-sqlite-extensions \
    && make \
    && make install

# Update the shared library cache to include the newly installed shared libraries
RUN echo '/usr/local/lib' > /etc/ld.so.conf.d/python3.8.conf && \
    ldconfig

# Link Python3 to make it default
RUN ln -s /usr/local/bin/python3.8 /usr/bin/python3 \
    && ln -s /usr/local/bin/pip3.8 /usr/bin/pip3

RUN python3.8 -m pip install pip --upgrade

ENV PATH="/usr/lib64/openmpi/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH}"
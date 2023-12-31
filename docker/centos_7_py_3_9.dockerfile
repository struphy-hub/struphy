# Use CentOS 7 as the base image
FROM centos:7

# Install necessary packages and development tools
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

# Install Python 3.9
RUN cd /usr/local/src \
    && wget https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tar.xz \
    && tar -xf Python-3.9.0.tar.xz \
    && cd Python-3.9.0 \
    && ./configure --enable-shared CFLAGS="-fPIC" --with-openssl=/usr/local/ssl --enable-loadable-sqlite-extensions \
    && make \
    && make install

# Update the shared library cache to include the newly installed shared libraries
RUN echo '/usr/local/lib' > /etc/ld.so.conf.d/python3.9.conf && \
    ldconfig

# Link Python3 to make it default
RUN ln -s /usr/local/bin/python3.9 /usr/bin/python3 \
    && ln -s /usr/local/bin/pip3.9 /usr/bin/pip3

# Upgrade pip
RUN python3.9 -m pip install pip --upgrade

# Install OpenMPI development package (this is already installed in an earlier step, so this line is redundant)
# RUN yum install -y openmpi-devel

# Set environment variables for OpenMPI
ENV PATH="/usr/lib64/openmpi/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH}"

# Install struphy
RUN python3.9 -m pip install struphy

RUN scl enable devtoolset-8 'struphy compile'

# Create a new user and change ownership
RUN adduser centos_7_py_3_9_user \
    && chown -R centos_7_py_3_9_user:centos_7_py_3_9_user /usr/local/lib/python3.9/site-packages/struphy/

# Switch to non-root user
USER centos_7_py_3_9_user

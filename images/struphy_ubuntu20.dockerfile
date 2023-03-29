FROM gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu20_test

# install struphy and remove source code
COPY . .
RUN pip install . \
    && cd .. \
    && rm -rf struphy \
    && struphy compile 



FROM gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu20_test

# install struphy in dev mode
COPY . .
RUN pip install . \
    && cd .. \
    && rm -rf struphy \
    && struphy compile 



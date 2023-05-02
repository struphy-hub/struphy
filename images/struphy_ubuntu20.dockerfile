FROM gitlab-registry.mpcdf.mpg.de/struphy/struphy/ubuntu20_test

# install struphy and remove source code
COPY . .

RUN pip install . \
    && pip show struphy \
    && pip list \
    && struphy \
    && struphy compile \
    && cd .. \
    && rm -rf struphy  



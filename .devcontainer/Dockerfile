# Container image that runs your code
FROM python:3.13.0a1-alpine

# Sphinx
RUN pip3 install --no-cache-dir \
        sphinx \
        docutils \
        sphinx_rtd_theme \
        breathe \
        m2r2

# Tex
RUN apk update && \
    apk add --no-cache \
        git \
        cmake \
        doxygen \
        graphviz \
        ttf-freefont

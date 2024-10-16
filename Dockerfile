FROM ubuntu:20.04

RUN apt-get update && apt-get install -y gnupg2 software-properties-common wget

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 \
    && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/"

RUN apt-get update \
    && apt-get install -y --no-install-recommends r-base-dev python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/sharkdp/hyperfine/releases/download/v1.16.1/hyperfine_1.16.1_amd64.deb \
    && dpkg -i hyperfine_1.16.1_amd64.deb

ADD install.sh install.sh
RUN . ./install.sh

ADD benchmark.sh benchmark.sh
ADD R/ R/
ADD python/ python/
ADD bin/ bin/
CMD ["/bin/bash", "benchmark.sh"]

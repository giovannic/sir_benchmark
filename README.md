# Some SIR ABM benchmarks

## Usage

With docker

```bash
docker build -t benchmark .
docker run -it benchmark
```

or with different population counts, first arg, and batch size, second arg

```bash
docker run -it benchmark /bin/bash ./benchmark.sh 10,100,1000 1000"
```

## TODO

 - [] Review implementations for correctness
 - [] Sweep through batch sizes
 - [] JAX GPU version
 - [] Competing hazards versions
 - [] Household and network SIR versions
 - [] Agents.jl version
 - [] Documentation

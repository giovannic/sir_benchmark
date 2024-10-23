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

To run plots to validate model outputs:

```bash
docker run -it benchmark /bin/bash ./plot.sh"
```

## TODO

 - [x] Review implementations for correctness
 - [] Household and network SIR versions
 - [] Competing hazards versions
 - [] Agents.jl version
 - [] JAX GPU version
 - [] Sweep through batch sizes
 - [] Documentation

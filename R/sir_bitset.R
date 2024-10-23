args = commandArgs(trailingOnly=TRUE)

# Read CLI arguments
N <- strtoi(args[1])
parameters <- args[2]
output_file <- args[3]

# Read parameters from file
parameters <- read.csv(parameters, header = FALSE, col.names=c("I0","R0","gamma"))

dt <- 0.1
tmax <- 100
steps <- tmax/dt
aggregate_health_states <- matrix(0, nrow = steps * nrow(parameters), ncol = 5)

# Loop over each run
for (i in 1:nrow(parameters)) {
    # Initialise model arguments
    I0 <- round(parameters[[i,'I0']] * N)
    R0 <- parameters[[i,'R0']]
    S0 <- N - I0
    gamma <- parameters[i,]$gamma
    beta <- R0 * gamma
    S = individual::Bitset$new(N)$not()
    I = individual::Bitset$new(N)$insert(sample.int(n = N,size = I0))
    R = individual::Bitset$new(N)
    S$and(I$copy()$not())

	# Loop over each timestep
	for (t in 1:steps) {
        # Save aggregate health states
        aggregate_health_states[(i-1)*steps + t,] <- c(i,t,S$size(),I$size(),R$size())

        # calculate the force of infection
        foi <- I$size() * beta / N

        new_infections = S$copy()$sample(min(foi * dt, 1))
        I$or(new_infections)
        S$and(new_infections$not())

        new_recoveries = I$copy()$sample(gamma * dt)
        R$or(new_recoveries)
        I$and(new_recoveries$not())
	}
}

# Write aggregate health states to file
write.table(
    aggregate_health_states,
    file = output_file,
    sep = ",",
    row.names = FALSE,
    col.names = TRUE
)
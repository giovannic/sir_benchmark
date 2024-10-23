args <- commandArgs(trailingOnly = TRUE)

# Read CLI arguments
N <- strtoi(args[1])
parameters <- args[2]
output_file <- args[3]

# Read parameters from file
parameters <- read.csv(parameters, header = FALSE, col.names = c("I0", "R0", "gamma"))

dt <- 0.1
tmax <- 100
steps <- tmax / dt
aggregate_health_states <- matrix(0, nrow = steps * nrow(parameters), ncol = 5)

# Loop over each run
for (i in 1:nrow(parameters)) {
    # Initialise model arguments
    I0 <- round(parameters[[i, "I0"]] * N)
    R0 <- parameters[[i, "R0"]]
    S0 <- N - I0
    gamma <- parameters[i, ]$gamma
    beta <- R0 * gamma
    health_states <- c("S", "I", "R")
    health_states_t <- rep("S", N)
    health_states_t[sample.int(n = N, size = I0)] <- "I"

    # Loop over each timestep
    for (t in 1:steps) {
        # Save aggregate health states
        S <- sum(health_states_t == "S")
        I <- sum(health_states_t == "I")
        R <- sum(health_states_t == "R")
        aggregate_health_states[(i - 1) * steps + t, ] <- c(i, t, S, I, R)

        # calculate the force of infection
        foi <- sum(health_states_t == "I") * beta / N

        # Count susceptible
        S <- health_states_t == "S"
        n_S <- sum(S)
        # Select individuals to infect
        to_infect <- which(S)[runif(n = n_S) < foi * dt]

        # Count infected
        I <- health_states_t == "I"
        n_I <- sum(I)
        # Select individuals to recover
        to_recover <- which(I)[runif(n = n_I) < gamma * dt]

        # Apply changes
        health_states_t[to_infect] <- "I"
        health_states_t[to_recover] <- "R"
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
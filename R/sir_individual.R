args = commandArgs(trailingOnly=TRUE)

library(individual)

# Read CLI arguments
N <- strtoi(args[1])
parameters <- args[2]
output_file <- args[3]

# Read parameters from file
parameters <- read.csv(parameters, header = FALSE, col.names=c("I0","R0","gamma"))

# Make environment for model parameters
p <- new.env()

# Declare static arguments
dt <- 0.1
tmax <- 100
steps <- tmax/dt
health_states <- c("S","I","R")
output <- NULL

infection_process <- function(t){
    I <- health$get_size_of("I")
    foi <- p$beta * I/N
    S <- health$get_index_of("S")
    S$sample(rate = pexp(q = foi * dt))
    health$queue_update(value = "I", index = S)
}

recovery_process <- function(t){
    I <- health$get_index_of("I")
    I$sample(rate = pexp(q = p$gamma * dt))
    health$queue_update(value = "R", index = I)
}

# Run simulation for every parameter set
for (i in 1:nrow(parameters)) {
    # Initialise model arguments from parameter set
    p$I0 <- round(parameters[[i,'I0']] * N)
    
    p$gamma <- parameters[[i,'gamma']]
    p$R0 <- round(parameters[[i,'R0']] * N)
    p$S0 <- N - p$I0 - p$R0
    p$beta <- p$R0 * p$gamma
    health_states_t0 <- rep("S",N)
    health_states_t0[sample.int(n = N,size = p$I0)] <- "I"
    
    health <- CategoricalVariable$new(
      categories = health_states,
      initial_values = health_states_t0
    )
    health_render <- Render$new(timesteps = steps)
    health_render_process <- categorical_count_renderer_process(
      renderer = health_render,
      variable = health,
      categories = health_states
    )
    simulation_loop(
      variables = list(health),
      processes = list(infection_process,recovery_process,health_render_process),
      timesteps = steps
    )
    run_output <- health_render$to_dataframe()
    run_output$run <- i
    output <- rbind(output,run_output)
}

# Write health dataframe to file
write.table(
    output,
    file = output_file,
    sep = ",",
    row.names = FALSE,
    col.names = TRUE
)
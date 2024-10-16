// SIR individual based model in C++

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>

enum class State {S, I, R};

int main(int argc, char *argv[]) {
	// Read in command line arguments
	// N as an integer
	// parameters is the path to a file with the parameters
	// output is the path to the output file
	int N = atoi(argv[1]);
	std::string parameters = argv[2];
	std::string output = argv[3];

	// Read in parameters
	std::ifstream param_file(parameters);

	// Consume header
	std::string line;
	std::getline(param_file, line);

	// Open output file
	std::ofstream output_file(output);

	// While there are parameters left in the file
	auto run = 0u;
	while(std::getline(param_file, line)) {
		// Read in parameters
		std::istringstream iss(line);
		float I0, R0, gamma;
		iss >> I0 >> R0 >> gamma;
		float beta = R0 * gamma;

		// Initialize the population
		auto population = std::vector<State>(N, State::S);
		for (int i = 0; i < I0; i++) {
			population[i] = State::I;
		}
		std::default_random_engine generator(42);

		// Run the simulation
		for (auto t = 0u; t < 100; ++t) {
			// Calculate foi
			auto foi = 0u;
			for (const auto p : population) {
				if (p == State::I) {
					foi += 1;
				}
			}
			foi = foi * beta / N;

			auto to_infect = std::vector<unsigned int>();
			auto to_recover = std::vector<unsigned int>();
			for (auto i = 0u; i < N; ++i) {
				// Sample population to infect
				if (population[i] == State::S) {
					std::binomial_distribution<int> distribution(1, foi);
					if (distribution(generator) == 1) {
						to_infect.push_back(i);
					}
				}

				// Sample population to recover
				if (population[i] == State::I) {
					std::binomial_distribution<int> distribution(1, gamma);
					if (distribution(generator) == 1) {
						to_recover.push_back(i);
					}
				}
			}

			// Apply changes
			for (const auto i : to_infect) {
				population[i] = State::I;
			}
			for (const auto i : to_recover) {
				population[i] = State::R;
			}

			// Count the number of each state
			auto S = 0u;
			auto I = 0u;
			auto R = 0u;
			for (const auto p : population) {
				if (p == State::S) {
					S += 1;
				} else if (p == State::I) {
					I += 1;
				} else {
					R += 1;
				}
			}

			// Write out the state of the population
			output_file << run << "," << t << "," << S << "," << I << "," << R << std::endl;

		}
	}
}

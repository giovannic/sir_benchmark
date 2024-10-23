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

	// Open output file
	std::ofstream output_file(output);

	// While there are parameters left in the file
	auto run = 0u;

	output_file << "r,t,S,I,R" << std::endl;
	while(std::getline(param_file, line)) {
		// Read in parameters
		std::istringstream iss(line);
		std::string token;
		float I0, R0, gamma;
		std::getline(iss, token, ','); I0 = std::stof(token);
		std::getline(iss, token, ','); R0 = std::stof(token);
		std::getline(iss, token, ','); gamma = std::stof(token);
		float beta = R0 * gamma;

		// Initialize the population
		auto population = std::vector<State>(N, State::S);
		for (int i = 0; i < std::floor(I0 * N); i++) {
			population[i] = State::I;
		}
		std::default_random_engine generator(42);

		auto dt = .1;
		auto tmax = 100.;

		// Run the simulation
		for (auto t = 0u; t < tmax / dt; ++t) {
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

			// Calculate foi
			auto foi = 0.;
			for (const auto p : population) {
				if (p == State::I) {
					foi += 1.;
				}
			}
			foi = foi * beta / N * dt;

			auto to_infect = std::vector<unsigned int>();
			auto to_recover = std::vector<unsigned int>();
			for (auto i = 0u; i < N; ++i) {
				// Sample population to infect
				if (population[i] == State::S) {
					std::uniform_real_distribution<float> distribution(0, 1);
					if (distribution(generator) < foi) {
						to_infect.push_back(i);
					}
				}

				// Sample population to recover
				if (population[i] == State::I) {
					std::uniform_real_distribution<float> distribution(0, 1);
					if (distribution(generator) < gamma * dt) {
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
		}
		++run;
	}
}

#include <iostream>

#include <memory>

#include <boost/program_options.hpp>
#include <clever/clever.hpp>

#include "EventProcessor.h"

namespace po = boost::program_options;

int main(int argc, char **argv)
{
	std::string inputFileName;
	// Declare the supported options.
	po::options_description desc("Allowed options");
	desc.add_options()("help", "produce help message")("input-file",
			po::value<std::string>(), "data input file")("max-events",
			po::value<int>(), "maximum number of events to process")(
			"parallel-events", po::value<int>(),
			"number of events processed in parallel")("gpu",
			"run the first available GPU")("cpu",
			"run the first available CPU (default)");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		std::cout << desc << std::endl;
		return 1;
	}

	if (vm.count("input-file"))
	{
		inputFileName = vm["input-file"].as<std::string>();
	}
	else
	{
		std::cout << "No input file set" << std::endl;
	}

	std::cout << "Input file was set to " << vm["input-file"].as<std::string>()
			<< std::endl;

	int maxEvents = -1;
	if (vm.count("max-events"))
	{
		maxEvents = vm["max-events"].as<int>();
	}

	int parEvents = 1;
	if (vm.count("parallel-events"))
	{
		parEvents = vm["parallel-events"].as<int>();
	}

	std::unique_ptr<clever::context> clContext;
	if (vm.count("gpu"))
	{
		clContext.reset(
				new clever::context(clever::context_settings::default_gpu()));
		std::cout << "Running on default GPU" << std::endl;
	}
	else
	{
		clContext.reset(
				new clever::context(clever::context_settings::default_cpu()));
		std::cout << "Running on default CPU" << std::endl;
	}

	EventProcessor proc(inputFileName, clContext.get(), maxEvents, parEvents);
	proc.run();

	return 0;
}

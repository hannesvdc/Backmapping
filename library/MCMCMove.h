#ifndef _MCMCMOVE_H_
#define _MCMCMOVE_H_

#include <vector>

class MCMCMove {
public:
	virtual ~MCMCMove() {}

	virtual std::vector<OpenMM::Vec3> step() = 0;	
};

class MCMCSampler {
public:
	MCMCSampler(MCMCMove &_move) : move(_move) {}

	std::vector<std::vector<OpenMM::Vec3> > sample(int Nsamples ) {
		std::vector<std::vector<OpenMM::Vec3> > samples;

		for( int i = 0; i < Nsamples; ++i) {
			std::vector<OpenMM::Vec3> x = move.step();
			samples.push_back(x);
		}

		return samples;
	}
private:
	MCMCMove& move;
};

#endif

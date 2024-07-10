#include "OpenMM.h"

#include <string>
#include <map>
#include <fstream>
#include <iostream>

class MARTINI : public OpenMM::ReactionCoordinate {
public:
	MARTINI(std::vector<std::string> _amino_acids) :
			amino_acids(_amino_acids),
			rc_len(0),
			micro_len(0) {

		// Initialize all maps
		createAminoAcidCGSizeMap();
		createAmnioAcidMicroSizeMap();
		createAminoAcidCGIndexMap();
		createAminoAcidCGMassMap();

		// Compute the size of the macroscopic molecule.
		for ( int i = 0; i < amino_acids.size(); ++i) {
			rc_len += amino_cg_sizes[amino_acids[i]];
			micro_len += amino_micro_sizes[amino_acids[i]];
		}

		std::cout << "jacobian size " << rc_len << "x" << micro_len << std::endl;

		computeJacobian();
	}

	virtual std::vector<OpenMM::Vec3> value(const std::vector<OpenMM::Vec3>& x) {
		std::vector<OpenMM::Vec3> pos(rc_len, OpenMM::Vec3(0., 0., 0.));

		int current_micro_index = 0;
		int current_cg_index = 0;
		for( int i = 0; i < amino_acids.size(); ++i ) {
			std::string amino = amino_acids[i];
			int len = amino_cg_sizes[amino];
			
			for( int j = 0; j < len; ++j ) {
				std::vector<int> indices = amino_cg_indices[amino][j];
				std::vector<double> masses = amino_cg_masses[amino][j];

				for( int k = 0; k < indices.size(); ++k) {
					pos[current_cg_index] += x[current_micro_index + indices[k]]*masses[k];
				}
				pos[current_cg_index] /= amino_cg_total_masses[amino][j];

				current_cg_index++;
			}

			current_micro_index += amino_micro_sizes[amino];
		}

		return pos;
	}

	virtual std::vector<OpenMM::Vec3> gradMatMul(const std::vector<OpenMM::Vec3>& x, const std::vector<OpenMM::Vec3>& z) {
		std::vector<OpenMM::Vec3> y(micro_len, OpenMM::Vec3(0., 0., 0.));
		for( int i = 0; i < y.size(); ++i ) {
			OpenMM::Vec3 vec(0., 0., 0.);
			for ( int j = 0; j < z.size(); ++j ) {
				vec += jacobian[j][i]*z[j];
			}
			y[i] = vec;
		}

		return y;
	}

	virtual std::vector<OpenMM::Vec3> partialGradMatMul(const OpenMM::Vec3 &zp, int k) {
		std::vector<OpenMM::Vec3> z(rc_len, OpenMM::Vec3(0.0, 0.0, 0.0)); z[k] = zp;
		std::vector<OpenMM::Vec3> y(micro_len, OpenMM::Vec3(0.0, 0.0, 0.0));

		for( int i = 0; i < y.size(); ++i ) {
			OpenMM::Vec3 vec(0., 0., 0.);
			for ( int j = 0; j < z.size(); ++j ) {
				vec += jacobian[j][i]*z[j];
			}
			y[i] = vec;
		}

		return y;
	}

	virtual double getBiasedEnergy(const std::vector<OpenMM::Vec3> &x,
                                   const std::vector<OpenMM::Vec3> &z) {
		std::vector<OpenMM::Vec3> xi = this->value(x);

		return this->norm_int(xi, z);
	}

	std::vector<std::vector<double> > getJacobian() const {
		return jacobian;
	}

private:
	int rc_len, micro_len;
	std::vector<std::string> amino_acids;
	std::map<std::string, int> amino_cg_sizes;
	std::map<std::string, int> amino_micro_sizes;
	std::map<std::string, std::vector<std::vector<int> > > amino_cg_indices;
	std::map<std::string, std::vector<std::vector<double> > > amino_cg_masses;
	std::map<std::string, std::vector<double> > amino_cg_total_masses;

	std::vector<std::vector<double> > jacobian;

	double norm_int(const std::vector<OpenMM::Vec3> &xi, const std::vector<OpenMM::Vec3> &z) const {
		double n = 0.0;
		for( int i = 0; i < xi.size(); ++i ) {
			n += (xi[i][0]-z[i][0])*(xi[i][0]-z[i][0]) + (xi[i][1]-z[i][1])*(xi[i][1]-z[i][1]) + (xi[i][2]-z[i][2])*(xi[i][2]-z[i][2]);
		}

		return n;
	}

	void createAminoAcidCGSizeMap() {
		amino_cg_sizes.insert(std::pair<std::string, int>("GLY", 1));
		amino_cg_sizes.insert(std::pair<std::string, int>("ALA", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("SER", 2));
		amino_cg_sizes.insert(std::pair<std::string, int>("CYS", 2));
		amino_cg_sizes.insert(std::pair<std::string, int>("THR", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("VAL", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("LEU", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("IIE", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("MET", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("PRO", 2));
		amino_cg_sizes.insert(std::pair<std::string, int>("APS", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("GLU", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("PHE", 4));
		amino_cg_sizes.insert(std::pair<std::string, int>("TYR", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("TRP", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("ASN", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("GLN", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("LYS", 0));
		amino_cg_sizes.insert(std::pair<std::string, int>("ARG", 3));
	}

	void createAmnioAcidMicroSizeMap() {
		amino_micro_sizes.insert(std::pair<std::string, int>("GLY", 9));
		amino_micro_sizes.insert(std::pair<std::string, int>("ALA", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("SER", 11));
		amino_micro_sizes.insert(std::pair<std::string, int>("CYS", 10));
		amino_micro_sizes.insert(std::pair<std::string, int>("THR", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("VAL", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("LEU", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("IIE", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("MET", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("PRO", 14));
		amino_micro_sizes.insert(std::pair<std::string, int>("APS", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("GLU", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("PHE", 20));
		amino_micro_sizes.insert(std::pair<std::string, int>("TYR", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("TRP", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("ASN", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("GLN", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("LYS", 0));
		amino_micro_sizes.insert(std::pair<std::string, int>("ARG", 24));
	}

	void createAminoAcidCGIndexMap() {
		std::vector<std::vector<int> > GLY;
		GLY.push_back({0,1,2,3,4,5,6});
		std::vector<std::vector<int> > PHE;
		PHE.push_back({0,1,2,3,11}); PHE.push_back({4,5,6,15}); PHE.push_back({7, 9, 16, 18}); PHE.push_back({8, 10, 17, 19});
		std::vector<std::vector<int> > ARG;
		ARG.push_back({0,1,2,3,11}); ARG.push_back({4,5,6}); ARG.push_back({7, 8, 9, 10, 21, 22});
		std::vector<std::vector<int> > SER;
		SER.push_back({0,1,2,3,6}); SER.push_back({4,5,10});
		std::vector<std::vector<int> > PRO;
		PRO.push_back({0,1,2,3}); PRO.push_back({4,5,6});
		std::vector<std::vector<int> > CYS;
		CYS.push_back({0,1,2,3,6}); CYS.push_back({4,5});

		amino_cg_indices.insert(std::pair<std::string, std::vector<std::vector<int> > >("GLY", GLY));
        amino_cg_indices.insert(std::pair<std::string, std::vector<std::vector<int> > >("PHE", PHE));
        amino_cg_indices.insert(std::pair<std::string, std::vector<std::vector<int> > >("ARG", ARG));
        amino_cg_indices.insert(std::pair<std::string, std::vector<std::vector<int> > >("SER", SER));
        amino_cg_indices.insert(std::pair<std::string, std::vector<std::vector<int> > >("PRO", PRO));
        amino_cg_indices.insert(std::pair<std::string, std::vector<std::vector<int> > >("CYS", CYS));
	}

	void createAminoAcidCGMassMap() {
		double H = 1.00784;
		double C = 12.0107;
        double N = 14.0067;
        double O = 15.999;
        double S = 32.065;

		std::vector<std::vector<double> > GLY;
		GLY.push_back({N, C, C, O, H, H, H});
		std::vector<std::vector<double> > PHE;
		PHE.push_back({N, C, C, O, H}); PHE.push_back({C, C, C, H}); PHE.push_back({C, C, H, H}); PHE.push_back({C, C, H, H});
		std::vector<std::vector<double> > ARG;
		ARG.push_back({N, C, C, O, H}); ARG.push_back({C, C, C}); ARG.push_back({N, C, N, H, H});
		std::vector<std::vector<double> > SER;
		SER.push_back({N, C, C, O, H}); SER.push_back({C, O, H});
		std::vector<std::vector<double> > PRO;
		PRO.push_back({N, C, C, O}); PRO.push_back({C, C, C});
		std::vector<std::vector<double> > CYS;
		CYS.push_back({N, C, C, O, H}); CYS.push_back({C, S});

		amino_cg_masses.insert(std::pair<std::string, std::vector<std::vector<double> > >("GLY", GLY));
        amino_cg_masses.insert(std::pair<std::string, std::vector<std::vector<double> > >("PHE", PHE));
        amino_cg_masses.insert(std::pair<std::string, std::vector<std::vector<double> > >("ARG", ARG));
        amino_cg_masses.insert(std::pair<std::string, std::vector<std::vector<double> > >("SER", SER));
        amino_cg_masses.insert(std::pair<std::string, std::vector<std::vector<double> > >("PRO", PRO));
        amino_cg_masses.insert(std::pair<std::string, std::vector<std::vector<double> > >("CYS", CYS));

        amino_cg_total_masses.insert(std::pair<std::string, std::vector<double> >("GLY", {N+2*C+O+3*H}));
        amino_cg_total_masses.insert(std::pair<std::string, std::vector<double> >("PHE", {N+2*C+O+H, 3*C+H, 2*C+2*H, 2*C+2*H}));
        amino_cg_total_masses.insert(std::pair<std::string, std::vector<double> >("ARG", {N+2*C+O+H, 3*C, 2*N+C+2*H}));
        amino_cg_total_masses.insert(std::pair<std::string, std::vector<double> >("SER", {N+2*C+O+H, C+O+H}));
        amino_cg_total_masses.insert(std::pair<std::string, std::vector<double> >("PRO", {N+2*C+O, 3*C}));
        amino_cg_total_masses.insert(std::pair<std::string, std::vector<double> >("CYS", {N+2*C+O+H, C+S}));
	}

	void computeJacobian() {
		jacobian.resize(rc_len, std::vector<double>(micro_len, 0.));

		int micro_index = 0;
		int cg_index = 0;
		double norm_sq = 0.;
		for ( int i = 0; i < amino_acids.size(); ++i ) {
			std::string amino = amino_acids[i];
			int dim = amino_cg_sizes[amino];
            
            for ( int j = 0; j < dim; ++j ) {
            	std::vector<int> indices = amino_cg_indices[amino][j];
            	std::vector<double> masses = amino_cg_masses[amino][j];
            	for ( int k = 0; k < indices.size(); ++k) {
            		jacobian[cg_index][micro_index + indices[k]] = masses[k]/amino_cg_total_masses[amino][j];
            		norm_sq += jacobian[cg_index][micro_index + indices[k]]*jacobian[cg_index][micro_index + indices[k]];
            	}

            	cg_index++;
            }

            micro_index += amino_micro_sizes[amino];
		}

		//norm_sq = sqrt(norm_sq);
		//for ( int i = 0; i < jacobian.size(); ++i ) {
		//	for ( int j = 0; j < jacobian[i].size(); ++j)
		//		jacobian[i][j] /= norm_sq;
		//}
	}
};

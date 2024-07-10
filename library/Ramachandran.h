#include <vector>
#include <map>
#include <string>

#include "OpenMM.h"

class Ramachandran {
public:
	Ramachandran() {
		indices["GLY"] = std::vector<int>({ 4, 0, 1, 2, 0, 1, 2, 9});
		indices["PHE"] = std::vector<int>({-1, 0, 1, 2, 0, 1, 2, 20});
		indices["ARG"] = std::vector<int>({-1, 0, 1, 2, 0, 1, 2, 24});
		indices["SER"] = std::vector<int>({-1, 0, 1, 2, 0, 1, 2, 11});
		indices["PRO"] = std::vector<int>({-1, 0, 1, 2, 0, 1, 2, 14});
		indices["CYS"] = std::vector<int>({-1, 0, 1, 2, 0, 1, 2, 10});
		offsets["GLY"] = 9;
		offsets["PHE"] = 20;
		offsets["ARG"] = 24;
		offsets["SER"] = 11;
		offsets["PRO"] = 14;
		offsets["CYS"] = 10;
	}

	std::vector<double> computeDihedrals(const std::vector<OpenMM::Vec3>& x, const std::vector<std::string>& amino_acids) const {
		std::vector<double> ram;

		int total_offset = 0;
		int prev_C = 4;
		for ( int i = 0; i < amino_acids.size(); ++i ) {
			std::vector<int> ind = indices.at(amino_acids[i]);

			double phi = torsion(prev_C, total_offset+ind[1], total_offset+ind[2], total_offset+ind[3], x);
			double psi = torsion(total_offset+ind[4], total_offset+ind[5], total_offset+ind[6], total_offset+ind[7], x);
			ram.push_back(phi); ram.push_back(psi);

			prev_C = total_offset + 2;
			total_offset += offsets.at(amino_acids[i]);
		}

		return ram;
	}

private:
	std::map<std::string, int> offsets;
	std::map<std::string, std::vector<int> > indices;

	double torsion(int A, int B, int C, int D, const std::vector<OpenMM::Vec3>& x ) const {
		OpenMM::Vec3 b1 = x[A] - x[B];
        OpenMM::Vec3 b2 = x[C] - x[B];
        OpenMM::Vec3 b3 = x[D] - x[C];
        
        OpenMM::Vec3 n1 = b1.cross(b2);
        n1 = n1/sqrt(n1.dot(n1));
        OpenMM::Vec3 n2 = b2.cross(b3);
        n2 = n2/sqrt(n2.dot(n2));
        OpenMM::Vec3 m = n1.cross(b2/sqrt(b2.dot(b2)));
        
        double xp = n1.dot(n2);
        double yp = m.dot(n2);
        double phi = atan2((-1)*yp, (-1)*xp);
        
        return -phi;
	}
};
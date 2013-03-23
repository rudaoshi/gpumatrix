#ifndef TEST_UTIL_H
#define TEST_UTIL_H

	template <typename E1, typename E2>
	bool check_diff(const E1 & m1, const E2 & m2)
	{
		double error = (m1-(E1)m2).squaredNorm();
		
		if (error > 1e-5)
		{

			std::cout << error << std::endl;
		}

		return error < 1e-5;
	}


#endif
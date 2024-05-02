#ifndef RTP_UTILITY_H
#define RTP_UTILITY_H
#include "AuxPort.h"
namespace AuxPort
{
	template<class from, class to>
	class DataConverter
	{
	public:
		static void convert(const std::vector<from>& source, std::vector<to>& destination)
		{
			AuxAssert((source.size() * sizeof(from)) % sizeof(to) == 0, "Invalid Conversion: Source Vector's size should allow exact conversion!");
			size_t destinationSize = source.size() * sizeof(from) / sizeof(to);
			destination.resize(destinationSize);
			to* ptr = (to*)source.data();
			destination.assign(ptr, ptr + destinationSize);
		}

		static std::vector<to> convert(const std::vector<from>& source)
		{
			std::vector<to> destination;
			convert(source, destination);
			return destination;
		}
	};
}


#endif

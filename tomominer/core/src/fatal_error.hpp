#ifndef TOMO_FATAL_ERROR_HPP
#define TOMO_FATAL_ERROR_HPP

#include <sstream>
#include <stdexcept>
#include <string>

/**
    A subclass of std::exception which allows for the use of << style
    construction of what() message.

    copied from: http://marknelson.us/2007/11/13/no-exceptions/
*/

class fatal_error : public std::exception
{

public:

	fatal_error() {};
	
    
    fatal_error(const fatal_error &that)
	{
		m_what += that.stream.str();
	}
	
    
    virtual ~fatal_error() throw(){};
	
    
    virtual const char *what() const throw()
	{
		if (stream.str().size()) 
        {
			m_what += stream.str();
			stream.str("");
		}
		return m_what.c_str();
	}
	
    
    template<typename T>
	fatal_error& operator<<(const T& t)
	{
		stream << t;
		return *this;
	 }


private:
    mutable std::stringstream stream;
    mutable std::string m_what;
};

#endif

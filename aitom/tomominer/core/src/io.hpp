#ifndef TOMO_IO_HPP
#define TOMO_IO_HPP

#include <armadillo>

/** 
    Output a real valued cube in MRC format.

    This is a very minimal writer.  It does not set many of the optional fields.

    @param c The cube to write to a file.
    @param filename the file to write c to.
*/
void write_mrc(const arma::cube &c, const char *filename);

/** 
    readr for MRC files.
    
    The MRC file format is defined: http://ami.scripps.edu/prtl_data/mrc_specification.htm
*/
arma::cube read_mrc(const char *filename);



#endif


#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "io.hpp"

void write_mrc(const arma::cube &c, const char *filename)
{
    char *zeros;
    float *data;

    std::ofstream fout(filename, std::ios::out | std::ios::binary);

    if(!fout)
    {
        std::stringstream msg;
        msg << "Unable to open file " << filename << " for writing." << std::endl;
        std::cerr << msg.str();
        throw std::ios_base::failure(msg.str());
    }

    // we are trying to read 4byte float values.
    if(sizeof(float) != 4)
    {
        std::stringstream msg;
        msg << "read_mrc: current version requires sizeof(float) == 4." << std::endl;
        std::cerr << msg.str();
        throw std::runtime_error(msg.str());
    }

    // header is 1024 bytes.  
    // 
    // first 12 bytes are 4byte x,y,z size values.
    int32_t dim[3];
    dim[0] = c.n_rows; dim[1] = c.n_cols; dim[2] = c.n_slices;

    // same values as 32-bit floats.
    float len[3];
    len[0] = (float)dim[0]; len[1] = (float)dim[1]; len[2] = (float)dim[2];

    unsigned int buf_len = 3 * sizeof(int32_t);

    // TODO: Do we need to wrap these fout.write calls in a try/catch block?
    fout.write(reinterpret_cast<char *>(dim), buf_len);

    // next 4 bytes determine the file type.
    int32_t mode = 2;
    fout.write( reinterpret_cast<char*>(&mode), sizeof(mode));

    // use value initialization to set all values to 0.
    // new char[N]() == calloc(N)
    zeros = new char[1024]();
    
    // write out nxstart, nystart, nzstart.  all 0 is ignored. 3 float32 values.
    fout.write( reinterpret_cast<char*>(zeros), 12);

    // write mx,my,mz
    // same as nx, ny, nz
    fout.write(reinterpret_cast<char*>(dim), 3*sizeof(int32_t));

    // write cella (size of image in angstroms)
    // float values.
    fout.write(reinterpret_cast<char*>(len), 3*sizeof(float));

    // write cell angles in degrees (0 or 90?).
    fout.write(reinterpret_cast<char *>(zeros), 3*sizeof(int32_t));

    // mapping dimension of columns, rows, slices to axes.
    int32_t map_crs[3];
    map_crs[0] = 1; map_crs[1] = 2; map_crs[2] = 3;
    fout.write(reinterpret_cast<char*>(map_crs), 3*sizeof(int32_t));

    float stats[3];

    stats[0] = float(c.min());
    stats[1] = float(c.max());
    stats[2] = float(arma::accu(c) / c.n_elem);

    // min value in file/max value/avg value.
    fout.write(reinterpret_cast<char*>(stats), 3*sizeof(float));

    // write out a bunch of junk.  29 zero bytes.
    fout.write(reinterpret_cast<char *>(zeros), 30*sizeof(int32_t));

    // write the word MAP.
    fout.write("MAP ", 4);

    // everything else is zeros.
    fout.write(reinterpret_cast<char *>(zeros), 812);

    buf_len = c.n_elem * sizeof(float);
    data = new float[buf_len];

    // move from double to float.
    for(size_t i = 0; i < c.n_elem; i++)
        data[i] = c(i);

    // write the data.
    fout.write(reinterpret_cast<char*>(data), buf_len);

    fout.close();
    delete[] data;
    delete[] zeros;
}


arma::cube read_mrc(const char *filename)
{
    // open the file.
    std::ifstream fin(filename, std::ios::in | std::ios::binary);

    if(!fin)
    {
        std::stringstream msg;
        msg << "Unable to open file " << filename << " for reading." << std::endl;
        std::cerr << msg.str();
        throw std::ios_base::failure(msg.str());
        //throw std::exception();
    }

    // The header is 1024 bytes.  The beginning of the header is 56 4-byte values.

    // the first three 4-byte values are integers, which give the array shape. (NX,NY,NZ)
    int32_t dim[3];
    fin.read(reinterpret_cast<char *>(dim), 3*sizeof(int32_t));

    if(fin.gcount() != 3*sizeof(int32_t) || fin.fail())
    {
        std::stringstream msg;
        msg << "read_mrc failed parsing: " << filename << ".  Failed to read first 3 int32_t values of file." << std::endl;
        std::cerr << msg.str();
        throw std::ios_base::failure(msg.str());
    }
    unsigned int n_elem = dim[0]*dim[1]*dim[2];

    // The next entry is the type of data stored in the file.
    int32_t mode;
    fin.read(reinterpret_cast<char*>(&mode), sizeof(mode));

    if(fin.gcount() != sizeof(int32_t) || fin.fail())
    {
        std::stringstream msg;
        msg << "read_mrc failed parsing: " << filename << ".  Failed to read mode." << std::endl;
        std::cerr << msg.str();
        throw std::ios_base::failure(msg.str());
    }
    switch(mode)
    {
        case 0: // unsigned bytes;
            break;
        case 1: // signed short int (16bit)
            break;
        case 2: // float (32bit)
            break;
        case 3: // complex short 2*(16bit)
            break;
        case 4: // complex float 2*(32bit)
            break;
    }

    // for now only deal with 32-bit float valued matrices.
    if(mode != 2)
    {
        std::stringstream msg;
        msg << "read_mrc failed parsing: " << filename << ".  Current version of software only handles MRC files with mode = 2. Given mode = " << mode << std::endl;
        std::cerr << msg.str();
        throw std::runtime_error(msg.str());
    }

    // skip the rest of the first 1024 bytes.  Data starts at 1024.
    fin.seekg(1024);

    // TODO: actually read the header and use it correctly.

    // we are trying to read 4byte float values.
    if(sizeof(float) != 4)
    {
        std::stringstream msg;
        msg << "read_mrc: current version requires sizeof(float) == 4." << std::endl;
        std::cerr << msg.str();
        throw std::runtime_error(msg.str());
    }

    unsigned int buf_len = sizeof(float) * n_elem;

    float *fvol = new float[buf_len];

    // read data into memory.
    fin.read(reinterpret_cast<char*>(fvol), buf_len);
    // check that we read enough.
    if(fin.gcount() != buf_len || fin.fail())
    {
        delete[] fvol;
        std::stringstream msg;
        msg << "read_mrc: Failed to read file: " << filename << std::endl;
        std::cerr << msg.str();
        throw std::ios_base::failure(msg.str());
    }

    fin.close();

    arma::cube vol(dim[0], dim[1], dim[2]);
    
    for(size_t i = 0; i < n_elem; i++)
        vol(i) = fvol[i];
    
    delete[] fvol;

    return vol;
}

arma::cube read_em(const char *filename)
{

    // open the file.
    std::ifstream fin(filename, std::ios::in | std::ios::binary);

    try
    {
        if(!fin)
        {
            std::cerr << "Unable to open file " << filename << " for reading." << std::endl;
            throw std::exception();
        }

        // first 4 bytes are magic values.
        char magic[4];

        fin.read(magic, 4);

        // First byte is machine coding.
        // Machine code:
        // 0 = OS-9, 1 = VAX, 2 = Convex, 3 = SGI, 4 = Sun, 5 = Mac, 6 = PC
        // This only matters because OS-9, Convex, and Mac use big-endian encoding.
        if(magic[0] == 0 || magic[0] == 3 || magic[0] == 5)
        {
            std::cerr << "EM files with big-endian encoding are not currently supported" << std::endl;
            throw std::exception();
        }

        // Second byte is ignored.
        // Third byte is ignored.
        // Fourth byte is data type: 1 = byte, 2 = int-16, 4 = int-32, 5 = float-32, 8 = complex-64 (float-32, float-32), 9 = float-64
        
        // for now only deal with 32-bit float valued matrices.
        if(magic[3] != 5)
        {
            std::cerr << "read_em: current version of software only handles EM files with mode = 5." << std::endl;
            std::cerr << "Given file has mode = " << magic[3] << std::endl;
            std::cerr << "1 = byte, 2 = int-16, 4 = int-32, 5 = float-32, 8 = complex-64 (total), 9 = float-64" << std::endl;
            throw std::exception();
        }

        // Read 3 32-bit integers as dimensions.
        int32_t dim[3];
        fin.read(reinterpret_cast<char *>(dim), 3*sizeof(int32_t));
        unsigned int n_elem = dim[0]*dim[1]*dim[2];

        // next 80 chars are comment
        char comment[80];
        fin.read(comment, 80);

        // next 40 32-bit ints are parameter values
        int32_t params[40];
        fin.read(reinterpret_cast<char *>(params), 40*sizeof(int32_t));

        // next 256 bytes are user data.
        char user_data[256];
        fin.read(user_data, 256);

        arma::cube vol(dim[0], dim[1], dim[2]);

        if(magic[3] == 5)
        {
            unsigned int buf_len = sizeof(float) * n_elem;
            float *fvol          = new float[buf_len];

            // read data into memory.
            fin.read(reinterpret_cast<char*>(fvol), buf_len);
            // check that we read enough.
            if(fin.gcount() != buf_len || fin.fail())
            {
                std::cerr << "read_em: Failed to read file." << std::endl;
                fin.close();
                throw std::exception();
            }
            
            for(size_t i = 0; i < n_elem; i++)
                vol(i) = fvol[i];
        
            delete[] fvol;
        }
        else if(magic[3] == 9)
        {
            unsigned int buf_len = sizeof(double) * n_elem;
            fin.read(reinterpret_cast<char*>(vol.memptr()), buf_len);

            if(fin.gcount() != buf_len || fin.fail())
            {
                std::cerr << "read_em: Failed to read file." << std::endl;
                fin.close();
                throw std::exception();
            }
        }
            
        if(fin.fail())
        {
            std::cerr << "read_em: Failed to read file." << std::endl;
            fin.close();
            throw std::exception();
        }
        fin.close();

        return vol;
    }
    catch (std::exception &ex)
    {
        fin.close();
        throw ex;
    }
}

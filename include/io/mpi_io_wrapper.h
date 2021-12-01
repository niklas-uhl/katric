#ifndef MPI_IO_WRAPPER_H
#define MPI_IO_WRAPPER_H
#include <mpi.h>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>
#include "util.h"

namespace cetric {

class ConcurrentFile {
public:
    enum class AccessMode { ReadOnly, WriteOnly, ReadAndWrite, Create };

private:
    static int mpi_amode(ConcurrentFile::AccessMode access_mode) {
        using AccessMode = ConcurrentFile::AccessMode;
        switch (access_mode) {
            case AccessMode::ReadOnly:
                return MPI_MODE_RDONLY;
            case AccessMode::WriteOnly:
                return MPI_MODE_WRONLY;
            case AccessMode::ReadAndWrite:
                return MPI_MODE_RDWR;
            case AccessMode::Create:
                return MPI_MODE_RDWR;
        }
        return 0;
    }

    static int mpi_amode(const std::vector<ConcurrentFile::AccessMode>& access_mode) {
        return std::accumulate(access_mode.begin(), access_mode.end(), 0, [](int acc, auto mode) {
            acc |= mpi_amode(mode);
            return acc;
        });
    }

public:
    template <class AmodeType>
    ConcurrentFile(const std::string& file_name, AmodeType access_mode, MPI_Comm comm) : handle(nullptr) {
        int err = MPI_File_open(comm, file_name.c_str(), mpi_amode(access_mode), MPI_INFO_NULL, &handle);
        check_mpi_error(err);
    }

    virtual ~ConcurrentFile() {
        if (handle != nullptr) {
            MPI_File_close(&handle);
            handle = nullptr;
        }
    }

    /**
     * @brief
     *
     * @tparam T the type of elements to read
     * @param buffer the read buffer (will be resized)
     * @param elements_to_read number of elements to read
     * @param position the file position to start from
     *
     * @return the actual number of read elements
     */
    template <typename T>
    size_t read(std::vector<T>& buffer, size_t elements_to_read, size_t position = 0) {
        buffer.resize(elements_to_read);
        MPI_Status status;
        size_t bytes_to_read = elements_to_read * sizeof(T);
        int err = MPI_File_read_at(handle, position, buffer.data(), bytes_to_read, MPI_BYTE, &status);
        check_mpi_error(err);
        int bytes_read;
        MPI_Get_count(&status, MPI_BYTE, &bytes_read);
        size_t elements_read = static_cast<size_t>(bytes_read) / sizeof(T);
        buffer.resize(elements_read);
        return elements_read;
    }

    /**
     * @brief
     *
     * @tparam T the type of elements to read
     * @param buffer the read buffer (will be resized)
     * @param elements_to_read number of elements to read
     * @param position the file position to start from
     *
     * @return the actual number of read elements
     */
    template <typename T>
    size_t read_collective(std::vector<T>& buffer, size_t elements_to_read, size_t position = 0) {
        buffer.resize(elements_to_read);
        MPI_Status status;
        size_t bytes_to_read = elements_to_read * sizeof(T);
        int err = MPI_File_read_at_all(handle, position, buffer.data(), bytes_to_read, MPI_BYTE, &status);
        check_mpi_error(err);
        int bytes_read;
        MPI_Get_count(&status, MPI_BYTE, &bytes_read);
        size_t elements_read = static_cast<size_t>(bytes_read) / sizeof(T);
        buffer.resize(elements_read);
        return elements_read;
    }

    template <typename T>
    size_t write(const std::vector<T>& buffer, size_t position = 0) {
        MPI_Status status;
        size_t bytes_to_write = buffer.size() * sizeof(T);
        int err = MPI_File_write_at(handle, position, buffer.data(), bytes_to_write, MPI_BYTE, &status);
        check_mpi_error(err);
        int bytes_written;
        MPI_Get_count(&status, MPI_BYTE, &bytes_written);
        size_t elements_written = static_cast<size_t>(bytes_written) / sizeof(T);
        return elements_written;
    }

    template <typename T>
    size_t write_collective(const std::vector<T>& buffer, size_t position = 0) {
        MPI_Status status;
        size_t bytes_to_write = buffer.size() * sizeof(T);
        int err = MPI_File_write_at_all(handle, position, buffer.data(), bytes_to_write, MPI_BYTE, &status);
        check_mpi_error(err);
        int bytes_written;
        MPI_Get_count(&status, MPI_BYTE, &bytes_written);
        size_t elements_written = static_cast<size_t>(bytes_written) / sizeof(T);
        return elements_written;
    }

    size_t size() {
        MPI_Offset filesize;
        MPI_File_get_size(handle, &filesize);
        return filesize;
    }

private:
    MPI_File handle;
};

inline std::vector<ConcurrentFile::AccessMode> operator|(const ConcurrentFile::AccessMode& lhs,
                                                         const ConcurrentFile::AccessMode& rhs) {
    return {lhs, rhs};
}
inline std::vector<ConcurrentFile::AccessMode> operator|(std::vector<ConcurrentFile::AccessMode>& lhs,
                                                         const ConcurrentFile::AccessMode& rhs) {
    lhs.emplace_back(rhs);
    return lhs;
}

}  // namespace cetric

#endif /* MPI_IO_WRAPPER_H */

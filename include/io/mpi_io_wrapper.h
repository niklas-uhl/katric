#ifndef MPI_IO_WRAPPER_H
#define MPI_IO_WRAPPER_H
#include <mpi.h>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
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
        check_mpi_error(err, __FILE__, __LINE__);
        blocksize = std::numeric_limits<int>::max();
        MPI_Type_contiguous(blocksize, MPI_BYTE, &page_type);
        MPI_Type_commit(&page_type);
    }

    virtual ~ConcurrentFile() {
        if (handle != nullptr) {
            int err = MPI_File_close(&handle);
            check_mpi_error(err, __FILE__, __LINE__);
            handle = nullptr;
        }
        MPI_Type_free(&page_type);
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
        size_t total_bytes_read;
        if (bytes_to_read <= blocksize) {
            int err = MPI_File_read_at(handle, position, buffer.data(), bytes_to_read, MPI_BYTE, &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int bytes_read;
            MPI_Get_count(&status, MPI_BYTE, &bytes_read);
            total_bytes_read = bytes_read;
        } else {
            size_t pages_to_read = bytes_to_read / blocksize;
            if (pages_to_read > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw MPIException("To many blocks in read");
            }
            int err = MPI_File_read_at(handle, position, buffer.data(), pages_to_read, page_type, &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int pages_read;
            MPI_Get_count(&status, page_type, &pages_read);
            size_t remaining_bytes = bytes_to_read % blocksize;
            position += pages_read * blocksize;
            std::byte* buffer_pointer = reinterpret_cast<std::byte*>(buffer.data()) + pages_read * blocksize;
            err = MPI_File_read_at(handle, position, buffer_pointer, remaining_bytes, MPI_BYTE, &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int bytes_read;
            MPI_Get_count(&status, MPI_BYTE, &bytes_read);
            total_bytes_read = pages_read * blocksize + bytes_read;
        }
        size_t elements_read = total_bytes_read / sizeof(T);
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
        size_t total_bytes_read;
        size_t pages_to_read = bytes_to_read / blocksize;
        if (pages_to_read > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw MPIException("To many blocks in read");
        }
        int err = MPI_File_read_at_all(handle, position, buffer.data(), pages_to_read, page_type, &status);
        check_mpi_error(err, __FILE__, __LINE__);
        int pages_read;
        MPI_Get_count(&status, page_type, &pages_read);
        size_t remaining_bytes = bytes_to_read % blocksize;
        position += pages_read * blocksize;
        std::byte* buffer_pointer = reinterpret_cast<std::byte*>(buffer.data()) + pages_read * blocksize;
        err = MPI_File_read_at_all(handle, position, buffer_pointer, remaining_bytes, MPI_BYTE, &status);
        check_mpi_error(err, __FILE__, __LINE__);
        int bytes_read;
        MPI_Get_count(&status, MPI_BYTE, &bytes_read);
        total_bytes_read = pages_read * blocksize + bytes_read;
        size_t elements_read = total_bytes_read / sizeof(T);
        buffer.resize(elements_read);
        return elements_read;
    }

    template <typename T>
    size_t write(const std::vector<T>& buffer, size_t position = 0) {
        MPI_Status status;
        size_t bytes_to_write = buffer.size() * sizeof(T);
        size_t total_bytes_written;
        if (bytes_to_write <= blocksize) {
            int err = MPI_File_write_at(handle, position, buffer.data(), bytes_to_write, MPI_BYTE, &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int bytes_written;
            MPI_Get_count(&status, MPI_BYTE, &bytes_written);
            total_bytes_written = bytes_written;
        } else {
            size_t pages_to_write = bytes_to_write / blocksize;
            if (pages_to_write > std::numeric_limits<int>::max()) {
                throw MPIException("To many blocks in write");
            }
            int err = MPI_File_write_at(handle, position, buffer.data(), pages_to_write, page_type, &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int pages_written;
            MPI_Get_count(&status, page_type, &pages_written);
            size_t remaining_bytes = bytes_to_write % blocksize;
            position += pages_written * blocksize;
            const std::byte* buffer_pointer =
                reinterpret_cast<const std::byte*>(buffer.data()) + pages_written * blocksize;
            err = MPI_File_write_at(handle, position, buffer_pointer, remaining_bytes, MPI_BYTE, &status);
            check_mpi_error(err, __FILE__, __LINE__);
            int bytes_read;
            MPI_Get_count(&status, MPI_BYTE, &bytes_read);
            total_bytes_written = pages_written * blocksize + bytes_read;
        }
        size_t elements_written = total_bytes_written / sizeof(T);
        return elements_written;
    }

    template <typename T>
    size_t write_collective(const std::vector<T>& buffer, size_t position = 0) {
        // MPI_Status status;
        // size_t bytes_to_write = buffer.size() * sizeof(T);
        // int err = MPI_File_write_at_all(handle, position, buffer.data(), bytes_to_write, MPI_BYTE, &status);
        // check_mpi_error(err, __FILE__, __LINE__);
        // int bytes_written;
        // MPI_Get_count(&status, MPI_BYTE, &bytes_written);
        // size_t elements_written = static_cast<size_t>(bytes_written) / sizeof(T);
        // return elements_written;
        MPI_Status status;
        size_t bytes_to_write = buffer.size() * sizeof(T);
        size_t total_bytes_written;
        size_t pages_to_write = bytes_to_write / blocksize;
        if (pages_to_write > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw MPIException("To many blocks in write");
        }
        int err = MPI_File_write_at_all(handle, position, buffer.data(), pages_to_write, page_type, &status);
        check_mpi_error(err, __FILE__, __LINE__);
        int pages_written;
        MPI_Get_count(&status, page_type, &pages_written);
        size_t remaining_bytes = bytes_to_write % blocksize;
        position += pages_written * blocksize;
        const std::byte* buffer_pointer = reinterpret_cast<const std::byte*>(buffer.data()) + pages_written * blocksize;
        err = MPI_File_write_at_all(handle, position, buffer_pointer, remaining_bytes, MPI_BYTE, &status);
        check_mpi_error(err, __FILE__, __LINE__);
        int bytes_read;
        MPI_Get_count(&status, MPI_BYTE, &bytes_read);
        total_bytes_written = pages_written * blocksize + bytes_read;
        size_t elements_written = total_bytes_written / sizeof(T);
        return elements_written;
    }

    size_t size() {
        MPI_Offset filesize;
        MPI_File_get_size(handle, &filesize);
        return filesize;
    }

private:
    MPI_File handle;
    MPI_Datatype page_type;
    size_t blocksize;
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

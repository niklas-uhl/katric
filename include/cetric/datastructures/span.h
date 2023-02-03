#pragma once

#include <memory>
#include <vector>

#include <kassert/kassert.hpp>

namespace cetric {
template <typename T>
class SharedVectorSpan {
public:
    using iterator = typename std::vector<T>::iterator;
    SharedVectorSpan() : SharedVectorSpan(std::make_shared<std::vector<T>>(), 0, 0) {}
    SharedVectorSpan(std::shared_ptr<std::vector<T>> ptr, size_t begin, size_t end)
        : ptr_(std::move(ptr)),
          begin_(begin),
          end_(end) {
        KASSERT(begin <= end);
    }

    SharedVectorSpan<T> subspan(size_t begin, size_t end) {
        return SharedVectorSpan<T>(ptr_, begin_ + begin, begin_ + end);
    }
    size_t size() const {
        return end_ - begin_;
    }

    iterator begin() const {
        return ptr_->begin() + begin_;
    }

    iterator end() const {
        return ptr_->begin() + end_;
    }

private:
    std::shared_ptr<std::vector<T>> ptr_;
    size_t                          begin_;
    size_t                          end_;
};
} // namespace cetric

#pragma once

#include <memory>
#include <vector>

namespace cetric {
template <typename T>
class SharedVectorSpan {
public:
    using iterator = typename std::vector<T>::iterator;
    SharedVectorSpan(std::shared_ptr<std::vector<T>> ptr, size_t begin, size_t end)
        : ptr_(std::move(ptr)),
          begin_(begin),
          end_(end) {}

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

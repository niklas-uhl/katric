/*
 * Copyright (c) 2020-2023 Tim Niklas Uhl
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

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

/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.flag4j.core;

/**
 * This interface specifies methods which provide properties of a vector. All vectors should implement this interface.
 *
 * @param <T> Vector type.
 * @param <U> Dense Vector type.
 * @param <V> Sparse Vector type.
 * @param <W> Complex Vector type.
 * @param <Y> Real Vector type.
 * @param <X> Vector entry type.
 */
public interface VectorPropertiesMixin {

    /**
     * Gets the length of a vector.
     * @return The length, i.e. the number of entries, in this vector.
     */
    int length();


    // This method is specified here and in TensorPropertiesMixin for compatibility purposes in generic methods.
    /**
     * Finds the maximum value, in absolute value, in this tensor. If this tensor is complex, then this method is equivalent
     * to {@link TensorPropertiesMixin#max()}.
     * @return The maximum value, in absolute value, in this tensor.
     */
    double maxAbs();
}

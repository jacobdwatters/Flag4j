/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.arrays.backend_new;

import org.flag4j.arrays.Shape;
import org.flag4j.util.ArrayUtils;

import java.util.List;

/**
 * <p>Data class for storing information for a sparse COO vector.
 * <p>This record stores two lists: the non-zero entries and the indices of the vector.
 *
 * @param shape Shape of the vector.
 * @param entries Non-zero entries of the sparse COO vector.
 * @param indices Non-zero indices of the sparse COO vector.
 */
public record SparseVectorData<T>(Shape shape, List<T> entries, List<Integer> indices) {


    /**
     * Converts the indices of this sparse vector data to a primitive integer array.
     * @return A primitive integer array containing the indices of this sparse vector data.
     */
    public int[] indicesToArray() {
        return ArrayUtils.fromIntegerList(indices);
    }
}

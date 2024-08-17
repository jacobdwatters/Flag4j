/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.core.dense_base;

import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.arrays_old.sparse.CooVector;


/**
 * Interface which specifies methods which any dense vector should implement.
 */
public interface DenseVectorMixin {

    /**
     * Computes the element-wise addition between this vector and the specified vector and stores the result
     * in this vector.
     * @param B VectorOld to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    void addEq(VectorOld B);


    /**
     * Computes the element-wise addition between this vector and the specified vector and stores the result
     * in this vector.
     * @param B VectorOld to add to this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    void addEq(CooVector B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector and stores the result
     * in this vector.
     * @param B VectorOld to subtract this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    void subEq(VectorOld B);


    /**
     * Computes the element-wise subtraction between this vector and the specified vector and stores the result
     * in this vector.
     * @param B VectorOld to subtract this vector.
     * @throws IllegalArgumentException If this vector and the specified vector have different lengths.
     */
    void subEq(CooVector B);
}

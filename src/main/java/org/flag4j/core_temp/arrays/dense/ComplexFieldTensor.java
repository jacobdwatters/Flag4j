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

package org.flag4j.core_temp.arrays.dense;

import org.flag4j.core.Shape;
import org.flag4j.core_temp.ComplexOpsMixin;
import org.flag4j.core_temp.structures.fields.ComplexField;
import org.flag4j.operations_old.TransposeDispatcher;


/**
 * A tensor whose entries are comeplex-valued field elements.
 * @param <T> Type of the comeplex-valued field element for the tensor.
 */
public class ComplexFieldTensor<T extends ComplexField<T>> extends FieldTensor<T> implements ComplexOpsMixin<ComplexFieldTensor<T>> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public ComplexFieldTensor(Shape shape, T[] entries) {
        super(shape, entries);
    }


    /**
     * Computes the Hermitian transpose of a tensor by exchanging and conjugating the first and last axes of this tensor.
     *
     * @return The Hermitian transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public ComplexFieldTensor<T> H() {
        return H(0, getRank()-1);
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor acording to the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #H()
     * @see #H(int...)
     */
    @Override
    public ComplexFieldTensor<T> H(int axis1, int axis2) {
        return TransposeDispatcher.dispatchTensorHermitian(this, axis1, axis2);
    }


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {1, 2, 3, ... N-1}}.
     * @see #H(int, int)
     * @see #H()
     */
    @Override
    public ComplexFieldTensor<T> H(int... axes) {
        return TransposeDispatcher.dispatchTensorHermitian(this, axes);
    }


    /**
     * Comuputes the element-wise conjugation of this tensor.
     *
     * @return The element-wise conjugation of this tensor.
     */
    @Override
    public ComplexFieldTensor<T> conj() {
        ComplexField<T>[] conj = new ComplexField[entries.length];

        for(int i=0, size=entries.length; i<size; i++) {
            conj[i] = entries[i].conj();
        }

        return new ComplexFieldTensor(shape, conj);
    }
}

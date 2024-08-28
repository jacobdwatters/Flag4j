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
import org.flag4j.core_temp.arrays.sparse.CooFieldVector;
import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;
import java.util.List;

/**
 * <p>A dense vector whose entries are {@link Field field} elements.</p>
 *
 * <p>Vectors are 1D tensors (i.e. rank 1 tensor).</p>
 *
 * <p>FieldVectors have mutable entries but a fixed size.</p>
 *
 * @param <T> Type of the field element for the vector.
 */
public class FieldVector<T extends Field<T>>
        extends DenseFieldVectorBase<FieldVector<T>, FieldMatrix<T>, CooFieldVector<T>, T> {


    /**
     * Creates a vector with the specified entries.
     *
     * @param entries Entries of this vector.
     */
    public FieldVector(T... entries) {
        super(new Shape(entries.length), entries);
    }


    /**
     * Creates a vector with the specified size filled with the {@code fillValue}.
     *
     * @param size
     * @param fillValue Value to fill this vector with.
     */
    public FieldVector(int size, T fillValue) {
        super(new Shape(size), (T[]) new Field[size]);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a vector with the specified size filled with the {@code fillValue}.
     *
     * @param size
     * @param fillValue Value to fill this vector with.
     */
    @Override
    public FieldVector<T> makeLikeTensor(int size, T fillValue) {
        return new FieldVector<T>(size, fillValue);
    }


    /**
     * Creates a vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    @Override
    public FieldVector<T> makeLikeTensor(T... entries) {
        return new FieldVector<T>(entries);
    }


    /**
     * Constructs a matrix of similar type to this vector with the specified {@code shape} and {@code entries}.
     *
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix to construct.
     *
     * @return A matrix of similar type to this vector with the specified {@code shape} and {@code entries}.
     */
    @Override
    public FieldMatrix<T> makeLikeMatrix(Shape shape, T[] entries) {
        return new FieldMatrix<>(shape, entries);
    }


    /**
     * Constructs a sparse vector of similar type to this dense vector.
     *
     * @param size The size of the sparse vector.
     * @param entries The non-zero entries of the sparse vector.
     * @param indices The non-zero indices of the sparse vector.
     *
     * @return A sparse vector of similar type to this dense vector with the specified size, entries, and indices.
     */
    @Override
    public CooFieldVector<T> makeSparseVector(int size, List<T> entries, List<Integer> indices) {
        return new CooFieldVector<T>(size, entries, indices);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and entries.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and entries.
     */
    @Override
    public FieldVector<T> makeLikeTensor(Shape shape, T[] entries) {
        ParameterChecks.ensureEquals(shape.totalEntriesIntValueExact(), entries.length);
        ParameterChecks.ensureRank(shape, 1);
        return new FieldVector<T>(entries);
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return True if the two vectors have the same shape, are numerically equivalent, and are of type {@link FieldVector}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        FieldVector<T> src2 = (FieldVector<T>) object;

        return shape.equals(src2.shape) && Arrays.equals(entries, src2.entries);
    }
}

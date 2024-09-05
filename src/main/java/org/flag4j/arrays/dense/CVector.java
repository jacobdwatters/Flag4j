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

package org.flag4j.arrays.dense;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DenseFieldVectorBase;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.util.ParameterChecks;

import java.util.Arrays;
import java.util.List;


/**
 * <p>A complex dense vector whose entries are {@link Complex128}'s.</p>
 *
 * <p>A vector is essentially equivalent to a rank 1 tensor but has some extended functionality and may have improved performance
 * for some operations.</p>
 *
 * <p>CVector's have mutable entries but a fixed size.</p>
 */
public class CVector extends DenseFieldVectorBase<CVector, CMatrix, CooCVector, Complex128> {

    /**
     * Creates a complex vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(Complex128... entries) {
        super(new Shape(entries.length), entries);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, Complex128 fillValue) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(entries, fillValue);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex zero vector with the specified {@code size}.
     * @param size The size of the vector.
     */
    public CVector(int size) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(entries, Complex128.ZERO);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs an empty complex vector with the specified {@code size}. The entries of the resulting vector will be {@code null}.
     * @param size The size of the vector.
     * @return An empty complex vector with the specified {@code size}.
     */
    public static CVector getEmpty(int size) {
        return new CVector(new Complex128[size]);
    }


    /**
     * Creates a vector with the specified size filled with the {@code fillValue}.
     *
     * @param size
     * @param fillValue Value to fill this vector with.
     */
    @Override
    public CVector makeLikeTensor(int size, Complex128 fillValue) {
        return new CVector(size, fillValue);
    }


    /**
     * Creates a vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    @Override
    public CVector makeLikeTensor(Complex128... entries) {
        return new CVector(entries);
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
    public CMatrix makeLikeMatrix(Shape shape, Complex128[] entries) {
        return new CMatrix(shape, entries);
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
    public CooCVector makeSparseVector(int size, List<Complex128> entries, List<Integer> indices) {
        return new CooCVector(size, entries, indices);
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
    public CVector makeLikeTensor(Shape shape, Complex128[] entries) {
        ParameterChecks.ensureRank(shape, 1);
        ParameterChecks.ensureEquals(shape.totalEntriesIntValueExact(), entries.length);
        return new CVector(entries);
    }


    /**
     * Computes the inner product between this vector and itself.
     *
     * @return The inner product between this vector and itself.
     */
    public double innerSelf() {
        double inner = 0;
        for(Complex128 value : entries)
            inner += (value.re*value.re + value.im*value.im);

        return inner;
    }


    /**
     * Converts this complex vector to a real vector. This is done by ignoring the imaginary component of all entries.
     * @return A real vector containing the real components of this complex vectors entries.
     */
    public Vector toReal() {
        double[] real = new double[entries.length];
        for(int i=0, size=entries.length; i<size; i++)
            real[i] = entries[i].re;

        return new Vector(shape, real);
    }
}

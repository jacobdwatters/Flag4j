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
import org.flag4j.algebraic_structures.fields.Complex64;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend_new.field.AbstractDenseFieldVector;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.io.PrintOptions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

public class CVector extends AbstractDenseFieldVector<CVector, CMatrix, Complex128> {


    /**
     * Creates a complex vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(Field<Complex128>... entries) {
        super(new Shape(entries.length), entries);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(Complex64... entries) {
        super(new Shape(entries.length), ArrayUtils.wrapAsComplex128(entries, null));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(double... entries) {
        super(new Shape(entries.length), ArrayUtils.wrapAsComplex128(entries, null));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    public CVector(int... entries) {
        super(new Shape(entries.length), ArrayUtils.wrapAsComplex128(entries, null));
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
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, Complex64 fillValue) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(entries, new Complex128(fillValue));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex vector with the specified {@code size} and filled with {@code fillValue}.
     * @param size The size of the vector.
     * @param fillValue The value to fill the vector with.
     */
    public CVector(int size, double fillValue) {
        super(new Shape(size), new Complex128[size]);
        Arrays.fill(entries, new Complex128(fillValue));
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
     * Creates a copy of the provided vector.
     * @param vector Vector to create a copy of.
     */
    public CVector(CVector vector) {
        super(vector.shape, vector.entries.clone());
        setZeroElement(Complex128.ZERO);
    }
    
    
    /**
     * Constructs a dense vector with the specified {@code entries} of the same type as the vector.
     *
     * @param entries Entries of the dense vector to construct.
     */
    @Override
    public CVector makeLikeTensor(Field<Complex128>[] entries) {
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
    public CMatrix makeLikeMatrix(Shape shape, Field<Complex128>[] entries) {
        return new CMatrix(shape, entries);
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero entries of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooCTensor makeLikeCooTensor(Shape shape, Field<Complex128>[] entries, int[][] indices) {
        return new CooCTensor(shape, entries, indices);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code entries}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code entries}.
     */
    @Override
    public CVector makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        ValidateParameters.ensureRank(shape, 1);
        ValidateParameters.ensureEquals(shape.totalEntriesIntValueExact(), entries.length);
        return new CVector(entries);
    }


    /**
     * Converts this complex matrix to a real matrix. This conversion is done by taking the real component of each entry and
     * ignoring the imaginary component.
     * @return A real matrix containing the real components of the entries of this matrix.
     */
    public Matrix toReal() {
        double[] re = new double[entries.length];

        for(int i=0, size=entries.length; i<size; i++)
            re[i] = ((Complex128) entries[i]).re;

        return new Matrix(shape, re);
    }


    /**
     * Checks if an object is equal to this vector object.
     * @param object Object to check equality with this vector.
     * @return {@code true} if the two vectors have the same shape, are numerically equivalent, and are of type {@link CVector}.
     * {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CVector src2 = (CVector) object;

        return shape.equals(src2.shape) && Arrays.equals(entries, src2.entries);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(entries);

        return hash;
    }


    /**
     * Converts this vector to a human-readable string format. To specify the maximum number of entries to print, use
     * {@link PrintOptions#setMaxColumns(int)}.
     * @return A human-readable string representation of this vector.
     */
    public String toString() {
        StringBuilder result = new StringBuilder("shape: ").append(shape).append("\n");
        result.append("[");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        // Get entries up until the stopping point.
        for(int i=0; i<stopIndex; i++) {
            value = StringUtils.ValueOfRound(entries[i], PrintOptions.getPrecision());
            width = PrintOptions.getPadding() + value.length();
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        if(stopIndex < size-1) {
            width = PrintOptions.getPadding() + 3;
            value = "...";
            value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        // Get last entry now
        value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}

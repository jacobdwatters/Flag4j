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


import org.flag4j.algebraic_structures.fields.Complex64;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DenseFieldTensorBase;
import org.flag4j.arrays.sparse.CooCTensor64;
import org.flag4j.operations.common.complex.Complex64Operations;
import org.flag4j.util.Flag4jConstants;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * <p>A dense complex tensor backed by an array of {@link org.flag4j.algebraic_structures.fields.Complex128}'s.</p>
 *
 * <p>The {@link #entries} of a tensor are mutable but the {@link #shape} is fixed.</p>
 */
public class CTensor64 extends DenseFieldTensorBase<CTensor64, CooCTensor64, Complex64> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public CTensor64(Shape shape, Field<Complex64>[] entries) {
        super(shape, entries);
        if(super.entries.length == 0 || super.entries[0] == null) setZeroElement(Complex64.ZERO);
    }


    /**
     * Creates a zero tensor with the specified shape.
     *
     * @param shape Shape of this tensor.
     */
    public CTensor64(Shape shape) {
        super(shape, new Complex64[shape.totalEntriesIntValueExact()]);
        Arrays.fill(entries, Complex64.ZERO);
        setZeroElement(Complex64.ZERO);
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor64(Shape shape, Complex64 fillValue) {
        super(shape, new Complex64[shape.totalEntries().intValueExact()]);
        Arrays.fill(entries, fillValue);
        setZeroElement(Complex64.ZERO);
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with. Must be a string representation of a complex number.
     */
    public CTensor64(Shape shape, String fillValue) {
        super(shape, new Complex64[shape.totalEntries().intValueExact()]);
        Arrays.fill(entries, new Complex64(fillValue));
        setZeroElement(Complex64.ZERO);
    }


    /**
     * <p>Creates a tensor with the specified entries and shape.</p>
     * <p>String array must contain properly formatted string representation of complex numbers.</p>
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     * @throws org.flag4j.util.exceptions.ComplexNumberParsingException If any entry in {@code entries} is not a properly formatted
     * string representation of complex number.
     */
    public CTensor64(Shape shape, String[] entries) {
        super(shape, new Complex64[entries.length]);
        setZeroElement(Complex64.ZERO);

        // Parse string values.
        for(int i=0, size=entries.length; i<size; i++)
            super.entries[i] = new Complex64(entries[i]);
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
    public CTensor64 makeLikeTensor(Shape shape, Field<Complex64>[] entries) {
        return new CTensor64(shape, entries);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooCTensor64 toCoo() {
        List<Field<Complex64>> spEntries = new ArrayList<>();
        List<int[]> indices = new ArrayList<>();

        int size = entries.length;
        Field<Complex64> value;

        for(int i=0; i<size; i++) {
            value = entries[i];

            if(value.isZero()) {
                spEntries.add(value);
                indices.add(shape.getIndices(i));
            }
        }

        return new CooCTensor64(shape, spEntries, indices);
    }


    /**
     * Rounds this tensor to the nearest whole number. If the tensor is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @return A copy of this tensor with each entry rounded to the nearest whole number.
     */
    public CTensor64 round() {
        return round(0);
    }


    /**
     * Rounds a matrix to the nearest whole number. If the matrix is complex, both the real and imaginary component will
     * be rounded independently.
     *
     * @param precision The number of decimal places to round to. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If <code>precision</code> is negative.
     */
    public CTensor64 round(int precision) {
        return new CTensor64(this.shape, Complex64Operations.round(this.entries, precision));
    }


    /**
     * Rounds values which are close to zero in absolute value to zero. If the tensor is complex, both the real and imaginary components will be rounded
     * independently. By default, the values must be within {@link Flag4jConstants#EPS_F32} of zero. To specify a threshold value see
     * {@link #roundToZero(double)}.
     *
     * @return A copy of this tensor with rounded values.
     */
    public CTensor64 roundToZero() {
        return roundToZero(Flag4jConstants.EPS_F32);
    }


    /**
     * Rounds values which are close to zero in absolute value to zero.
     *
     * @param threshold Threshold for rounding values to zero. That is, if a value in this matrix is less than the threshold in absolute value then it
     *                  will be rounded to zero. This value must be non-negative.
     * @return A copy of this matrix with rounded values.
     * @throws IllegalArgumentException If threshold is negative.
     */
    public CTensor64 roundToZero(double threshold) {
        return new CTensor64(this.shape, Complex64Operations.roundToZero(this.entries, (float) threshold));
    }
}

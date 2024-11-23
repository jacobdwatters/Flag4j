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
import org.flag4j.arrays.backend.field.AbstractDenseFieldTensor;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.io.PrintOptions;
import org.flag4j.io.parsing.ComplexNumberParser;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;

import java.util.Arrays;


/**
 * <p>A dense complex tensor backed by an array of {@link Complex128}'s.</p>
 *
 * <p>The {@link #entries} of a tensor are mutable but the {@link #shape} is fixed.</p>
 */
public class CTensor extends AbstractDenseFieldTensor<CTensor, Complex128> {


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public CTensor(Shape shape, Field<Complex128>[] entries) {
        super(shape, entries);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public CTensor(Shape shape, Complex64[] entries) {
        super(shape, new Complex128[entries.length]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);

        tensorDot(this, new int[3], new int[3]);

        for(int i=0, size=entries.length; i<size; i++)
            this.entries[i] = new Complex128(entries[i]);
    }


    /**
     * Creates a zero tensor with the specified shape.
     *
     * @param shape Shape of this tensor.
     */
    public CTensor(Shape shape) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, Complex128.ZERO);
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor(Shape shape, Complex128 fillValue) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor(Shape shape, Complex64 fillValue) {
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public CTensor(Shape shape, double fillValue) {
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates a tensor with the specified shape and filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with. Must be a string representation of a complex number parsable by 
     * {@link ComplexNumberParser#parseNumberToComplex128(String)}.
     */
    public CTensor(Shape shape, String fillValue) {
        super(shape, new Complex128[shape.totalEntries().intValueExact()]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
        Arrays.fill(entries, new Complex128(fillValue));
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. Each value in {@code entries} must be formated as a complex number such as:
     * <ul>
     *     <li>"a"</li>
     *     <li>"a + bi", "a - bi", "a + i", or "a - i"</li>
     *     <li>"bi", "i", or "-i"</li>
     * </ul>
     *
     * where "a" and "b" are integers or decimal numbers and white space does not matter.
     */
    public CTensor(Shape shape, String[] entries) {
        super(shape, new Complex128[entries.length]);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);

        for(int i=0, size=entries.length; i<size; i++)
            this.entries[i] = new Complex128(entries[i]);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public CTensor(Shape shape, double[] entries) {
        super(shape, ArrayUtils.wrapAsComplex128(entries, null));
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a copy of the specified tensor.
     * @param tensor Tensor to create copy of.
     */
    public CTensor(CTensor tensor) {
        super(tensor.shape, (Complex128[]) tensor.entries);
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
    public CTensor makeLikeTensor(Shape shape, Field<Complex128>[] entries) {
        return new CTensor(shape, entries);
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
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link CTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CTensor src2 = (CTensor) object;

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
     * Formats this tensor as a human-readable string. Specifically, a string containing the
     * shape and flattened entries of this tensor.
     * @return A human-readable string representing this tensor.
     */
    public String toString() {
        int size = shape.totalEntries().intValueExact();
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("[");

        int stopIndex = Math.min(PrintOptions.getMaxColumns()-1, size-1);
        int width;
        String value;

        // Get entries up until the stopping point.
        for(int i=0; i<stopIndex; i++) {
            value = StringUtils.ValueOfRound((Complex128) entries[i], PrintOptions.getPrecision());
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
        value = StringUtils.ValueOfRound((Complex128) entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}

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

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend_new.field.AbstractDenseFieldVector;
import org.flag4j.arrays.sparse.CooFieldVector;
import org.flag4j.io.PrintOptions;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * <p>A dense vector whose entries are {@link Field field} elements.</p>
 *
 * <p>Vectors are 1D tensors (i.e. rank 1 tensor).</p>
 *
 * <p>FieldVectors have mutable entries but a fixed size.</p>
 *
 * @param <T> Type of the field element for the vector.
 */
public class FieldVector<T extends Field<T>> extends AbstractDenseFieldVector<FieldVector<T>, FieldMatrix<T>, T> {


    /**
     * Creates a vector with the specified entries.
     *
     * @param entries Entries of this vector.
     */
    public FieldVector(Field<T>... entries) {
        super(new Shape(entries.length), entries);
    }


    /**
     * Creates a vector with the specified size filled with the {@code fillValue}.
     *
     * @param size
     * @param fillValue Value to fill this vector with.
     */
    public FieldVector(int size, T fillValue) {
        super(new Shape(size), new Field[size]);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a vector with the specified {@code entries}.
     *
     * @param entries Entries of this vector.
     */
    @Override
    public FieldVector<T> makeLikeTensor(Field<T>... entries) {
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
    public FieldMatrix<T> makeLikeMatrix(Shape shape, Field<T>[] entries) {
        return new FieldMatrix<>(shape, entries);
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
    public FieldVector<T> makeLikeTensor(Shape shape, Field<T>[] entries) {
        ValidateParameters.ensureEquals(shape.totalEntriesIntValueExact(), entries.length);
        ValidateParameters.ensureRank(shape, 1);
        return new FieldVector<T>(entries);
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
    protected CooFieldVector<T> makeLikeCooTensor(Shape shape, Field<T>[] entries, int[][] indices) {
        return new CooFieldVector<>(shape.totalEntriesIntValueExact(), entries, indices[0]);
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

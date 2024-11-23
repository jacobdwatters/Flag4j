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
import org.flag4j.arrays.backend.field.AbstractDenseFieldTensor;
import org.flag4j.arrays.sparse.CooFieldTensor;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.operations.dense.field_ops.DenseFieldEquals;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;


/**
 * <p>A dense tensor whose entries are {@code Field} elements.</p>
 *
 * <p>The {@link #entries} of a field tensor are mutable but the {@link #shape} is fixed.</p>
 *
 * @param <T> Type of the field element for the tensor.
 */
public class FieldTensor<T extends Field<T>> extends AbstractDenseFieldTensor<FieldTensor<T>, T> {

    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all entries within the tensor.
     * If this tensor is sparse, this specifies only the non-zero entries of the tensor.
     */
    public FieldTensor(Shape shape, Field<T>[] entries) {
        super(shape, entries);
    }


    /**
     * Creates a tensor with the specified shape filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill tensor with.
     */
    public FieldTensor(Shape shape, T fillValue) {
        super(shape, new Field[shape.totalEntries().intValueExact()]);
        Arrays.fill(entries, fillValue);
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
    protected CooFieldTensor<T> makeLikeCooTensor(Shape shape, Field<T>[] entries, int[][] indices) {
        return new CooFieldTensor<>(shape, entries, indices);
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
    public FieldTensor<T> makeLikeTensor(Shape shape, Field<T>[] entries) {
        return new FieldTensor<T>(shape, entries);
    }


    /**
     * Converts this tensor to an equivalent vector. If this tensor is not rank 1, then it will be flattened.
     * @return A vector equivalent of this tensor.
     */
    public FieldVector<T> toVector() {
        return new FieldVector<T>(this.entries.clone());
    }


    /**
     * Converts this tensor to a matrix with the specified shape.
     * @param matShape Shape of the resulting matrix. Must be {@link ValidateParameters#ensureBroadcastable(Shape, Shape) broadcastable}
     * with the shape of this tensor.
     * @return A matrix of shape {@code matShape} with the values of this tensor.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If {@code matShape} is not of rank 2.
     */
    public FieldMatrix<T> toMatrix(Shape matShape) {
        ValidateParameters.ensureBroadcastable(shape, matShape);
        ValidateParameters.ensureRank(matShape, 2);

        return new FieldMatrix<T>(matShape, entries.clone());
    }


    /**
     * Converts this tensor to an equivalent matrix.
     * @return If this tensor is rank 2, then the equivalent matrix will be returned.
     * If the tensor is rank 1, then a matrix with a single row will be returned. If the rank of this tensor is larger than 2, it will
     * be flattened to a single row.
     */
    public FieldMatrix<T> toMatrix() {
        FieldMatrix<T> mat;

        if(this.getRank()==2) {
            mat = new FieldMatrix<T>(this.shape, this.entries.clone());
        } else {
            mat = new FieldMatrix<T>(1, this.entries.length, this.entries.clone());
        }

        return mat;
    }


    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link FieldTensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        FieldTensor<T> src2 = (FieldTensor<T>) object;

        return DenseFieldEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
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

        // Get last entry.
        value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}

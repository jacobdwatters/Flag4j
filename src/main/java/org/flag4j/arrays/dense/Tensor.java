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


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.primitive.AbstractDenseDoubleTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.operations.dense.real.RealDenseEquals;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>A real dense tensor backed by a primitive double array.</p>
 *
 * <p>A tensor is a multidimensional array. If N indices are required to uniquely identify all elements of a tensor, then the
 * tensor is considered an N-dimensional tensor/array or a rank-N tensor.</p>
 *
 * <p>The {@link #entries} of a Tensor are mutable but the {@link #shape} is fixed.</p>
 */
public class Tensor extends AbstractDenseDoubleTensor<Tensor> {

    /**
     * Creates a zero tensor with the shape.
     *
     * @param shape Shape of this tensor.
     */
    public Tensor(Shape shape) {
        super(shape, new double[shape.totalEntries().intValueExact()]);
    }


    /**
     * Creates a tensor with the specified shape filled with {@code fillValue}.
     *
     * @param shape Shape of this tensor.
     * @param fillValue Value to fill this tensor with.
     */
    public Tensor(Shape shape, double fillValue) {
        super(shape, new double[shape.totalEntries().intValueExact()]);
        Arrays.fill(entries, fillValue);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public Tensor(Shape shape, double[] entries) {
        super(shape, entries);
    }


    /**
     * Flattens tensor to single dimension while preserving order of entries.
     *
     * @return The flattened tensor.
     *
     * @see #flatten(int)
     */
    @Override
    public Tensor flatten() {
        return new Tensor(new Shape(shape.totalEntriesIntValueExact()), entries.clone());
    }


    /**
     * Flattens a tensor along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     *
     * @throws ArrayIndexOutOfBoundsException If the axis is not positive or larger than <code>this.{@link #getRank()}-1</code>.
     * @see #flatten()
     */
    @Override
    public Tensor flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        int[] dims = new int[rank];
        Arrays.fill(dims, 1);
        dims[axis] = shape.totalEntriesIntValueExact();
        return new Tensor(new Shape(dims), entries.clone());
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public Tensor(Shape shape, int[] entries) {
        super(shape, new double[entries.length]);
        ArrayUtils.asDouble(entries, this.entries);
    }


    /**
     * Constructs a copy of the specified tensor.
     * @param src The tensor to make a copy of.
     */
    public Tensor(Tensor src) {
        super(src.shape, src.entries.clone());
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public Tensor(Shape shape, Double[] entries) {
        super(shape, new double[entries.length]);
        ArrayUtils.unbox(entries, super.entries);
    }


    /**
     * Creates a tensor with the specified entries and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor.
     */
    public Tensor(Shape shape, Integer[] entries) {
        super(shape, new double[entries.length]);
        ArrayUtils.asDouble(entries, super.entries);
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
    public Tensor makeLikeTensor(Shape shape, double[] entries) {
        return new Tensor(shape, entries);
    }


    /**
     * Converts this tensor to an equivalent vector. If this vector is not rank-1 it will first be flattened then converted to a
     * vector.
     * @return A vector with entries equivalent to this vector.
     */
    public Vector toVector() {
        return new Vector(entries.clone());
    }


    /**
     * Converts this tensor to an equivalent matrix. If this matrix is not rank-2 it will first be flattened to a row vector then
     * converted to a matrix.
     * @return A matrix with entries equivalent to this tensor.
     */
    public Matrix toMatrix() {
        if(rank == 2) return new Matrix(shape, entries.clone());
        else return new Matrix(new Shape(1, entries.length), entries.clone());
    }


    /**
     * Converts this tensor to an equivalent matrix with the specified shape.
     * @param shape New shape for the matrix. Must be rank-2 and broadcastable to {@code this.shape}.
     * @return A matrix with the specified shape and entries equivalent to this tensor.
     * @throws IllegalArgumentException If {@code shape} is not broadcastable to {@code this.shape}.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code shape.getRank() != 2}.
     */
    public Matrix toMatrix(Shape shape) {
        // Matrix constructor checks the rank of the shape and
        // ensures that shape.totalEntriesIntValueExact() == entries.length.
        return new Matrix(shape, entries.clone());
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo(double)
     */
    public CooTensor toCoo() {
        return toCoo(0.9);
    }


    /**
     * Converts this tensor to an equivalent sparse COO tensor.
     * @param estimatedSparsity Estimated sparsity of the tensor. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <i>may</i> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * over-estimate the sparsity. If speed is the concern it is better to under-estimate the sparsity.
     * @return A sparse COO tensor that is equivalent to this dense tensor.
     * @see #toCoo(double)
     */
    public CooTensor toCoo(double estimatedSparsity) {
        ValidateParameters.ensureInRange(estimatedSparsity, 0.0, 1.0, "estimatedSparsity");
        int estimatedSize = (int) (entries.length*(1.0-estimatedSparsity));
        List<Double> cooEntries = new ArrayList<>(estimatedSize);
        List<int[]> cooIndices = new ArrayList<>(estimatedSize);
        final Double ZERO = Double.valueOf(0d);

        final int rows = shape.get(0);
        final int cols = shape.get(1);

        for(int i=0, size=entries.length; i<size; i++) {
            Double val = entries[i];

            if(val.equals(ZERO)) {
                cooEntries.add(val);
                cooIndices.add(shape.getNdIndices(i));
            }
        }

        return new CooTensor(shape, cooEntries, cooIndices);
    }

    /**
     * Checks if an object is equal to this tensor object.
     * @param object Object to check equality with this tensor.
     * @return True if the two tensors have the same shape, are numerically equivalent, and are of type {@link Tensor}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        Tensor src2 = (Tensor) object;

        return RealDenseEquals.tensorEquals(this.entries, this.shape, src2.entries, src2.shape);
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

        // Get last entry now
        value = StringUtils.ValueOfRound(entries[size-1], PrintOptions.getPrecision());
        width = PrintOptions.getPadding() + value.length();
        value = PrintOptions.useCentering() ? StringUtils.center(value, width) : value;
        result.append(String.format("%-" + width + "s", value));

        result.append("]");

        return result.toString();
    }
}
